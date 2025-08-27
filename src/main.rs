// Main module for Rig Analyzer with rig-sqlite integration
//
// This implementation uses rig-sqlite for vector similarity search
// + improved bug finding & false positive checks (parallel scan, .gitignore support,
// hybrid FP filtering, persistent FP cache, dedup, and tunable thresholds)

use anyhow::Result;
use tracing::{debug, error, info, trace, warn};

mod deepseek;
mod false_positive_filter;
mod mcp;
pub mod patterns;
mod vector_store;

use crate::false_positive_filter::FalsePositiveFilter;
use crate::false_positive_filter::{RigFpValidator, IssueView, Neighbor};
use rig_bug_finder::{init_dev_logging, Config};
use deepseek::DeepSeekClient;
use mcp::run_mcp_server;
use patterns::scan;
use vector_store::VectorStoreManager;

// ===== New imports for improvements =====
use blake3;
use futures::stream::{self, StreamExt};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::fs as aiofs;
use tokio::sync::{Mutex, Semaphore};

// ---------- FP Cache ----------
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FpCacheEntry {
    /// Stable fingerprint: pattern_id | relative_path | line | blake3(excerpt)
    issue_key: String,
    file_hash: String,         // blake3(file bytes)
    is_false_positive: bool,   // final decision
    confidence: f32,           // AI confidence (0.0..1.0) or heuristic 1.0
    decided_by: String,        // "heuristics" | "ai" | "mixed"
    reasoning: Option<String>, // why it was marked FP
    last_seen_ts: String,      // UTC timestamp string
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
struct FpCache {
    map: HashMap<String, FpCacheEntry>,
}

impl FpCache {
    async fn load(path: &Path) -> Self {
        if let Ok(bytes) = aiofs::read(path).await {
            if let Ok(cache) = serde_json::from_slice::<FpCache>(&bytes) {
                return cache;
            }
        }
        FpCache::default()
    }

    async fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            aiofs::create_dir_all(dir).await?;
        }
        let bytes = serde_json::to_vec_pretty(self)?;
        aiofs::write(path, bytes).await?;
        Ok(())
    }

    fn get(&self, key: &str) -> Option<&FpCacheEntry> {
        self.map.get(key)
    }

    fn put(&mut self, entry: FpCacheEntry) {
        self.map.insert(entry.issue_key.clone(), entry);
    }
}

// ---------- Utilities ----------
fn is_excluded_dir_component(c: &str) -> bool {
    matches!(
        c,
        ".git"
            | "target"
            | "node_modules"
            | "__pycache__"
            | "build"
            | "dist"
            | "tests"
            | "test"
            | "benches"
            | "benchmarks"
            | "benchmark"
            | "fixtures"
            | "mocks"
            | "vendor"
            | "third_party"
            | "generated"
            | "gen"
            | "out"
            | "tmp"
            | "cache"
    )
}

fn has_rust_extension(path: &Path) -> bool {
    match path.extension().and_then(|s| s.to_str()) {
        Some(ext) => Config::RUST_FILE_EXTENSIONS.contains(&ext),
        None => false,
    }
}

fn file_component_excluded(path: &Path) -> bool {
    path.components()
        .filter_map(|c| c.as_os_str().to_str())
        .any(is_excluded_dir_component)
}

fn is_probably_comment(snippet: &str) -> bool {
    let t = snippet.trim();
    t.starts_with("//") || t.starts_with("/*") || t.ends_with("*/")
}

fn blake3_hex(bytes: &[u8]) -> String {
    let hash = blake3::hash(bytes);
    hash.to_hex().to_string()
}

fn blake3_hex_of_str(s: &str) -> String {
    blake3_hex(s.as_bytes())
}

// Compute a stable fingerprint for an issue
fn issue_fingerprint(pattern_id: &str, relative_path: &str, line: usize, excerpt: &str) -> String {
    let ex_hash = blake3_hex_of_str(excerpt);
    format!("{pattern_id}|{relative_path}|{line}|{ex_hash}")
}

// Determine if we should trust AI FP result (hybrid guardrail)
fn ai_says_false_positive(confidence: f32, threshold: f32) -> bool {
    confidence >= threshold
}

/// Enhanced DeepSeek analysis for validated issues using multiple specialized methods
async fn enhance_issue_with_deepseek(
    deepseek_client: &DeepSeekClient,
    issue: &patterns::Issue,
    file_content: &str,
    file_path: &str,
) -> Result<String> {
    let context = format!("File: {}, Pattern: {} ({}), Severity: {}", 
                         file_path, issue.pattern_id, issue.category, issue.severity);

    // Run multiple specialized analyses concurrently for comprehensive enhancement
    let code_section = format!("// Code around line {}\n{}", issue.line, issue.excerpt);
    
    // Primary comprehensive analysis
    let enhancement_prompt = format!(
        "COMPREHENSIVE ISSUE ANALYSIS
==============================

File: {file_path}
Location: Line {line}, Column {col}
Pattern ID: {pattern_id}
Category: {category}
Severity: {severity}
Issue Name: {name}

Code Context:
```rust
{excerpt}
```

Full File Analysis Available: {file_size} characters

Please provide a comprehensive analysis including:

1. **SECURITY IMPACT**: Assess potential security vulnerabilities
2. **BUSINESS LOGIC RISK**: Impact on critical systems, numerical calculations, or data integrity
3. **PERFORMANCE IMPLICATIONS**: Memory usage, CPU impact, or scalability concerns
4. **FIX RECOMMENDATIONS**: Specific code changes with examples
5. **PREVENTION STRATEGIES**: How to avoid similar issues in the future
6. **TESTING SUGGESTIONS**: Unit tests or integration tests to verify the fix
7. **PRIORITY ASSESSMENT**: Immediate, urgent, or can be scheduled

Focus specifically on Rust best practices, memory safety, and overall application reliability.
Provide concrete, actionable recommendations.",
        file_path = file_path,
        line = issue.line,
        col = issue.col,
        pattern_id = issue.pattern_id,
        category = issue.category,
        severity = issue.severity,
        name = issue.name,
        excerpt = issue.excerpt,
        file_size = file_content.len()
    );

    // Get the primary analysis
    let primary_analysis = deepseek_client.analyze_code(&enhancement_prompt).await?;

    // Get specialized analyses based on issue category and severity
    let mut specialized_analyses = Vec::new();
    
    match issue.category.as_ref() {
        "Security" | "Authentication" | "Authorization" | "Cryptography" => {
            match deepseek_client.analyze_security(&code_section, &context).await {
                Ok(security_analysis) => {
                    specialized_analyses.push(format!("=== SPECIALIZED SECURITY ANALYSIS ===\n{}", security_analysis));
                }
                Err(e) => {
                    debug!("Security analysis failed: {}", e);
                }
            }
        }
        "Performance" | "Memory" | "Concurrency" => {
            match deepseek_client.analyze_performance(&code_section, &context).await {
                Ok(perf_analysis) => {
                    specialized_analyses.push(format!("=== SPECIALIZED PERFORMANCE ANALYSIS ===\n{}", perf_analysis));
                }
                Err(e) => {
                    debug!("Performance analysis failed: {}", e);
                }
            }
        }
        _ => {
            // For all other categories, run code quality analysis
            match deepseek_client.analyze_code_quality(&code_section, &context).await {
                Ok(quality_analysis) => {
                    specialized_analyses.push(format!("=== SPECIALIZED CODE QUALITY ANALYSIS ===\n{}", quality_analysis));
                }
                Err(e) => {
                    debug!("Code quality analysis failed: {}", e);
                }
            }
        }
    }

    // For Critical and High severity issues, always include architectural analysis
    if issue.severity == "Critical" || issue.severity == "High" {
        match deepseek_client.analyze_architecture(&code_section, &context).await {
            Ok(arch_analysis) => {
                specialized_analyses.push(format!("=== ARCHITECTURAL IMPACT ANALYSIS ===\n{}", arch_analysis));
            }
            Err(e) => {
                debug!("Architectural analysis failed: {}", e);
            }
        }

        // Generate test cases for critical issues
        match deepseek_client.generate_test_cases(&code_section, &context).await {
            Ok(test_cases) => {
                specialized_analyses.push(format!("=== GENERATED TEST CASES ===\n{}", test_cases));
            }
            Err(e) => {
                debug!("Test case generation failed: {}", e);
            }
        }
    }

    // Combine all analyses into a comprehensive report
    let mut final_analysis = format!(
        "=== COMPREHENSIVE DEEPSEEK ANALYSIS ===\n\
         Issue: {} ({})\n\
         Severity: {} | Category: {}\n\
         Location: {}:{}:{}\n\n\
         === PRIMARY ANALYSIS ===\n\
         {}\n",
        issue.name, issue.pattern_id, issue.severity, issue.category,
        file_path, issue.line, issue.col, primary_analysis
    );

    // Add specialized analyses
    for analysis in specialized_analyses {
        final_analysis.push_str(&format!("\n{}\n", analysis));
    }

    // Add metadata
    final_analysis.push_str(&format!(
        "\n=== ANALYSIS METADATA ===\n\
         - Generated: {}\n\
         - Analysis Methods: {} specialized methods\n\
         - Total Analysis Length: {} characters\n\
         - File Context: {} characters analyzed\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        if issue.severity == "Critical" || issue.severity == "High" { "4-5" } else { "2-3" },
        final_analysis.len(),
        file_content.len()
    ));

    Ok(final_analysis)
}

/// Scan repository directories for pattern matches (improved)
async fn scan_repository() -> Result<()> {
    info!("🔍 Scanning repository for CRITICAL and HIGH pattern matches (excluding Panic patterns and test files, with hybrid false positive filtering)...");
    let _start_time = std::time::Instant::now();

    // Focus scan on rig-core only (src/), while keeping relative paths from repo root
    let repo_root = Config::rig_repo_path();
    if !repo_root.exists() {
        warn!("Repository path not found: {}", repo_root.display());
        return Ok(());
    }
    // Primary scan root (rig-core under the repo root)
    let mut scan_root = repo_root.join("rig-core/src");
    if !scan_root.exists() {
        // Fallback: if REPO_PATH mistakenly points to a nested folder, try its parent
        if let Some(parent) = repo_root.parent() {
            let alt = parent.join("rig-core/src");
            if alt.exists() {
                info!(
                    "Adjusted scan root to parent rig-core: {} (REPO_PATH was: {})",
                    alt.display(),
                    repo_root.display()
                );
                scan_root = alt;
            } else {
                warn!(
                    "rig-core path not found: {} (REPO_PATH: {}), nothing to scan",
                    scan_root.display(),
                    repo_root.display()
                );
                return Ok(());
            }
        } else {
            warn!(
                "rig-core path not found and no parent to try (REPO_PATH: {})",
                repo_root.display()
            );
            return Ok(());
        }
    }
    info!(
        "Scanning rig-core only under repo root: repo_root={} scan_root={}",
        repo_root.display(),
        scan_root.display()
    );

    // Configuration constants
    let fp_confidence_threshold = 0.75;
    let deepseek_concurrency = 3;
    let scan_concurrency = {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        (cores * 2).min(16).max(4)
    };
    let max_file_size_bytes = 2 * 1024 * 1024; // 2MB

    // Rig FP validator configuration
    let borderline_low = 0.60;
    let borderline_high = 0.80;
    let context_radius = 24;
    let neighbors_k = 3;

    // Initialize Rig FP validator (always enabled)
    let rig_fp_validator = match RigFpValidator::new(
        fp_confidence_threshold,
        borderline_low,
        borderline_high,
        context_radius,
        neighbors_k,
    ) {
        Ok(validator) => {
            info!("✅ Rig FP validator initialized successfully");
            Some(Arc::new(validator))
        }
        Err(e) => {
            warn!("⚠️ Failed to initialize Rig FP validator: {}. Falling back to legacy FP filter.", e);
            None
        }
    };

    // Initialize false positive filter + rate limiter
    let deepseek_client = DeepSeekClient::from_env().ok();
    let ds_semaphore = Arc::new(Semaphore::new(deepseek_concurrency));

    // Cache Git info once to avoid repeated spawning
    let repo_branch = get_git_branch()
        .await
        .unwrap_or_else(|| "unknown".to_string());
    let commit_hash = get_git_commit_hash()
        .await
        .unwrap_or_else(|| "unknown".to_string());

    // Persistent FP cache
    let cache_path = Config::manifest_dir()
        .join(".rig_cache")
        .join("false_positive_cache.json");
    let fp_cache = Arc::new(Mutex::new(FpCache::load(&cache_path).await));

    // First pass: collect files to process (respect .gitignore)
    info!("🔍 Collecting files to scan (respecting .gitignore)...");
    let mut files_to_process: Vec<PathBuf> = Vec::new();

    let walker = WalkBuilder::new(&scan_root)
        .hidden(false)
        .git_ignore(true)
        .git_exclude(true)
        .follow_links(true)
        .build();

    for dent in walker {
        let Ok(entry) = dent else { continue };
        let path = entry.path();
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }
        if file_component_excluded(path) {
            continue;
        }
        if !has_rust_extension(path) {
            continue;
        }
        // Skip test files based on name patterns
        let path_str = path.to_string_lossy();
        if path_str.contains("_test.rs")
            || path_str.ends_with("_tests.rs")
            || path_str.contains("/test_")
            || path_str.contains("/tests/")
            || path_str.contains("/test/")
        {
            continue;
        }
        // Skip large files
        if let Ok(meta) = std::fs::metadata(path) {
            if meta.len() > max_file_size_bytes {
                trace!(
                    "Skipping large file (> {} B): {}",
                    max_file_size_bytes,
                    path.display()
                );
                continue;
            }
        }
        files_to_process.push(path.to_path_buf());
    }

    let total_files = files_to_process.len();
    info!("📊 Found {} Rust files to scan", total_files);
    if total_files == 0 {
        info!("✅ No files to scan.");
        return Ok(());
    }

    let total_issues = Arc::new(AtomicUsize::new(0));
    let scanned_files = Arc::new(AtomicUsize::new(0));
    let filtered_false_positives = Arc::new(AtomicUsize::new(0));

    // Dedup set for current run (avoid writing duplicate artifacts)
    let seen_issue_keys: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));

    // Process files in parallel with bounded concurrency
    let repo_root_arc = Arc::new(repo_root.clone());
    let info_every = std::cmp::min(50, std::cmp::max(1, total_files / 10));

    stream::iter(files_to_process.into_iter().enumerate())
        .map(|(file_index, path)| {
            let repo_root = repo_root_arc.clone();
            let fp_cache = fp_cache.clone();
            let total_issues = total_issues.clone();
            let scanned_files = scanned_files.clone();
            let filtered_false_positives = filtered_false_positives.clone();
            let seen_issue_keys = seen_issue_keys.clone();
            let ds_semaphore = ds_semaphore.clone();
            let deepseek_client = deepseek_client.clone();
            let rig_fp_validator = rig_fp_validator.clone(); // Clone the Arc<RigFpValidator>
            let mut fp_filter = FalsePositiveFilter::new(deepseek_client.clone());
            let repo_branch = repo_branch.clone();
            let commit_hash = commit_hash.clone();
            async move {
                let progress = file_index + 1;
                let remaining = total_files - progress;
                let percentage = ((progress as f64 / total_files as f64) * 100.0) as u32;

                // Compute relative path (from repo root) once and reuse it everywhere below
                let relative_path = pathdiff::diff_paths(&path, &*repo_root).unwrap_or(path.clone());
                let relative_path_str = relative_path.to_string_lossy().to_string();

                debug!(
                    "📄 [{}/{}] ({:3}%) Scanning: {} (📁 {} remaining)",
                    progress,
                    total_files,
                    percentage,
                    &relative_path_str,
                    remaining
                );
                trace!("Absolute path: {}", path.display());

                if progress % info_every == 0 || progress == total_files {
                    info!(
                        "📊 Progress: {}/{} files processed ({:3}% complete, {} remaining)",
                        progress, total_files, percentage, remaining
                    );
                }

                // Read file (lossy UTF-8)
                let content_bytes = match aiofs::read(&path).await {
                    Ok(b) => b,
                    Err(e) => {
                        warn!("Failed to read {}: {}", path.display(), e);
                        return;
                    }
                };
                let file_hash = blake3_hex(&content_bytes);
                let content = String::from_utf8_lossy(&content_bytes).to_string();

                scanned_files.fetch_add(1, Ordering::SeqCst);

                let path_hash = blake3_hex(relative_path_str.as_bytes())[..10].to_string();
                let file_stem = relative_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let issues = scan(&content);

                // Filter for Critical or High severity non-Panic (temporarily expanded for debugging)
                let mut candidates: Vec<_> = issues
                    .into_iter()
                    .filter(|issue| {
                        (issue.severity == "Critical" || issue.severity == "High") && issue.category != "Panic"
                    })
                    .collect();

                if candidates.is_empty() {
                    return;
                }

                let mut validated_non_fp = Vec::new();

                for issue in candidates.drain(..) {
                    // Stable issue key
                    let key = issue_fingerprint(
                        &issue.pattern_id,
                        &relative_path_str,
                        issue.line,
                        &issue.excerpt,
                    );

                    // Dedup within this run
                    {
                        let mut seen = seen_issue_keys.lock().await;
                        if !seen.insert(key.clone()) {
                            trace!("Dedup (same run): {}", key);
                            continue;
                        }
                    }

                    // Heuristics first (cheap)
                    let heuristics_fp = is_probably_comment(&issue.excerpt);

                    // Consult cache
                    let cached_decision = {
                        let cache = fp_cache.lock().await;
                        cache.get(&key).cloned()
                    };

                    if let Some(entry) = cached_decision {
                        if entry.file_hash == file_hash {
                            if entry.is_false_positive {
                                filtered_false_positives.fetch_add(1, Ordering::SeqCst);
                                info!(
                                    "Filtered FP from cache {} (by: {}, conf: {:.2})",
                                    issue.pattern_id, entry.decided_by, entry.confidence
                                );
                                // Write an informational FP artifact again (optional) – keep noise low: skip
                                continue;
                            } else {
                                // Previously validated as real issue
                                validated_non_fp.push((issue, None)); // None for no enhancement yet
                                continue;
                            }
                        }
                        // If file hash changed, fall through to re-eval
                    }

                    // Decide using heuristics + AI
                    let mut is_fp = heuristics_fp;
                    let mut decided_by = if heuristics_fp { "heuristics".to_string() } else { "unknown".to_string() };
                    let mut confidence = if heuristics_fp { 1.0 } else { 0.0 };
                    let mut reasoning: Option<String> = None;
                    let mut deepseek_enhancement: Option<String> = None;

                    // AI validation (Rig validator first, then enhanced DeepSeek analysis for all cases)
                    if !is_fp {
                        // Try Rig validator first (preferred method)
                        if let Some(rig_validator) = &rig_fp_validator {
                            let _permit = ds_semaphore.acquire().await.unwrap();
                            
                            // Create issue view for Rig validator
                            let issue_view = IssueView {
                                pattern_id: &issue.pattern_id,
                                name: &issue.name,
                                category: &issue.category,
                                severity: &issue.severity,
                                relative_path: &relative_path_str,
                                line: issue.line,
                                col: issue.col,
                                excerpt: &issue.excerpt,
                            };

                            // Optional: gather neighbors from vector store (placeholder for now)
                            let neighbors: Vec<Neighbor> = Vec::new(); // TODO: integrate with vector store if available

                            // Run Rig validation
                            let rig_result = rig_validator
                                .validate(&issue_view, &content, &repo_branch, &commit_hash, &neighbors)
                                .await;

                            is_fp = rig_result.is_false_positive;
                            decided_by = rig_result.decided_by.to_string();
                            confidence = rig_result.confidence;
                            reasoning = Some(rig_result.reasoning);
                        }
                        // Fallback to legacy DeepSeek validation if Rig validator unavailable
                        else if deepseek_client.is_some() {
                            // rate-limited
                            let _permit = ds_semaphore.acquire().await.unwrap();
                            match fp_filter.validate_issue(&issue, &content).await {
                                Ok(validation) => {
                                    if validation.is_false_positive
                                        && ai_says_false_positive(validation.confidence, fp_confidence_threshold)
                                    {
                                        is_fp = true;
                                        decided_by = "legacy_ai".to_string();
                                        confidence = validation.confidence;
                                        reasoning = Some(validation.reasoning.clone());
                                    } else {
                                        is_fp = false;
                                        decided_by = "legacy_ai".to_string();
                                        confidence = validation.confidence;
                                        reasoning = Some(validation.reasoning.clone());
                                    }
                                }
                                Err(e) => {
                                    warn!("Legacy validation failed for {}: {}", issue.pattern_id, e);
                                    // keep as non-FP (fail-open to avoid hiding criticals)
                                    is_fp = false;
                                    decided_by = "ai_error".to_string();
                                    confidence = 0.0;
                                }
                            }
                        } else {
                            trace!("No AI validation configured; skipping FP validation");
                        }
                    } else {
                        reasoning = Some("Snippet appears to be a comment or block comment".into());
                    }

                    // Enhanced DeepSeek Analysis for all validated issues (not just FP checks)
                    if !is_fp && deepseek_client.is_some() {
                        let ds_client = deepseek_client.as_ref().unwrap();
                        let _permit = ds_semaphore.acquire().await.unwrap();
                        
                        // Get comprehensive DeepSeek analysis for real issues
                        match enhance_issue_with_deepseek(ds_client, &issue, &content, &relative_path_str).await {
                            Ok(enhancement) => {
                                info!(
                                    "🤖 DeepSeek enhanced analysis for {} @ {}: {} chars",
                                    issue.pattern_id,
                                    &relative_path_str,
                                    enhancement.len()
                                );
                                deepseek_enhancement = Some(enhancement);
                            }
                            Err(e) => {
                                debug!("DeepSeek enhancement failed for {}: {}", issue.pattern_id, e);
                            }
                        }
                    }

                    // Update cache
                    {
                        let mut cache = fp_cache.lock().await;
                        cache.put(FpCacheEntry {
                            issue_key: key.clone(),
                            file_hash: file_hash.clone(),
                            is_false_positive: is_fp,
                            confidence,
                            decided_by: decided_by.clone(),
                            reasoning: reasoning.clone(),
                            last_seen_ts: chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string(),
                        });
                    }

                    if is_fp {
                        filtered_false_positives.fetch_add(1, Ordering::SeqCst);
                        info!(
                            "Filtered false positive {} [{}] (by: {}, conf: {:.2})",
                            issue.pattern_id, relative_path_str, decided_by, confidence
                        );

                        // Skip saving false positive reports - just log them
                        continue;
                    }

                    // Non-FP validated - store with enhancement
                    validated_non_fp.push((issue, deepseek_enhancement));
                }

                if !validated_non_fp.is_empty() {
                    info!(
                        "Found {} CRITICAL/HIGH non-panic issues in: {}",
                        validated_non_fp.len(),
                        &relative_path_str
                    );
                    total_issues.fetch_add(validated_non_fp.len(), Ordering::SeqCst);

                    // Store each validated issue as a bug report (enhanced with DeepSeek analysis)
                    for (issue, deepseek_enhancement) in validated_non_fp {
                        let bug_id = format!(
                            "SCAN_{}_{}_{}_L{}",
                            issue.pattern_id, file_stem, path_hash, issue.line
                        );
                        let bugs_dir = Config::manifest_dir().join(Config::BUGS_DIRECTORY);
                        if let Err(e) = aiofs::create_dir_all(&bugs_dir).await {
                            warn!("Failed to create bugs directory: {}", e);
                            continue;
                        }
                        let bug_file = bugs_dir.join(format!("{}.json", bug_id));
                        
                        // Enhanced bug data with DeepSeek analysis
                        let mut bug_data = serde_json::json!({
                            "bug_id": bug_id,
                            "file_name": file_stem,
                            "analysis_context": "Automated pattern detection with AI enhancement",
                            "description": format!("{} detected: {}", issue.category, issue.name),
                            "severity": issue.severity,
                            "file_location": {
                                "relative_path": &relative_path_str,
                                "line": issue.line,
                                "column": issue.col
                            },
                            "code_sample": issue.excerpt,
                            "pattern_id": issue.pattern_id,
                            "fix_suggestion": format!("Review {} pattern usage", issue.category.to_lowercase()),
                            "timestamp": chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string(),
                            "workspace_info": {
                                "repository": "repository",
                                "branch": &repo_branch,
                                "commit_hash": &commit_hash
                            },
                            "ai_powered": true,
                            "deepseek_enhancement_available": deepseek_enhancement.is_some()
                        });

                        // Add DeepSeek enhancement if available
                        if let Some(enhancement) = deepseek_enhancement {
                            bug_data["deepseek_analysis"] = serde_json::Value::String(enhancement);
                            bug_data["analysis_context"] = serde_json::Value::String("Automated pattern detection with comprehensive DeepSeek AI analysis".to_string());
                        }

                        if let Err(e) = aiofs::write(
                            &bug_file,
                            serde_json::to_string_pretty(&bug_data).unwrap_or_default(),
                        )
                        .await
                        {
                            warn!("Failed to write bug report: {}", e);
                        } else {
                            debug!("📝 Enhanced bug report saved: {}", bug_file.display());
                        }
                    }
                }
            }
        })
        .buffer_unordered(scan_concurrency)
        .collect::<Vec<()>>()
        .await;

    // Save FP cache
    if let Err(e) = fp_cache.lock().await.save(&cache_path).await {
        warn!("Failed to persist FP cache: {}", e);
    }

    info!(
        "✅ CRITICAL/HIGH (non-panic, non-test) pattern scanning complete. Scanned {} Rust files, found {} issues. Filtered false positives: {}",
        scanned_files.load(Ordering::SeqCst),
        total_issues.load(Ordering::SeqCst),
        filtered_false_positives.load(Ordering::SeqCst),
    );
    Ok(())
}

/// Re-run dual validation (Rig extractor + legacy/pattern) on existing bug JSONs and annotate with `double_test`.
async fn revalidate_existing_bugs() -> Result<()> {
    use std::fs;
    use std::path::PathBuf;

    info!("🔁 Starting double revalidation over existing bug artifacts...");

    let bugs_dir = Config::bugs_directory_path();
    if !bugs_dir.exists() {
        warn!("Bugs directory not found: {}", bugs_dir.display());
        return Ok(());
    }

    let mut entries: Vec<PathBuf> = fs::read_dir(&bugs_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    entries.sort();

    info!("Found {} bug JSONs to revalidate", entries.len());

    // Initialize AI helpers
    let deepseek_client = DeepSeekClient::from_env().ok();
    let mut legacy = FalsePositiveFilter::new(deepseek_client.clone());

    // Rig validator only if DEEPSEEK_API_KEY is set to avoid provider panic.
    let rig_validator = if std::env::var("DEEPSEEK_API_KEY").is_ok() {
        RigFpValidator::new(0.75, 0.60, 0.80, 24, 3).ok()
    } else {
        None
    };

    for path in entries {
        // Load JSON
        let raw = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to read {}: {}", path.display(), e);
                continue;
            }
        };
        let mut v: Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                warn!("Failed to parse JSON {}: {}", path.display(), e);
                continue;
            }
        };

        // Extract minimal fields
        let file_rel = v
            .pointer("/file_location/relative_path")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let line = v
            .pointer("/file_location/line")
            .and_then(|x| x.as_u64())
            .unwrap_or(1) as usize;
        let col = v
            .pointer("/file_location/column")
            .and_then(|x| x.as_u64())
            .unwrap_or(1) as usize;
        let severity = v
            .pointer("/severity")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let pattern_id = v
            .pointer("/pattern_id")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let code_sample = v
            .pointer("/code_sample")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let name = v
            .pointer("/description")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();

        if file_rel.is_empty() {
            warn!(
                "{} missing file_location.relative_path; skipping",
                path.display()
            );
            continue;
        }

        // Reconstruct absolute path from repo root
        let repo_root = Config::rig_repo_path();
        let abs = repo_root.join(&file_rel);
        let content = match fs::read_to_string(&abs) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to read source {}: {}", abs.display(), e);
                continue;
            }
        };

        // Prepare issue view
        let issue_view = IssueView {
            pattern_id: &pattern_id,
            name: &name,
            category: "Unknown",
            severity: &severity,
            relative_path: &file_rel,
            line,
            col,
            excerpt: &code_sample,
        };

        // Run Rig validator
        let (rig_ok, rig_obj) = if let Some(rig) = &rig_validator {
            let res = rig
                .validate(
                    &issue_view,
                    &content,
                    "revalidate",
                    "revalidate",
                    &Vec::<Neighbor>::new(),
                )
                .await;
            let o = json!({
                "is_false_positive": res.is_false_positive,
                "confidence": res.confidence,
                "reason": res.reasoning,
                "decided_by": res.decided_by,
            });
            (true, o)
        } else {
            (false, json!({"error": "rig_validator_unavailable"}))
        };

        // Run legacy validator
        let (legacy_ok, legacy_obj) = {
            // We need Issue struct minimally; map fields
            let issue = IssueShim {
                pattern_id: pattern_id.clone(),
                name: name.clone(),
                category: "Unknown".into(),
                severity: severity.clone(),
                line,
                col,
                excerpt: code_sample.clone(),
            };
            match FalsePositiveFilter::validate_issue(&mut legacy, &issue.into_issue(), &content).await {
                Ok(val) => {
                    let o = json!({
                        "is_false_positive": val.is_false_positive,
                        "confidence": val.confidence,
                        "reason": val.reasoning,
                        "decided_by": if deepseek_client.is_some() { "legacy_ai" } else { "pattern_based" },
                    });
                    (true, o)
                }
                Err(e) => (false, json!({"error": format!("legacy_error: {}", e)})),
            }
        };

        // Determine double confirmation: both available and both say not FP
        let mut double_confirmed = false;
        if rig_ok && legacy_ok {
            let a = rig_obj
                .get("is_false_positive")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);
            let b = legacy_obj
                .get("is_false_positive")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);
            double_confirmed = !a && !b;
        }

        // Attach results
        v["double_test"] = json!({
            "rig": rig_obj,
            "legacy": legacy_obj,
            "double_confirmed": double_confirmed,
        });

        // Persist
        if let Err(e) = fs::write(&path, serde_json::to_string_pretty(&v)? ) {
            warn!("Failed to write {}: {}", path.display(), e);
        } else {
            debug!("Updated {}", path.display());
        }
    }

    info!("✅ Double revalidation complete");
    Ok(())
}

// Minimal shim to construct a patterns::Issue for legacy validator
#[derive(Clone)]
struct IssueShim {
    pattern_id: String,
    name: String,
    category: String,
    severity: String,
    line: usize,
    col: usize,
    excerpt: String,
}

impl IssueShim {
    fn into_issue(self) -> patterns::Issue {
        patterns::Issue {
            pattern_id: Box::leak(self.pattern_id.into_boxed_str()),
            name: Box::leak(self.name.into_boxed_str()),
            severity: Box::leak(self.severity.into_boxed_str()),
            category: Box::leak(self.category.into_boxed_str()),
            line: self.line,
            col: self.col,
            excerpt: self.excerpt,
            severity_confirmed: true,
            confidence_score: 0.9,
        }
    }
}

// Helper function to get current git branch
async fn get_git_branch() -> Option<String> {
    let output = tokio::process::Command::new("git")
        .args(["branch", "--show-current"])
        .output()
        .await
        .ok()?;

    if output.status.success() {
        String::from_utf8(output.stdout)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    }
}

// Helper function to get current git commit hash
async fn get_git_commit_hash() -> Option<String> {
    let output = tokio::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .await
        .ok()?;

    if output.status.success() {
        String::from_utf8(output.stdout)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    }
}

pub struct UnifiedServerState {
    pub vector_store: Option<VectorStoreManager>,
    pub deepseek_client: Option<DeepSeekClient>,
}

impl UnifiedServerState {
    pub async fn new() -> Result<Self> {
        info!("🚀 Initializing Unified Server State with FastEmbed and DeepSeek...");
        debug!("Starting initialization of vector store and AI client components");

        // Initialize vector store with FastEmbed (no API key required)
        trace!("Attempting to initialize vector store with FastEmbed local embeddings");
        let vector_store = match VectorStoreManager::new().await {
            Ok(store) => {
                info!("✅ Vector store initialized with FastEmbed local embeddings");
                debug!("Vector store ready for similarity search operations");
                Some(store)
            }
            Err(e) => {
                error!("❌ Failed to initialize vector store: {}", e);
                warn!("⚠️ Vector similarity search will be unavailable");
                None
            }
        };

        // Initialize DeepSeek client
        trace!("Attempting to initialize DeepSeek AI client from environment variables");
        let deepseek_client = match DeepSeekClient::from_env() {
            Ok(client) => {
                info!("✅ DeepSeek client initialized from environment");
                debug!("Testing DeepSeek API connection...");
                // Test the connection
                match client.validate_connection().await {
                    Ok(_) => {
                        info!("✅ DeepSeek API connection validated successfully");
                        debug!("DeepSeek client ready for code analysis operations");
                        Some(client)
                    }
                    Err(e) => {
                        error!("❌ DeepSeek API connection failed: {}", e);
                        warn!("⚠️ DeepSeek client available but connection unreliable");
                        Some(client) // Keep client even if validation fails
                    }
                }
            }
            Err(e) => {
                error!("❌ Failed to initialize DeepSeek client: {}", e);
                warn!("⚠️ AI code analysis will be unavailable");
                None
            }
        };

        debug!("Unified server state initialization complete");
        Ok(Self {
            vector_store,
            deepseek_client,
        })
    }
}



// Test functions removed - use scan_repository for real file analysis

// Main function to run the rig application with FastEmbed and DeepSeek
async fn run_rig_sqlite_application() -> Result<()> {
    info!("🎯 Starting Rig Analyzer application with AI integration");
    println!("🎯 Starting Rig Analyzer with FastEmbed and DeepSeek integration");
    println!("=======================================================================");
    debug!("Application startup initiated with comprehensive AI toolchain");

    println!("\n✅ FastEmbed + DeepSeek integration initialized successfully!");
    info!("🎉 Integration initialization complete");

    // Run automated bug analysis using repository scanning on real files
    debug!("Starting automated bug analysis on repository files");
    scan_repository().await?;

    // After scanning, re-run double validation to annotate bug JSONs
    debug!("Starting double revalidation of bug artifacts");
    if let Err(e) = revalidate_existing_bugs().await {
        warn!("Double revalidation failed: {}", e);
    }

    info!("✅ Application operations complete");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env file FIRST (before any client initialization)
    // Try absolute path first, then relative paths
    let env_paths = [
        Config::manifest_dir().join(".env"),
        std::path::Path::new(".env").to_path_buf(),
        std::path::Path::new("nautilus-trader-rig/.env").to_path_buf(),
    ];

    for env_path in &env_paths {
        if env_path.exists() {
            dotenvy::from_path(env_path).ok();
            break;
        }
    }

    // Initialize centralized logging system
    init_dev_logging()?;

    // Log environment file status
    debug!("Environment file path: {}", Config::ENV_FILE_PATH);
    debug!("Environment file exists: {}", Config::env_file_exists());
    if let Ok(api_key) = std::env::var("DEEPSEEK_API_KEY") {
        debug!("DEEPSEEK_API_KEY loaded (length: {} chars)", api_key.len());
    } else {
        warn!("DEEPSEEK_API_KEY not found in environment");
    }

    // Log rig repository configuration
    if let Some(rig_path) = Config::rig_repo_path_env_value() {
        info!("REPO_PATH set to: {}", rig_path);
        debug!("Using rig repository from environment variable");
    } else {
        let default_path = Config::rig_repo_path();
        info!(
            "REPO_PATH not set, using default: {}",
            default_path.display()
        );
        debug!("Using default rig repository path");
    }

    info!("🚀 Starting Rig Analyzer with MCP server...");
    debug!("Application entry point - initializing concurrent services");

    // Check configuration paths at startup
    info!("ℹ️ Validating repository configuration");
    let repo_path = Config::rig_repo_path();
    let exists = repo_path.exists();
    let status = if exists { "Found" } else { "Missing" };
    tracing::info!("📁 Repository path check: {:?} ({})", repo_path, status);

    if exists {
        info!("✅ Repository path is accessible for scanning");
    } else {
        warn!("⚠️ Repository path does not exist: {}", repo_path.display());
    }

    // Start MCP server on a separate thread
    debug!("Spawning MCP server on background thread");
    let _mcp_handle = tokio::spawn(async {
        info!("🌐 Starting MCP server thread");
        if let Err(e) = run_mcp_server().await {
            error!("❌ MCP server failed: {}", e);
        } else {
            info!("✅ MCP server completed successfully");
        }
    });

    // Run the main application
    debug!("Starting main application thread");
    if let Err(e) = run_rig_sqlite_application().await {
        error!("❌ Main application failed: {}", e);
        return Err(e);
    }

    info!("✅ Application shutdown complete");
    Ok(())
}
