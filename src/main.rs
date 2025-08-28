//! AI Architecture Improver - Automated Tiny Improvements
//!
//! This tool automatically finds, audits, and commits tiny architecture improvements
//! using AI-powered analysis with ranking and memory systems.

mod config;
mod deepseek;

use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
};

use anyhow::{anyhow, Context, Result};
use apalis::layers::retry::RetryPolicy;
use apalis::prelude::*;
use apalis_cron::CronStream;
use apalis_cron::Schedule;
use chrono::{DateTime, Utc};
use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use walkdir::WalkDir;

use config::Config;

// ============================================================================
// AI Tools for Code Analysis (for future enhancement)
// ============================================================================

/// Security-focused file reader with extension whitelisting
///
/// Prevents reading of unauthorized file types by checking extensions
/// against a predefined whitelist of source and config files
#[allow(dead_code)]
struct ReadFileContentTool;

#[allow(dead_code)]
impl ReadFileContentTool {
    async fn read_file(&self, file_path: String) -> Result<String, String> {
        // Security check - only allow reading source files
        if !file_path.ends_with(".rs")
            && !file_path.ends_with(".toml")
            && !file_path.ends_with(".md")
        {
            return Err("Only .rs, .toml, and .md files are allowed".to_string());
        }

        if file_path.contains("..") || file_path.starts_with('/') {
            return Err("Path traversal not allowed".to_string());
        }

        match std::fs::read_to_string(&file_path) {
            Ok(content) => Ok(content),
            Err(e) => Err(format!("Failed to read file {}: {}", file_path, e)),
        }
    }
}

#[allow(dead_code)]
struct SearchCodeTool;

#[allow(dead_code)]
impl SearchCodeTool {
    async fn search_pattern(
        &self,
        pattern: String,
        file_extensions: Option<String>,
    ) -> Result<String, String> {
        let extensions = file_extensions.unwrap_or_else(|| "rs,toml".to_string());
        let ext_list: Vec<&str> = extensions.split(',').collect();

        let mut results = Vec::new();
        let regex = match Regex::new(&pattern) {
            Ok(r) => r,
            Err(e) => return Err(format!("Invalid regex pattern: {}", e)),
        };

        for entry in WalkDir::new(".") {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            if let Some(ext) = path.extension() {
                if ext_list.iter().any(|&e| e == ext.to_string_lossy()) {
                    if let Ok(content) = std::fs::read_to_string(path) {
                        for (line_num, line) in content.lines().enumerate() {
                            if regex.is_match(line) {
                                results.push(format!(
                                    "{}:{}: {}",
                                    path.display(),
                                    line_num + 1,
                                    line.trim()
                                ));
                            }
                        }
                    }
                }
            }
        }

        if results.is_empty() {
            Ok("No matches found".to_string())
        } else {
            Ok(results.join("\n"))
        }
    }
}

#[allow(dead_code)]
struct CodeValidationTool;

#[allow(dead_code)]
impl CodeValidationTool {
    async fn check_code(&self, check_type: String) -> Result<String, String> {
        let output = match check_type.as_str() {
            "check" => std::process::Command::new("cargo")
                .arg("check")
                .output()
                .map_err(|e| e.to_string())?,
            "clippy" => std::process::Command::new("cargo")
                .arg("clippy")
                .arg("--")
                .arg("-W")
                .arg("clippy::all")
                .output()
                .map_err(|e| e.to_string())?,
            "test" => std::process::Command::new("cargo")
                .arg("test")
                .arg("--no-run")
                .output()
                .map_err(|e| e.to_string())?,
            _ => return Err("Invalid check type. Use 'check', 'clippy', or 'test'".to_string()),
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        Ok(format!(
            "Exit code: {}\nStdout:\n{}\nStderr:\n{}",
            output.status.code().unwrap_or(-1),
            stdout,
            stderr
        ))
    }
}

#[allow(dead_code)]
struct CodeAnalysisTool;

#[allow(dead_code)]
impl CodeAnalysisTool {
    async fn analyze_function(
        &self,
        file_path: String,
        function_name: String,
    ) -> Result<String, String> {
        let content = std::fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        // Simple function analysis
        let lines: Vec<&str> = content.lines().collect();
        let mut in_function = false;
        let mut function_lines = Vec::new();
        let mut brace_count = 0;

        for (i, line) in lines.iter().enumerate() {
            if line.contains(&format!("fn {}", function_name)) {
                in_function = true;
                function_lines.push((i + 1, line.trim()));
            } else if in_function {
                function_lines.push((i + 1, line.trim()));
                brace_count += line.chars().filter(|&c| c == '{').count() as i32;
                brace_count -= line.chars().filter(|&c| c == '}').count() as i32;

                if brace_count <= 0 && line.contains('}') {
                    break;
                }
            }
        }

        if function_lines.is_empty() {
            return Err(format!(
                "Function '{}' not found in {}",
                function_name, file_path
            ));
        }

        let complexity = function_lines.len();
        let analysis =
            format!(
            "Function '{}' analysis:\n- Lines: {}\n- Complexity: {}\n- Suggestion: {}\n\nCode:\n{}",
            function_name,
            complexity,
            if complexity > 50 { "High" } else if complexity > 20 { "Medium" } else { "Low" },
            if complexity > 50 { "Consider breaking into smaller functions" } 
            else if complexity > 20 { "Could benefit from some refactoring" }
            else { "Complexity is reasonable" },
            function_lines.iter()
                .map(|(num, line)| format!("{:3}: {}", num, line))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(analysis)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn parse_ai_response_json<T: for<'a> Deserialize<'a>>(response: &str) -> Result<T, anyhow::Error> {
    let cleaned = response
        .trim()
        .trim_start_matches("```json")
        .trim_end_matches("```")
        .trim();
    serde_json::from_str(cleaned)
        .or_else(|_| {
            // Try to extract JSON from response if it's embedded in text
            let json_start = cleaned.find('{').unwrap_or(0);
            let json_end = cleaned.rfind('}').map(|i| i + 1).unwrap_or(cleaned.len());
            serde_json::from_str(&cleaned[json_start..json_end])
        })
        .context("Could not parse JSON from AI response")
}

// ============================================================================
// Cron Job Data Structure
// ============================================================================

#[derive(Clone)]
struct ArchitectService {
    message: String,
}

impl ArchitectService {
    fn new() -> Self {
        Self {
            message: "AI Architecture Improver".to_string(),
        }
    }

    async fn execute(&self, _item: ArchitectReminder) -> Result<()> {
        info!("{} - Starting architecture analysis", &self.message);
        run_architecture_analysis().await
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
struct ArchitectReminder(DateTime<Utc>);

impl From<DateTime<Utc>> for ArchitectReminder {
    fn from(t: DateTime<Utc>) -> Self {
        ArchitectReminder(t)
    }
}

// ============================================================================
// Configuration (No CLI - Environment Variables)
// ============================================================================
// Configuration (Now imported from config.rs)
// ============================================================================

// ============================================================================
// Rank / Profile System
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Profile {
    rank: Rank,
    success_streak: u32,
    total_success: u32,
    total_runs: u32,
    rolling_avg: f32,
    last_promo_ts: String,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            rank: Rank::Junior,
            success_streak: 0,
            total_success: 0,
            total_runs: 0,
            rolling_avg: 0.0,
            last_promo_ts: Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Rank {
    Junior,
    Mid,
    Senior,
}

// ============================================================================
// Structured Outputs
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProposedChange {
    title: String,
    rationale: String,
    patch: String,
    estimated_changed_lines: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditVerdict {
    ok: bool,
    best_practice_score: f32,
    security_ok: bool,
    secret_leak_risk: bool,
    vuln_risk: bool,
    comments: Vec<String>,
}

// ============================================================================
// Memory System
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryItem {
    ts: String,
    title: String,
    summary: String,
    rank: Rank,
    score: f32,
}

fn memory_dir() -> &'static str {
    "./_architect_ai"
}
/// Validates and cleans a patch to ensure it can be applied successfully
fn validate_and_clean_patch(patch: &str, target_file: &str) -> Result<String, String> {
    let lines: Vec<&str> = patch.lines().collect();
    
    // Basic structural validation
    if lines.is_empty() {
        return Err("Patch is empty".to_string());
    }
    
    // Check for required diff headers
    let has_minus_header = lines.iter().any(|line| line.starts_with("--- a/"));
    let has_plus_header = lines.iter().any(|line| line.starts_with("+++ b/"));
    
    if !has_minus_header || !has_plus_header {
        return Err("Patch missing required diff headers (--- a/ and +++ b/)".to_string());
    }
    
    // Check for hunk headers
    let has_hunk = lines.iter().any(|line| line.starts_with("@@"));
    if !has_hunk {
        return Err("Patch missing hunk headers (@@)".to_string());
    }
    
    // Validate file context by checking if target file exists
    if let Ok(file_content) = fs::read_to_string(target_file) {
        let file_lines: Vec<&str> = file_content.lines().collect();
        
        // Extract context lines (lines that start with space) from patch
        let context_lines: Vec<&str> = lines.iter()
            .filter(|line| line.starts_with(' '))
            .map(|line| &line[1..]) // Remove the leading space
            .collect();
            
        // If we have context lines, verify at least some exist in the target file
        if !context_lines.is_empty() {
            let context_found = context_lines.iter().any(|context| {
                file_lines.iter().any(|file_line| file_line.contains(context))
            });
            
            if !context_found {
                return Err(format!("Patch context does not match target file: {}", target_file));
            }
        }
    }
    
    // Clean the patch by ensuring proper line endings
    let mut cleaned_patch = patch.to_string();
    if !cleaned_patch.ends_with('\n') {
        cleaned_patch.push('\n');
    }
    
    Ok(cleaned_patch)
}

fn patch_path() -> &'static str {
    "./_architect_ai/patch.diff"
}
fn memory_file() -> &'static str {
    "./_architect_ai/memory.jsonl"
}
fn profile_file() -> &'static str {
    "./_architect_ai/profile.json"
}

fn load_profile() -> Profile {
    fs::read_to_string(profile_file())
        .ok()
        .and_then(|s| serde_json::from_str::<Profile>(&s).ok())
        .unwrap_or_default()
}

fn save_profile(p: &Profile) -> Result<()> {
    fs::create_dir_all(memory_dir()).ok();
    fs::write(profile_file(), serde_json::to_vec_pretty(p)?)?;
    Ok(())
}

fn load_memory() -> Vec<MemoryItem> {
    fs::read_to_string(memory_file())
        .map(|s| {
            s.lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect()
        })
        .unwrap_or_else(|_| vec![])
}

fn append_memory(item: &MemoryItem) -> Result<()> {
    fs::create_dir_all(memory_dir()).ok();
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(memory_file())?;
    use std::io::Write;
    writeln!(f, "{}", serde_json::to_string(item)?)?;
    Ok(())
}

async fn build_memory_index(
    _ds: &deepseek::Client,
    _items: &[MemoryItem],
) -> Result<()> {
    // For now, just log that memory index is initialized
    // In the future, we could implement a simple in-memory search
    info!("Memory index initialized (basic mode)");
    Ok(())
}

// ============================================================================
// Execution Logging System
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExecutionTrace {
    timestamp: String,
    rank: Rank,
    phase: String,
    status: String,
    details: String,
    proposal_title: Option<String>,
    patch_content: Option<String>,
    error_details: Option<String>,
    git_status: Option<String>,
    audit_score: Option<f32>,
}

fn execution_log_file() -> &'static str {
    "./_architect_ai/execution.log"
}

fn log_execution_trace(trace: &ExecutionTrace) -> Result<()> {
    fs::create_dir_all(memory_dir()).ok();

    // Format log entry with timestamp and details
    let log_entry = format!(
        "{} [{}] {} - {} - {}\n",
        trace.timestamp,
        rank_slug(&trace.rank).to_uppercase(),
        trace.phase,
        trace.status,
        trace.details
    );

    // Add proposal details if available
    let detailed_entry = if let Some(title) = &trace.proposal_title {
        format!("{}  Proposal: {}\n", log_entry, title)
    } else {
        log_entry
    };

    // Add patch content if available
    let full_entry = if let Some(patch) = &trace.patch_content {
        format!("{}  Patch:\n{}\n", detailed_entry, patch)
    } else {
        detailed_entry
    };

    // Add error details if available
    let final_entry = if let Some(error) = &trace.error_details {
        format!("{}  Error: {}\n", full_entry, error)
    } else {
        full_entry
    };

    // Add git status if available
    let complete_entry = if let Some(git_status) = &trace.git_status {
        format!("{}  Git Status:\n{}\n", final_entry, git_status)
    } else {
        final_entry
    };

    // Write to file
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(execution_log_file())?;

    file.write_all(complete_entry.as_bytes())?;
    file.flush()?;

    Ok(())
}

// ============================================================================
// Security & Git Helpers
// ============================================================================

fn likely_secret_present(s: &str) -> bool {
    let aws = Regex::new(r"AKIA[0-9A-Z]{16}").unwrap();
    let gh = Regex::new(r"gh[pousr]_[A-Za-z0-9]{24,}").unwrap();
    let jwt =
        Regex::new(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9._-]{10,}\.[A-Za-z0-9._-]{10,}").unwrap();
    let generic =
        Regex::new(r#"(?i)(secret|api[_-]?key|token|password)\s*[:=]\s*['"][^'"\n]{12,}['"]"#)
            .unwrap();
    aws.is_match(s) || gh.is_match(s) || jwt.is_match(s) || generic.is_match(s)
}

fn run(cwd: &Path, prog: &str, args: &[&str]) -> Result<String> {
    let out = Command::new(prog).args(args).current_dir(cwd).output()?;
    if !out.status.success() {
        return Err(anyhow!(
            "{prog} {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn repo_root() -> Result<PathBuf> {
    let out = run(Path::new("."), "git", &["rev-parse", "--show-toplevel"])?;
    Ok(PathBuf::from(out.trim()))
}

fn origin_url(root: &Path) -> Result<String> {
    Ok(run(root, "git", &["remote", "get-url", "origin"])?
        .trim()
        .into())
}

fn https_url_from_remote(remote: &str) -> Option<String> {
    if remote.starts_with("https://") {
        Some(remote.to_string())
    } else {
        remote
            .strip_prefix("git@github.com:")
            .map(|tail| format!("https://github.com/{tail}"))
    }
}

const USERINFO: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'@')
    .add(b'\\')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'|')
    .add(b'}');

fn enc(s: &str) -> String {
    utf8_percent_encode(s, USERINFO).to_string()
}

fn changed_lines_in_patch(patch: &str) -> usize {
    patch
        .lines()
        .filter(|l| {
            (l.starts_with('+') || l.starts_with('-'))
                && !l.starts_with("+++")
                && !l.starts_with("---")
        })
        .count()
}

fn sanitize_for_log(s: &str) -> String {
    Regex::new(r"[A-Za-z0-9_\-]{20,}\@[A-Za-z0-9\./:-]+")
        .unwrap()
        .replace_all(s, "***REDACTED***")
        .into_owned()
}

// ============================================================================
// AI Tools (Simple text-based helpers)
// ============================================================================

fn list_arch_files() -> Result<String> {
    let root = env::current_dir()?;
    let mut out = Vec::new();
    for dent in WalkDir::new(&root) {
        let e = dent?;
        if !e.file_type().is_file() {
            continue;
        }
        let p = e.path();
        let rel = p
            .strip_prefix(&root)
            .unwrap_or(p)
            .to_string_lossy()
            .to_string();
        // Focus on code and tests, ignore documentation
        let is_code = rel.ends_with(".rs")
            || rel == "Cargo.toml"
            || rel.starts_with("src/")
            || rel.starts_with("tests/")
            || rel.ends_with(".toml");
        let is_documentation = rel.ends_with(".md")
            || rel.starts_with("docs/")
            || rel.starts_with("doc/")
            || rel.contains("/README")
            || rel.contains("/CHANGELOG")
            || rel.contains("/LICENSE");
        let ignore = rel.contains("/.git/")
            || rel.contains("/target/")
            || rel.starts_with("target/")
            || rel.contains("/node_modules/")
            || rel.contains("/.cargo/")
            || is_documentation;
        if is_code && !ignore {
            out.push(rel);
        }
    }
    Ok(serde_json::to_string_pretty(&out).unwrap_or("[]".into()))
}

/// Collect recent patch failure patterns from stdout/logs to detect self-improvement needs
fn collect_recent_patch_failures() -> Option<String> {
    // Read recent failures from execution log
    if let Ok(log_content) = fs::read_to_string("_architect_ai/execution.log") {
        let recent_lines: Vec<&str> = log_content
            .lines()
            .rev()
            .take(200) // Last 200 lines
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        
        let recent_log = recent_lines.join("\n");
        
        // Look for failure patterns
        let has_failures = recent_log.contains("FAILED")
            || recent_log.contains("corrupt patch")
            || recent_log.contains("could not compile")
            || recent_log.contains("error[E0");
            
        if has_failures {
            return Some(recent_log);
        }
    }
    None
}

// ============================================================================
// AI Prompts
// ============================================================================

fn improver_preamble(
    rank: &Rank,
    max_lines: usize,
    include: &str,
    exclude: &str,
    files: &str,
    log_summary: Option<&str>,
    target_file_contents: Option<&str>,
) -> String {
    let scope = match rank {
        Rank::Junior => "Aim for small functional improvements: fix bugs, add error handling, optimize performance, improve logic.",
        Rank::Mid => "You may refactor functions, improve error messages, add helper functions, reorganize imports.",
        Rank::Senior => "You may create new modules, add advanced features, improve architecture within the line budget.",
    };

    let log_section = if let Some(logs) = log_summary {
        if !logs.trim().is_empty() {
            format!("\nRecent runtime log analysis:\n{}\n", logs)
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    // Check if logs indicate patch corruption issues - trigger self-improvement mode
    let self_improvement_needed = log_summary
        .map(|logs| {
            let has_corruption = logs.contains("corrupt patch")
                || logs.contains("PATCH_CHECK") && logs.contains("FAILED");
            let has_compilation_errors = logs.contains("could not compile")
                || logs.contains("error[E0")
                || logs.contains("❌ Code does not compile");
            let has_failures = has_corruption || has_compilation_errors;
            
            if has_failures {
                if has_compilation_errors {
                    info!("🔧 SELF-IMPROVEMENT TRIGGERED: Detected compilation failures from AI changes");
                } else {
                    info!("🔧 SELF-IMPROVEMENT TRIGGERED: Detected patch corruption patterns");
                }
            }
            has_failures
        })
        .unwrap_or(false);

    let priority_guidance = if self_improvement_needed {
        info!("🎯 Applying priority guidance for system fixes");
        "\n🔧 PRIORITY: System failures detected in logs. Focus on fixing core patch generation issues:\n\
         \n\
         🚨 IMMEDIATE FIXES NEEDED:\n\
         - Fix patch format corruption (corrupt patch at line X errors)\n\
         - Stop using placeholder text like 'XXX,XX' or 'someindex' in patches\n\
         - Ensure exact line number matching with current file content\n\
         - Validate import statements before changing them\n\
         - Add proper context lines (3-5 before/after changes)\n\
         \n\
         🎯 DEEPSEEK ANALYSIS INTEGRATION:\n\
         - Use the provided file context to generate accurate patches\n\
         - Match exact line numbers and content from the current file state\n\
         - Verify all imports and dependencies exist before modifying them\n\
         - Test logical changes against current codebase structure\n\
         \n\
         CRITICAL: Generate patches that apply cleanly on first try!\n\
         CRITICAL: Use real line numbers, not placeholders!\n\
         CRITICAL: Include sufficient context for git apply to work!\n"
    } else {
        ""
    };

    let file_context_section = if let Some(contents) = target_file_contents {
        format!("\nCRITICAL - Current file contents (use this to ensure patch accuracy):\n```\n{}\n```\n", contents)
    } else {
        String::new()
    };
    format!(
        r#"
You are an AI Architect at **{rank}** level. Find EXACTLY ONE tiny improvement to this Rust codebase
with the smallest blast radius. Keep changes within **{max_lines} lines** total.

Available files:
{files}
{log_section}{priority_guidance}
{file_context_section}

Scope guidance: {scope}

Examples of good improvements:
- Fix bugs and edge cases
- Improve error handling and recovery
- Add helper functions to reduce duplication
- Create new utility modules (src/utils.rs, src/git.rs, etc.)
- Add better logging or debug output
- Improve function signatures or return types
- Add configuration options
- Small performance optimizations
- Enhance safety and reliability

AVOID: Documentation comments, code formatting, non-functional changes

Guardrails:
- No breaking changes to public APIs
- No secrets, credentials, or tokens in code
- Ensure code compiles and tests pass
- Focus on maintainability and readability
- Focus includes: {include}
- Avoid paths containing: {exclude}

Output JSON format (respond with just the JSON, no fences):
{{
  "title": "...",
  "rationale": "...", 
  "patch": "<unified diff repo-relative>",
  "estimated_changed_lines": <int|null>
}}

CRITICAL PATCH FORMAT REQUIREMENTS:
1. The unified diff MUST apply cleanly with 'git apply --check'
2. Use EXACT file paths relative to repo root: "--- a/src/main.rs" and "+++ b/src/main.rs"
3. Include sufficient context lines (3-5 lines before and after changes)
4. Use proper hunk headers: @@ -67,7 +67,6 @@ struct ReadFileContentTool;
5. Context lines start with space, deletions with -, additions with +
6. Line numbers in hunk headers must match the actual file content
7. NEVER use placeholder text like "XXX,XX" or "someindex" - use real line numbers
8. Verify the context matches the current file state exactly

EXAMPLE CORRECT PATCH FORMAT:
--- a/src/main.rs
+++ b/src/main.rs
@@ -45,7 +45,7 @@ impl ReadFileContentTool {{
 
     async fn read_file(&self, file_path: String) -> Result<String, String> {{
         // Security check - only allow reading source files
-        if !file_path.ends_with(".rs") {{
+        if !file_path.ends_with(".rs") && !file_path.ends_with(".toml") {{
             return Err("Only .rs files are allowed".to_string());
         }}
"#,
        rank = match rank {
            Rank::Junior => "junior",
            Rank::Mid => "mid",
            Rank::Senior => "senior",
        },
        max_lines = max_lines,
        include = if include.is_empty() {
            "(none)"
        } else {
            include
        },
        exclude = if exclude.is_empty() {
            "(none)"
        } else {
            exclude
        },
        files = files,
        log_section = log_section,
        priority_guidance = priority_guidance,
        file_context_section = file_context_section,
        scope = scope
    )
}

fn auditor_preamble() -> String {
    r#"
You are a senior Rust code reviewer. Given a unified diff and post-patch files, return ONLY JSON:

{
  "ok": true|false,
  "best_practice_score": 0..1,
  "security_ok": true|false,
  "secret_leak_risk": true|false,
  "vuln_risk": true|false,
  "comments": ["..."]
}

Heuristics:
- Prefer proper error handling, good naming conventions, and robust code.
- Code must compile and follow Rust best practices.
- No unsafe code without justification.
- Absolutely no secrets/keys/tokens/passwords in code or comments.
- Be conservative: if unsure, set ok=false and explain in comments.
"#
    .to_string()
}

fn mentor_preamble() -> String {
    r#"
You are a strict Rust mentor. Produce a single short line of feedback teaching the junior/mid developer
what to improve next time (max 180 chars, no newlines, no JSON, no fences). Be concrete and kind.
Focus on Rust best practices, code quality, and maintainability.
"#
    .to_string()
}

// ============================================================================
// Leveling System
// ============================================================================

fn maybe_promote(p: &Profile) -> (bool, Rank) {
    match p.rank {
        Rank::Junior => {
            if p.success_streak >= 3 && p.rolling_avg >= 0.90 {
                return (true, Rank::Mid);
            }
        }
        Rank::Mid => {
            if p.success_streak >= 5 && p.rolling_avg >= 0.93 {
                return (true, Rank::Senior);
            }
        }
        Rank::Senior => {}
    }
    (false, p.rank.clone())
}

fn rank_slug(r: &Rank) -> &'static str {
    match r {
        Rank::Junior => "jr",
        Rank::Mid => "mid",
        Rank::Senior => "sr",
    }
}

// ============================================================================
// Cron Job Handler
// ============================================================================

async fn architect_cron_job(job: ArchitectReminder, svc: Data<ArchitectService>) {
    info!("🕒 Cron job triggered at {:?}", job.0);
    if let Err(e) = svc.execute(job).await {
        warn!("❌ Architecture analysis failed in cron job: {}", e);

        // Log cron failure
        log_execution_trace(&ExecutionTrace {
            timestamp: Utc::now().to_rfc3339(),
            rank: Rank::Junior, // Default rank for cron failures
            phase: "CRON_FAILURE".to_string(),
            status: "FAILED".to_string(),
            details: format!("Architecture analysis failed in cron job: {}", e),
            proposal_title: None,
            patch_content: None,
            error_details: Some(format!("Error: {}", e)),
            git_status: None,
            audit_score: None,
        })
        .ok();
    }
}

// ============================================================================
// Shuttle Service Setup
// ============================================================================

#[shuttle_runtime::main]
async fn main() -> Result<MyService, shuttle_runtime::Error> {
    dotenvy::dotenv().ok();

    info!("🚀 Initializing AI Architecture Improver Cron Service");

    Ok(MyService {})
}

// Customize this struct with things from `shuttle_main` needed in `bind`,
// such as secrets or database connections
struct MyService {}

#[shuttle_runtime::async_trait]
impl shuttle_runtime::Service for MyService {
    async fn bind(self, _addr: std::net::SocketAddr) -> Result<(), shuttle_runtime::Error> {
        info!("🔧 Setting up cron job service");

        // Load configuration
        let config = Config::from_env();

        // Validate the cron schedule
        config
            .validate_cron_schedule()
            .map_err(|e| shuttle_runtime::Error::Custom(anyhow::Error::msg(e)))?;

        info!("📅 Using cron schedule: {}", config.cron_schedule);
        let schedule = Schedule::from_str(&config.cron_schedule)
            .map_err(|e| shuttle_runtime::Error::Custom(e.into()))?;

        let architect_service = ArchitectService::new();

        // Create in-memory cron stream (no database needed)
        let cron_stream = CronStream::new(schedule);

        // Create a worker that uses the architect service
        let worker = WorkerBuilder::new("architect-improver")
            .data(architect_service)
            .retry(RetryPolicy::retries(3))
            .backend(cron_stream)
            .build_fn(architect_cron_job);

        info!(
            "🕒 Starting cron worker with schedule: {}",
            config.cron_schedule
        );

        // Start the worker
        worker.run().await;

        Ok(())
    }
}

// ============================================================================
// Core Analysis Logic (moved from main)
// ============================================================================

async fn run_architecture_analysis() -> Result<()> {
    dotenvy::dotenv().ok();

    let config = Config::from_env();

    info!("🚀 Starting AI Architecture Improver");
    info!(
        "Config: max_lines={}, min_score={:.2}",
        config.max_changed_lines, config.min_audit_avg
    );

    // Provider
    let api_key =
        env::var("DEEPSEEK_API_KEY").context("DEEPSEEK_API_KEY environment variable not set")?;
    let ds = deepseek::Client::new(&api_key);
    let chat_model = deepseek::models::DEEPSEEK_REASONER;

    // Repo setup
    let root = repo_root()?;
    env::set_current_dir(&root)?;
    fs::create_dir_all(memory_dir()).ok();

    // Profile + memory + index
    let mut profile = load_profile();
    let memory = load_memory();
    build_memory_index(&ds, &memory).await?;

    // Log analysis start
    log_execution_trace(&ExecutionTrace {
        timestamp: Utc::now().to_rfc3339(),
        rank: profile.rank.clone(),
        phase: "START".to_string(),
        status: "INFO".to_string(),
        details: format!(
            "Starting analysis - rank={:?}, streak={}, avg={:.2}",
            profile.rank, profile.success_streak, profile.rolling_avg
        ),
        proposal_title: None,
        patch_content: None,
        error_details: None,
        git_status: None,
        audit_score: None,
    })
    .ok();

    info!(
        "👤 Current profile: rank={:?}, streak={}, avg={:.2}",
        profile.rank, profile.success_streak, profile.rolling_avg
    );

    // Build Improver agent (simple approach)
    let include = config.include.clone().unwrap_or_default();
    let exclude = config.exclude.clone().unwrap_or_default();

    // Get available files
    let files = list_arch_files().unwrap_or_else(|_| "[]".to_string());

    // Collect recent log analysis to detect self-improvement needs
    let log_summary = collect_recent_patch_failures();

    // Get current file contents for the main target file with better error handling
    let target_file_contents = match fs::read_to_string("src/main.rs") {
        Ok(content) => {
            let first_lines = content.lines().take(50).collect::<Vec<_>>().join("\n");
            info!(
                "✅ File context loaded: {} total lines, first 50 lines = {} chars",
                content.lines().count(),
                first_lines.len()
            );
            info!(
                "📄 File preview: {}",
                &first_lines[..std::cmp::min(200, first_lines.len())]
            );
            Some(first_lines)
        }
        Err(e) => {
            warn!("❌ Failed to read src/main.rs: {}", e);
            // Try from repository root
            match fs::read_to_string("./src/main.rs") {
                Ok(content) => {
                    let first_lines = content.lines().take(50).collect::<Vec<_>>().join("\n");
                    info!(
                        "✅ File context loaded via relative path: {} lines",
                        content.lines().count()
                    );
                    info!(
                        "📄 File preview: {}",
                        &first_lines[..std::cmp::min(200, first_lines.len())]
                    );
                    Some(first_lines)
                }
                Err(e2) => {
                    warn!("❌ Failed to read via relative path: {}", e2);
                    None
                }
            }
        }
    };

    let mut improver = ds
        .agent(chat_model)
        .preamble(&improver_preamble(
            &profile.rank,
            config.max_changed_lines,
            &include,
            &exclude,
            &files,
            log_summary.as_deref(),
            target_file_contents.as_deref(),
        ))
        .build();

    // Ask for proposal.json
    let resp = improver
        .prompt("Propose one tiny improvement now. Follow the protocol strictly.")
        .await?;
    debug!("Improver said: {}", resp);

    // Parse JSON response directly
    let proposal: ProposedChange =
        parse_ai_response_json(&resp).context("Could not parse proposal JSON from response")?;

    info!("📝 Proposal: {}", proposal.title);

    // Log proposal generation
    log_execution_trace(&ExecutionTrace {
        timestamp: Utc::now().to_rfc3339(),
        rank: profile.rank.clone(),
        phase: "PROPOSAL".to_string(),
        status: "SUCCESS".to_string(),
        details: "AI generated proposal successfully".to_string(),
        proposal_title: Some(proposal.title.clone()),
        patch_content: Some(format!(
            "--- a/{}\n+++ b/{}\n{}",
            "src/main.rs", "src/main.rs", proposal.patch
        )),
        error_details: None,
        git_status: None,
        audit_score: None,
    })
    .ok();

    // Safety rails pre-audit
    let mut changed = proposal
        .estimated_changed_lines
        .unwrap_or_else(|| changed_lines_in_patch(&proposal.patch));
    if changed == 0 {
        changed = changed_lines_in_patch(&proposal.patch);
    }
    if changed > config.max_changed_lines {
        warn!("Patch too large: {changed} > {}", config.max_changed_lines);
        return Ok(());
    }
    if likely_secret_present(&proposal.patch) {
        warn!("Local secret scan flagged the patch. Aborting.");
        return Ok(());
    }

    // Check apply
    let target_file = "src/main.rs"; // Primary target file
    
    // Validate and clean the patch before writing, with fallback to simple approach
    let patch_content = match validate_and_clean_patch(&proposal.patch, target_file) {
        Ok(cleaned) => cleaned,
        Err(validation_error) => {
            warn!("❌ Patch validation failed: {}", validation_error);
            
            // Try to create a simple unified diff if the patch has basic structure
            if proposal.patch.contains("--- a/") && proposal.patch.contains("+++ b/") {
                // The patch has the right structure, just clean it up
                let mut simple_patch = proposal.patch.clone();
                if !simple_patch.ends_with('\n') {
                    simple_patch.push('\n');
                }
                simple_patch
            } else {
                // Log validation failure and abort
                log_execution_trace(&ExecutionTrace {
                    timestamp: Utc::now().to_rfc3339(),
                    rank: profile.rank.clone(),
                    phase: "PATCH_CHECK".to_string(),
                    status: "FAILED".to_string(),
                    details: format!("Patch validation failed: {}", validation_error),
                    proposal_title: Some(proposal.title.clone()),
                    patch_content: Some(proposal.patch.clone()),
                    error_details: Some(validation_error.clone()),
                    git_status: None,
                    audit_score: None,
                }).ok();
                
                return Err(anyhow!("Patch validation failed: {}", validation_error));
            }
        }
    };
    
    fs::write(patch_path(), &patch_content)?;

    // Debug: show the patch content
    debug!("Generated patch:\n{}", proposal.patch);

    // Try to apply the patch and provide better error info
    match run(&root, "git", &["apply", "--check", patch_path()]) {
        Ok(_) => {
            info!("✅ Patch applies cleanly");
            log_execution_trace(&ExecutionTrace {
                timestamp: Utc::now().to_rfc3339(),
                rank: profile.rank.clone(),
                phase: "PATCH_CHECK".to_string(),
                status: "SUCCESS".to_string(),
                details: "Patch validation successful".to_string(),
                proposal_title: Some(proposal.title.clone()),
                patch_content: Some(proposal.patch.clone()),
                error_details: None,
                git_status: None,
                audit_score: None,
            })
            .ok();
        }
        Err(e) => {
            warn!("❌ Patch failed to apply: {}", e);
            // Show patch content for debugging
            debug!("Patch content length: {} bytes", patch_content.len());
            debug!("Patch content:\n{}", patch_content);

            // Get git status for logging
            let git_status = run(&root, "git", &["status", "--porcelain"]).unwrap_or_default();

            // Log patch failure
            log_execution_trace(&ExecutionTrace {
                timestamp: Utc::now().to_rfc3339(),
                rank: profile.rank.clone(),
                phase: "PATCH_CHECK".to_string(),
                status: "FAILED".to_string(),
                details: format!("Patch failed to apply: {}", e),
                proposal_title: Some(proposal.title.clone()),
                patch_content: Some(proposal.patch.clone()),
                error_details: Some(format!(
                    "git [\"apply\", \"--check\", \"./_architect_ai/patch.diff\"] failed: {}",
                    e
                )),
                git_status: Some(git_status.clone()),
                audit_score: None,
            })
            .ok();

            // Show git status for debugging
            if !git_status.is_empty() {
                debug!("Git status:\n{}", git_status);
            }
            // Show current working directory files
            if let Ok(files) = list_arch_files() {
                debug!("Available files: {}", files);
            }
            return Err(anyhow!("git apply --check failed: {}", e));
        }
    }

    // Apply + stage
    run(&root, "git", &["apply", patch_path()]).context("git apply failed")?;
    run(&root, "git", &["add", "-A"])?;

    // Gather post-patch files for audit (only the staged ones)
    let staged = run(&root, "git", &["diff", "--cached", "--name-only"])?;
    let mut auditor_files = String::new();
    for f in staged.lines() {
        if let Ok(s) = fs::read_to_string(f) {
            auditor_files.push_str(&format!("\n--- {}\n{}\n", f, s));
        }
    }

    // Auditor - simple approach
    let mut auditor = ds.agent(chat_model).preamble(&auditor_preamble()).build();

    let audit_input = format!(
        "PATCH:\n{}\n\nFILES AFTER PATCH:\n{}\n",
        proposal.patch, auditor_files
    );
    let audit_resp = auditor
        .prompt(&audit_input)
        .await
        .context("auditor failed")?;

    // Parse audit response
    let verdict: AuditVerdict =
        parse_ai_response_json(&audit_resp).unwrap_or_else(|_| AuditVerdict {
            ok: false,
            best_practice_score: 0.5,
            security_ok: false,
            secret_leak_risk: true,
            vuln_risk: true,
            comments: vec!["Failed to parse audit response".to_string()],
        });

    let local_secret_flag =
        likely_secret_present(&auditor_files) || likely_secret_present(&proposal.patch);
    let avg_score = verdict.best_practice_score;
    let safety_ok = verdict.ok
        && verdict.security_ok
        && !verdict.secret_leak_risk
        && !verdict.vuln_risk
        && !local_secret_flag;

    info!(
        "🔍 Audit → ok:{} best_practice:{:.2} safety_ok:{}",
        verdict.ok, avg_score, safety_ok
    );
    debug!("Audit comments: {:?}", verdict.comments);

    // Log audit results
    log_execution_trace(&ExecutionTrace {
        timestamp: Utc::now().to_rfc3339(),
        rank: profile.rank.clone(),
        phase: "AUDIT".to_string(),
        status: if safety_ok && avg_score >= config.min_audit_avg {
            "SUCCESS"
        } else {
            "FAILED"
        }
        .to_string(),
        details: format!(
            "Audit completed - ok:{} score:{:.2} safety:{}",
            verdict.ok, avg_score, safety_ok
        ),
        proposal_title: Some(proposal.title.clone()),
        patch_content: None,
        error_details: if !safety_ok || avg_score < config.min_audit_avg {
            Some(format!(
                "Audit gate failed - avg={:.2}, safety_ok={}",
                avg_score, safety_ok
            ))
        } else {
            None
        },
        git_status: None,
        audit_score: Some(avg_score),
    })
    .ok();

    if avg_score < config.min_audit_avg || (config.strict_safety && !safety_ok) {
        warn!(
            "❌ Audit gate failed (avg={:.2}, safety_ok={}). Reverting.",
            avg_score, safety_ok
        );
        run(&root, "git", &["reset", "--hard"]).ok();
        profile.total_runs += 1;
        profile.success_streak = 0;
        profile.rolling_avg = (profile.rolling_avg * 0.9).max(0.15 * avg_score);
        save_profile(&profile).ok();
        return Ok(());
    }

    // Commit & push
    let user_name = "Vadim Nicolai";
    let user_email = "nicolai.vadim@gmail.com";

    info!(
        "🔧 Setting up Git config - user: {}, email: {}",
        user_name, user_email
    );
    run(&root, "git", &["config", "user.name", user_name])?;
    run(&root, "git", &["config", "user.email", user_email])?;

    // Ensure we're on main branch and push directly to main
    let target_branch = "main";
    
    // Get current branch
    let current_branch = run(&root, "git", &["rev-parse", "--abbrev-ref", "HEAD"])?
        .trim()
        .to_string();
    
    // Switch to main if not already there
    if current_branch != target_branch {
        info!("🌿 Switching to main branch from {}", current_branch);
        run(&root, "git", &["checkout", target_branch])
            .or_else(|_| {
                info!("🌿 Main branch doesn't exist locally, creating from origin");
                run(&root, "git", &["checkout", "-b", target_branch, "origin/main"])
            })
            .context("Failed to switch to main branch")?;
    } else {
        info!("🌿 Already on main branch");
    }

    let commit_msg = format!(
        "arch: {} [rank={:?} lines={} score={:.2}]\n\n{}\n\n[audited]\n",
        proposal.title, profile.rank, changed, avg_score, proposal.rationale
    );

    info!("💾 Committing changes");
    run(&root, "git", &["commit", "-m", &commit_msg])?;

    info!("🔑 Setting up authenticated push");
    let token = env::var("GITHUB_TOKEN").map_err(|_| anyhow!("GITHUB_TOKEN not set"))?;
    let remote = origin_url(&root)?;
    let https = https_url_from_remote(&remote)
        .ok_or_else(|| anyhow!("Unsupported origin URL: {remote}"))?;
    let push_url = https.replacen("https://", &format!("https://{}@", enc(&token)), 1);

    // Decide whether we need to set upstream or just push.
    let upstream_exists = run(
        &root,
        "git",
        &["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
    )
    .is_ok();

    if upstream_exists {
        info!(
            "⬆️ Pushing to {} (branch: {})",
            sanitize_for_log(&push_url),
            target_branch
        );
        run(&root, "git", &["push", &push_url, target_branch])
            .context("Failed to push to GitHub - check your GITHUB_TOKEN permissions")?;
    } else {
        info!(
            "⬆️ Pushing (and setting upstream) to {} (branch: {})",
            sanitize_for_log(&push_url),
            target_branch
        );
        run(
            &root,
            "git",
            &["push", "--set-upstream", &push_url, target_branch],
        )
        .context("Failed to push to GitHub - check your GITHUB_TOKEN permissions")?;
    }

    // Mentor feedback (one-liner) → append to memory
    let mut mentor = ds.agent(chat_model).preamble(&mentor_preamble()).build();
    let mentor_note = mentor
        .prompt(&format!(
            "Rank: {:?}\nTitle: {}\nRationale: {}\nAudit score: {:.2}\n",
            profile.rank, proposal.title, proposal.rationale, avg_score
        ))
        .await
        .unwrap_or_else(|_| "Focus on crisp NFRs and verifiable SLOs.".to_string());

    append_memory(&MemoryItem {
        ts: Utc::now().to_rfc3339(),
        title: proposal.title.clone(),
        summary: mentor_note,
        rank: profile.rank.clone(),
        score: avg_score,
    })
    .ok();

    // Leveling logic (EMA + streak gates)
    profile.total_runs += 1;
    profile.total_success += 1;
    profile.success_streak += 1;
    profile.rolling_avg = 0.85 * profile.rolling_avg + 0.15 * avg_score;

    let (promoted, new_rank) = maybe_promote(&profile);
    if promoted {
        profile.rank = new_rank;
        profile.success_streak = 0;
        profile.last_promo_ts = Utc::now().to_rfc3339();
        info!("🏅 Promoted to {:?}", profile.rank);
    }

    save_profile(&profile).ok();

    // Log final success
    log_execution_trace(&ExecutionTrace {
        timestamp: Utc::now().to_rfc3339(),
        rank: profile.rank.clone(),
        phase: "COMPLETE".to_string(),
        status: "SUCCESS".to_string(),
        details: format!(
            "Architecture improvement completed successfully - committed to main, promoted: {}",
            promoted
        ),
        proposal_title: Some(proposal.title.clone()),
        patch_content: None,
        error_details: None,
        git_status: None,
        audit_score: Some(avg_score),
    })
    .ok();

    info!("✅ Success! Committed to main branch");
    Ok(())
}
