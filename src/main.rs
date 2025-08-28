//! AI Architecture Improver - Automated Tiny Improvements
//!
//! This tool automatically finds, audits, and commits tiny architecture improvements
//! using AI-powered analysis with ranking and memory systems.

mod config;

use std::{
    env, fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
    time::SystemTime,
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

use rig::{
    client::CompletionClient, completion::Prompt, providers::deepseek,
    vector_store::in_memory_store::InMemoryVectorStore,
};

use config::Config;

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
fn patch_path() -> &'static str {
    "./_architect_ai/patch.diff"
}
fn memory_file() -> &'static str {
    "./_architect_ai/memory.jsonl"
}
fn profile_file() -> &'static str {
    "./_architect_ai/profile.json"
}
fn execution_log_file() -> &'static str {
    "./_architect_ai/execution.log"
}

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

fn log_execution_trace(trace: &ExecutionTrace) -> Result<()> {
    fs::create_dir_all(memory_dir()).ok();
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(execution_log_file())?;
    use std::io::Write;

    // Write human-readable log format
    writeln!(
        f,
        "{} [{}] {} - {} - {}",
        trace.timestamp,
        match trace.rank {
            Rank::Junior => "jr",
            Rank::Mid => "mid",
            Rank::Senior => "sr",
        },
        trace.phase,
        trace.status,
        trace.details
    )?;

    if let Some(title) = &trace.proposal_title {
        writeln!(f, "  Proposal: {}", title)?;
    }

    if let Some(patch) = &trace.patch_content {
        writeln!(f, "  Patch:")?;
        for line in patch.lines().take(10) {
            writeln!(f, "    {}", line)?;
        }
        if patch.lines().count() > 10 {
            writeln!(f, "    ... [truncated]")?;
        }
    }

    if let Some(error) = &trace.error_details {
        writeln!(f, "  Error: {}", error)?;
    }

    if let Some(git_status) = &trace.git_status {
        writeln!(f, "  Git Status:")?;
        for line in git_status.lines() {
            writeln!(f, "    {}", line)?;
        }
    }

    if let Some(score) = trace.audit_score {
        writeln!(f, "  Audit Score: {:.3}", score)?;
    }

    writeln!(f)?; // Empty line separator
    Ok(())
}

async fn build_memory_index(
    _ds: &deepseek::Client,
    _items: &[MemoryItem],
) -> Result<InMemoryVectorStore<String>> {
    // For now, just return empty store since embedding model is not available in this version
    let store = InMemoryVectorStore::default();
    info!("Memory index initialized (basic mode)");
    Ok(store)
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
    } else if remote.starts_with("git@github.com:") {
        let tail = &remote["git@github.com:".len()..];
        Some(format!("https://github.com/{tail}"))
    } else {
        None
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

fn clean_working_state(root: &Path) -> Result<()> {
    // Reset any uncommitted changes to ensure clean state
    run(root, "git", &["reset", "--hard"]).ok();
    run(root, "git", &["clean", "-fd"]).ok();
    Ok(())
}

fn is_self_improvement_needed(log_analysis: &str) -> bool {
    let indicators = [
        "patch corruption",
        "corrupt patch",
        "patch format",
        "patch generation system",
        "fix patch generation",
        "patch application system",
        "diff generation",
    ];
    
    let analysis_lower = log_analysis.to_lowercase();
    indicators.iter().any(|indicator| analysis_lower.contains(indicator))
}

fn validate_patch_context(patch: &str, root: &Path) -> Result<()> {
    // Extract file path and context from patch
    let lines: Vec<&str> = patch.lines().collect();
    let mut target_file: Option<&str> = None;
    let mut context_lines: Vec<&str> = Vec::new();
    
    for line in &lines {
        if line.starts_with("--- a/") {
            target_file = Some(&line[6..]); // Remove "--- a/"
        } else if line.starts_with(" ") || line.starts_with("-") {
            // Context or removal line - should exist in target file
            context_lines.push(&line[1..]); // Remove the diff prefix
        }
    }
    
    if let Some(file_path) = target_file {
        let full_path = root.join(file_path);
        if full_path.exists() {
            if let Ok(file_content) = fs::read_to_string(&full_path) {
                // Check if at least some context lines exist in the file
                let found_context = context_lines.iter()
                    .filter(|line| !line.trim().is_empty())
                    .take(3) // Check first 3 non-empty context lines
                    .any(|line| file_content.contains(line.trim()));
                
                if !found_context && !context_lines.is_empty() {
                    return Err(anyhow!(
                        "Patch context doesn't match file contents. File: {}, Expected context: {:?}", 
                        file_path, 
                        context_lines.iter().take(3).collect::<Vec<_>>()
                    ));
                }
            }
        } else {
            return Err(anyhow!("Target file {} doesn't exist", file_path));
        }
    }
    
    Ok(())
}

fn get_current_file_context() -> String {
    let mut context = String::new();
    
    // Sample key files that are commonly edited
    let key_files = ["src/main.rs", "src/config.rs"];
    
    for file_path in &key_files {
        if let Ok(content) = fs::read_to_string(file_path) {
            context.push_str(&format!("\n=== Current {} (first 20 lines) ===\n", file_path));
            let lines: Vec<&str> = content.lines().take(20).collect();
            context.push_str(&lines.join("\n"));
            context.push_str("\n...[truncated]\n");
        }
    }
    
    context
}

fn sanitize_for_log(s: &str) -> String {
    Regex::new(r"[A-Za-z0-9_\-]{20,}\@[A-Za-z0-9\./:-]+")
        .unwrap()
        .replace_all(s, "***REDACTED***")
        .into_owned()
}

// ============================================================================
// Log Collection & Analysis (NEW)
// ============================================================================

fn find_recent_log_files() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let dirs = ["./logs", "./log", "./_logs", "./tmp/logs"];

    for dir in dirs {
        let path = PathBuf::from(dir);
        if !path.exists() {
            continue;
        }

        for entry in WalkDir::new(path).max_depth(3) {
            if let Ok(e) = entry {
                if !e.file_type().is_file() {
                    continue;
                }

                let p = e.path();
                let name = p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_lowercase();

                // Skip binary/temp files
                let rel = p.to_string_lossy();
                if rel.contains("/.git/") || rel.contains("/target/") {
                    continue;
                }

                // Look for log-like files
                if name.ends_with(".log")
                    || name.ends_with(".out")
                    || name.ends_with(".txt")
                    || name.contains("log")
                {
                    candidates.push(p.to_path_buf());
                }
            }
        }
    }

    // Sort by modification time (newest first)
    candidates.sort_by(|a, b| {
        let a_time = fs::metadata(a)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let b_time = fs::metadata(b)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        b_time.cmp(&a_time)
    });

    // Take only recent files (last 4)
    candidates.into_iter().take(4).collect()
}

fn read_log_tail(path: &Path, max_bytes: usize) -> String {
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return String::new(),
    };

    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    if len == 0 {
        return String::new();
    }

    let start = if len as usize > max_bytes {
        len - max_bytes as u64
    } else {
        0
    };

    if file.seek(SeekFrom::Start(start)).is_err() {
        return String::new();
    }

    let mut content = String::new();
    if file.read_to_string(&mut content).is_err() {
        return String::new();
    }

    content
}

async fn collect_log_summary_with_ai(ds: &deepseek::Client, chat_model: &str) -> Result<String> {
    let files = find_recent_log_files();

    // Also read execution logs for AI learning
    let mut all_content = String::new();

    // First, read application logs
    if !files.is_empty() {
        all_content.push_str("=== APPLICATION LOGS ===\n");
        for file in &files {
            let content = read_log_tail(file, 24 * 1024); // 24KB per app log file

            // Sanitize and filter content
            let mut sanitized_lines = Vec::new();
            for line in content.lines() {
                if likely_secret_present(line) {
                    continue;
                }
                let clean_line = sanitize_for_log(line);
                if !clean_line.trim().is_empty() {
                    sanitized_lines.push(clean_line);
                }
            }

            if !sanitized_lines.is_empty() {
                let file_name = file
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");
                all_content.push_str(&format!("\n--- {} ---\n", file_name));
                all_content.push_str(&sanitized_lines.join("\n"));
                all_content.push('\n');
            }
        }
    }

    // Then, read execution logs (architect's own history)
    let exec_log_path = PathBuf::from(execution_log_file());
    if exec_log_path.exists() {
        all_content.push_str("\n=== ARCHITECT EXECUTION HISTORY ===\n");
        let exec_content = read_log_tail(&exec_log_path, 16 * 1024);
        all_content.push_str(&exec_content);
    }

    if all_content.trim().is_empty() {
        return Ok(String::new());
    }

    // Truncate if too large for AI context
    if all_content.len() > 12000 {
        all_content.truncate(11500);
        all_content.push_str("\n...[truncated for AI analysis]...");
    }

    // Create AI agent for comprehensive log analysis
    let log_analyzer = ds
        .agent(chat_model)
        .preamble(&format!(
            r#"You are analyzing logs from an AI Architecture Improver that automatically generates, audits, and commits tiny Rust code improvements.

The logs contain:
1. APPLICATION LOGS - runtime logs from the target application
2. ARCHITECT EXECUTION HISTORY - detailed traces of past improvement attempts including successes/failures

Your analysis should focus on:
1. Recurring application errors/warnings that suggest code improvements needed
2. Patterns in architect failures (patch apply errors, audit failures) to avoid repeating mistakes
3. What types of improvements have succeeded vs failed historically
4. Specific actionable recommendations for the next improvement

Be concrete and learn from past patterns. Limit response to 400 words maximum.

Log analysis covers {} characters of recent data."#,
            all_content.len()
        ))
        .build();

    // Ask AI to analyze comprehensive logs
    let analysis = log_analyzer
        .prompt(&format!(
            "Analyze these logs and provide actionable insights for the next architecture improvement:\n\n{}",
            all_content
        ))
        .await?;

    Ok(analysis)
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
        // Focus on code files for self-improvement
        let is_code = rel.ends_with(".rs")
            || rel == "Cargo.toml"
            || rel == "README.md"
            || rel.starts_with("src/")
            || rel.ends_with(".toml")
            || rel.ends_with(".md");
        let ignore = rel.contains("/.git/")
            || rel.contains("/target/")
            || rel.contains("/node_modules/")
            || rel.contains("/.cargo/");
        if is_code && !ignore {
            out.push(rel);
        }
    }
    Ok(serde_json::to_string_pretty(&out).unwrap_or("[]".into()))
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
) -> String {
    let scope = match rank {
        Rank::Junior => "Aim for small code improvements: add missing docs, fix typos, add error handling, simple refactors.",
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
        .map(|logs| logs.contains("corrupt patch") || logs.contains("PATCH_CHECK") && logs.contains("FAILED"))
        .unwrap_or(false);

    let priority_guidance = if self_improvement_needed {
        "\n🔧 PRIORITY: Multiple patch corruption failures detected. Focus on fixing the patch generation system itself:\n\
         - Improve patch format validation\n\
         - Fix git working directory cleanup\n\
         - Enhance diff generation logic\n\
         - Add better error recovery\n\
         Target: src/main.rs patch generation functions\n"
    } else {
        ""
    };

    let file_context = get_current_file_context();

    format!(
        r#"
You are an AI Architect at **{rank}** level. Find EXACTLY ONE tiny improvement to this Rust codebase
with the smallest blast radius. Keep changes within **{max_lines} lines** total.

Available files:
{files}
{log_section}{priority_guidance}
CRITICAL - Current file contents (use this to ensure patch accuracy):
{file_context}

Scope guidance: {scope}

Examples of good improvements:
- Add missing documentation comments
- Improve error messages 
- Add helper functions to reduce duplication
- Create new utility modules (src/utils.rs, src/git.rs, etc.)
- Add better logging or debug output
- Improve function signatures or return types
- Add configuration options
- Small performance improvements

Guardrails:
- No breaking changes to public APIs
- No secrets, credentials, or tokens in code
- Ensure code compiles and tests pass
- Focus on maintainability and readability
- Focus includes: {include}
- Avoid paths containing: {exclude}

When log analysis shows recurring errors/warnings, prefer improvements that address those issues.

Output JSON format (respond with just the JSON, no fences):
{{
  "title": "...",
  "rationale": "...", 
  "patch": "<unified diff repo-relative>",
  "estimated_changed_lines": <int|null>
}}

The unified diff MUST apply cleanly with 'git apply --check'.
Make sure file paths in the diff are relative to the repo root.
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
        file_context = file_context
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
- Prefer clear documentation, proper error handling, good naming conventions.
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
        warn!("❌ Architecture analysis failed: {}", e);

        // Log the high-level failure to execution log
        log_execution_trace(&ExecutionTrace {
            timestamp: Utc::now().to_rfc3339(),
            rank: load_profile().rank, // Get current rank for context
            phase: "CRON_FAILURE".to_string(),
            status: "FAILED".to_string(),
            details: format!("Architecture analysis failed in cron job: {}", e),
            proposal_title: None,
            patch_content: None,
            error_details: Some(e.to_string()),
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
            .map_err(|e| shuttle_runtime::Error::Custom(anyhow::Error::msg(e).into()))?;

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
    let chat_model =
        env::var("DEEPSEEK_CHAT_MODEL").unwrap_or_else(|_| "deepseek-chat".to_string());

    // Repo setup
    let root = repo_root()?;
    env::set_current_dir(&root)?;
    fs::create_dir_all(memory_dir()).ok();

    // Profile + memory + index
    let mut profile = load_profile();
    let memory = load_memory();
    let _store = build_memory_index(&ds, &memory).await?;

    // Log start of analysis
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

    // NEW: Collect log insights using AI to guide improvements
    let log_summary = match collect_log_summary_with_ai(&ds, &chat_model).await {
        Ok(summary) => summary,
        Err(e) => {
            warn!("📊 Failed to analyze logs with AI: {}", e);
            log_execution_trace(&ExecutionTrace {
                timestamp: Utc::now().to_rfc3339(),
                rank: profile.rank.clone(),
                phase: "LOG_ANALYSIS".to_string(),
                status: "FAILED".to_string(),
                details: "Failed to analyze logs with AI".to_string(),
                proposal_title: None,
                patch_content: None,
                error_details: Some(e.to_string()),
                git_status: None,
                audit_score: None,
            })
            .ok();
            String::new()
        }
    };

    let log_context = if log_summary.is_empty() {
        info!("📊 No recent logs found for analysis guidance");
        None
    } else {
        info!("📊 AI log analysis complete - guiding improvements with runtime insights");
        debug!("AI Log analysis: {}", log_summary);
        Some(log_summary.as_str())
    };

    let improver = ds
        .agent(&chat_model)
        .preamble(&improver_preamble(
            &profile.rank,
            config.max_changed_lines,
            &include,
            &exclude,
            &files,
            log_context,
        ))
        .build();

    // Ask for proposal.json
    let resp = improver
        .prompt("Propose one tiny improvement now. Follow the protocol strictly.")
        .await?;
    debug!("Improver said: {}", resp);

    // Parse JSON response directly
    let raw = resp
        .trim()
        .trim_start_matches("```json")
        .trim_end_matches("```")
        .trim();
    let proposal: ProposedChange = serde_json::from_str(raw)
        .or_else(|_| {
            // Try to extract JSON from response
            let json_start = raw.find('{').unwrap_or(0);
            let json_end = raw.rfind('}').map(|i| i + 1).unwrap_or(raw.len());
            serde_json::from_str(&raw[json_start..json_end])
        })
        .map_err(|e| {
            // Log JSON parsing failure
            log_execution_trace(&ExecutionTrace {
                timestamp: Utc::now().to_rfc3339(),
                rank: profile.rank.clone(),
                phase: "PROPOSAL".to_string(),
                status: "FAILED".to_string(),
                details: "Failed to parse AI proposal JSON response".to_string(),
                proposal_title: None,
                patch_content: None,
                error_details: Some(format!("JSON parse error: {} | Raw response: {}", e, raw)),
                git_status: None,
                audit_score: None,
            })
            .ok();
            e
        })
        .context("Could not parse proposal JSON from response")?;

    info!("📝 Proposal: {}", proposal.title);

    // Validate patch context before proceeding
    if let Err(e) = validate_patch_context(&proposal.patch, &root) {
        warn!("🚫 Patch validation failed: {}", e);
        log_execution_trace(&ExecutionTrace {
            timestamp: Utc::now().to_rfc3339(),
            rank: profile.rank.clone(),
            phase: "VALIDATION".to_string(),
            status: "FAILED".to_string(),
            details: "Patch context validation failed - content doesn't match actual files".to_string(),
            proposal_title: Some(proposal.title.clone()),
            patch_content: Some(proposal.patch.clone()),
            error_details: Some(e.to_string()),
            git_status: None,
            audit_score: None,
        }).ok();
        return Ok(());
    }

    // Log proposal generation
    log_execution_trace(&ExecutionTrace {
        timestamp: Utc::now().to_rfc3339(),
        rank: profile.rank.clone(),
        phase: "PROPOSAL".to_string(),
        status: "SUCCESS".to_string(),
        details: "AI generated proposal successfully".to_string(),
        proposal_title: Some(proposal.title.clone()),
        patch_content: Some(proposal.patch.clone()),
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
        log_execution_trace(&ExecutionTrace {
            timestamp: Utc::now().to_rfc3339(),
            rank: profile.rank.clone(),
            phase: "VALIDATION".to_string(),
            status: "REJECTED".to_string(),
            details: format!(
                "Patch too large: {} > {} lines",
                changed, config.max_changed_lines
            ),
            proposal_title: Some(proposal.title.clone()),
            patch_content: Some(proposal.patch.clone()),
            error_details: None,
            git_status: None,
            audit_score: None,
        })
        .ok();
        return Ok(());
    }
    if likely_secret_present(&proposal.patch) {
        warn!("Local secret scan flagged the patch. Aborting.");
        log_execution_trace(&ExecutionTrace {
            timestamp: Utc::now().to_rfc3339(),
            rank: profile.rank.clone(),
            phase: "VALIDATION".to_string(),
            status: "REJECTED".to_string(),
            details: "Patch rejected due to potential secret content".to_string(),
            proposal_title: Some(proposal.title.clone()),
            patch_content: Some(proposal.patch.clone()),
            error_details: None,
            git_status: None,
            audit_score: None,
        })
        .ok();
        return Ok(());
    }

    // Check apply
    fs::write(patch_path(), &proposal.patch)?;

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
                details: "Patch applies cleanly".to_string(),
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
            // Show git status for debugging
            let git_status = run(&root, "git", &["status", "--porcelain"]).ok();
            if let Some(ref status) = git_status {
                debug!("Git status:\n{}", status);
            }
            // Show current working directory files
            if let Ok(files) = list_arch_files() {
                debug!("Available files: {}", files);
            }

            // Log the failure with full details
            log_execution_trace(&ExecutionTrace {
                timestamp: Utc::now().to_rfc3339(),
                rank: profile.rank.clone(),
                phase: "PATCH_CHECK".to_string(),
                status: "FAILED".to_string(),
                details: format!("Patch failed to apply: {}", e),
                proposal_title: Some(proposal.title.clone()),
                patch_content: Some(proposal.patch.clone()),
                error_details: Some(e.to_string()),
                git_status,
                audit_score: None,
            })
            .ok();

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
    let auditor = ds.agent(&chat_model).preamble(&auditor_preamble()).build();

    let audit_input = format!(
        "PATCH:\n{}\n\nFILES AFTER PATCH:\n{}\n",
        proposal.patch, auditor_files
    );
    let audit_resp = auditor
        .prompt(&audit_input)
        .await
        .context("auditor failed")?;

    // Parse audit response
    let audit_json = audit_resp
        .trim()
        .trim_start_matches("```json")
        .trim_end_matches("```")
        .trim();
    let verdict: AuditVerdict = serde_json::from_str(audit_json)
        .or_else(|_| {
            let json_start = audit_json.find('{').unwrap_or(0);
            let json_end = audit_json
                .rfind('}')
                .map(|i| i + 1)
                .unwrap_or(audit_json.len());
            serde_json::from_str(&audit_json[json_start..json_end])
        })
        .unwrap_or_else(|_| AuditVerdict {
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
        status: if avg_score >= config.min_audit_avg && (!config.strict_safety || safety_ok) {
            "SUCCESS".to_string()
        } else {
            "FAILED".to_string()
        },
        details: format!(
            "Audit complete - ok:{} score:{:.2} safety:{} comments:{:?}",
            verdict.ok, avg_score, safety_ok, verdict.comments
        ),
        proposal_title: Some(proposal.title.clone()),
        patch_content: None,
        error_details: if avg_score < config.min_audit_avg || (config.strict_safety && !safety_ok) {
            Some(format!(
                "Audit gate failed - score:{:.2} safety:{}",
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
    let user_name = env::var("GIT_USER_NAME").unwrap_or_else(|_| "architect-ai-improver".into());
    let user_email =
        env::var("GIT_USER_EMAIL").unwrap_or_else(|_| "architect-ai@github.com".into());

    info!(
        "🔧 Setting up Git config - user: {}, email: {}",
        user_name, user_email
    );
    run(&root, "git", &["config", "user.name", &user_name])?;
    run(&root, "git", &["config", "user.email", &user_email])?;

    // Ensure we're on main branch for direct commits
    info!("🌿 Ensuring we're on main branch for direct commit");
    run(&root, "git", &["checkout", "main"])?;

    let commit_msg = format!(
        "arch: {} [rank={:?} lines={} score={:.2}{}]\n\n{}\n\n[audited]\n",
        proposal.title,
        profile.rank,
        changed,
        avg_score,
        if log_context.is_some() {
            " log-informed"
        } else {
            ""
        },
        proposal.rationale
    );

    info!("💾 Committing changes directly to main");
    run(&root, "git", &["commit", "-m", &commit_msg])?;

    info!("🔑 Setting up authenticated push");
    let token = env::var("GITHUB_TOKEN").map_err(|_| anyhow!("GITHUB_TOKEN not set"))?;
    let remote = origin_url(&root)?;
    let https = https_url_from_remote(&remote)
        .ok_or_else(|| anyhow!("Unsupported origin URL: {remote}"))?;
    let push_url = https.replacen("https://", &format!("https://{}@", enc(&token)), 1);

    info!("⬆️ Pushing to main branch on {}", sanitize_for_log(&push_url));
    run(
        &root,
        "git",
        &["push", &push_url, "main"],
    )
    .context("Failed to push to GitHub - check your GITHUB_TOKEN permissions")?;

    // Mentor feedback (one-liner) → append to memory
    let mentor = ds.agent(&chat_model).preamble(&mentor_preamble()).build();
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
            "Architecture improvement completed successfully - branch: {}, promoted: {}",
            branch, promoted
        ),
        proposal_title: Some(proposal.title.clone()),
        patch_content: None,
        error_details: None,
        git_status: None,
        audit_score: Some(avg_score),
    })
    .ok();

    info!("✅ Success! Branch: {}", branch);
    Ok(())
}
