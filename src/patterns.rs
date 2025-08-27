//! Bug detection patterns for the Rig codebase
//!
//! This module contains pattern definitions for identifying code issues
//! and the scanner engine for processing Rust source files.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Issue {
    pub pattern_id: &'static str,
    pub name: &'static str,
    pub severity: &'static str,
    pub category: &'static str,
    pub line: usize,
    pub col: usize,
    pub excerpt: String,
    pub severity_confirmed: bool,
    pub confidence_score: f64,
}

#[derive(Debug, Copy, Clone)]
pub enum Severity {
    High,
    Critical,
}
impl Severity {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Severity::High => "High",
            Severity::Critical => "Critical",
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Category {
    Panic,
    Unsafe,
    Pointer,
    Concurrency,
    Async,
    ErrorHandling,
    Security,
    Casting,
    Float,
    FS,
    Process,
    Crypto,
    Deprecated,
    Performance,
    Style,
    API,
    Time,
}
impl Category {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Category::Panic => "Panic",
            Category::Unsafe => "Unsafe",
            Category::Pointer => "Pointer",
            Category::Concurrency => "Concurrency",
            Category::Async => "Async",
            Category::ErrorHandling => "ErrorHandling",
            Category::Security => "Security",
            Category::Casting => "Casting",
            Category::Float => "Float",
            Category::FS => "FS",
            Category::Process => "Process",
            Category::Crypto => "Crypto",
            Category::Deprecated => "Deprecated",
            Category::Performance => "Performance",
            Category::Style => "Style",
            Category::API => "API",
            Category::Time => "Time",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatternDef {
    pub id: &'static str,
    pub name: &'static str,
    pub severity: Severity,
    pub category: Category,
    /// Main regex, matched line-by-line
    pub expr: &'static str,
    /// Additional regexes that must appear **somewhere** in the file
    pub requires_all: &'static [&'static str],
}

macro_rules! p {
    ($id:literal, $name:literal, $sev:ident, $cat:ident, $expr:literal) => {
        PatternDef {
            id: $id,
            name: $name,
            severity: Severity::$sev,
            category: Category::$cat,
            expr: $expr,
            requires_all: &[],
        }
    };
    ($id:literal, $name:literal, $sev:ident, $cat:ident, $expr:literal, [ $($req:literal),* $(,)? ]) => {
        PatternDef {
            id: $id,
            name: $name,
            severity: Severity::$sev,
            category: Category::$cat,
            expr: $expr,
            requires_all: &[$($req),*],
        }
    };
}

// ≥220 patterns total. Original 120 + 100 performance patterns (R121–R220).
static PATTERNS: &[PatternDef] = &[
    // Panic & assertions
    p!("R001", "panic!", Critical, Panic, r#"\bpanic!\s*\("#),
    p!("R002", "todo!", High, Panic, r#"\btodo!\s*\("#),
    p!(
        "R003",
        "unimplemented!",
        High,
        Panic,
        r#"\bunimplemented!\s*\("#
    ),
    // p!("R004", "assert!", High, Panic, r#"\bassert!\s*\("#), // Disabled - common in tests
    // p!("R005", "assert_eq!", High, Panic, r#"\bassert_eq!\s*\("#), // Disabled - common in tests
    // p!("R006", "assert_ne!", High, Panic, r#"\bassert_ne!\s*\("#), // Disabled - common in tests
    p!(
        "R007",
        "unreachable!",
        High,
        Panic,
        r#"\bunreachable!\s*\("#
    ),
    p!("R008", "unwrap()", High, ErrorHandling, r#"\.unwrap\s*\("#),
    p!("R009", "expect()", High, ErrorHandling, r#"\.expect\s*\("#),
    p!(
        "R010",
        "unwrap_unchecked()",
        Critical,
        Unsafe,
        r#"\bunwrap_unchecked\s*\("#
    ),
    // Unsafe & pointer primitives
    p!(
        "R011",
        "get_unchecked()",
        Critical,
        Unsafe,
        r#"\bget_unchecked\s*\("#
    ),
    p!(
        "R012",
        "get_unchecked_mut()",
        Critical,
        Unsafe,
        r#"\bget_unchecked_mut\s*\("#
    ),
    p!(
        "R013",
        "mem::transmute",
        Critical,
        Unsafe,
        r#"\bstd::mem::transmute\b"#
    ),
    p!(
        "R014",
        "mem::zeroed",
        Critical,
        Unsafe,
        r#"\bstd::mem::zeroed\s*\("#
    ),
    p!(
        "R015",
        "mem::uninitialized",
        Critical,
        Deprecated,
        r#"\bstd::mem::uninitialized\s*\("#
    ),
    p!(
        "R016",
        "from_utf8_unchecked",
        Critical,
        Unsafe,
        r#"\bfrom_utf8_unchecked\s*\("#
    ),
    p!(
        "R017",
        "slice::from_raw_parts",
        Critical,
        Pointer,
        r#"\bslice::from_raw_parts\s*\("#
    ),
    p!(
        "R018",
        "slice::from_raw_parts_mut",
        Critical,
        Pointer,
        r#"\bslice::from_raw_parts_mut\s*\("#
    ),
    p!(
        "R019",
        "ptr::null",
        High,
        Pointer,
        r#"\bstd::ptr::null\s*\("#
    ),
    p!(
        "R020",
        "ptr::null_mut",
        High,
        Pointer,
        r#"\bstd::ptr::null_mut\s*\("#
    ),
    p!(
        "R021",
        "ptr::read",
        Critical,
        Pointer,
        r#"\bstd::ptr::read\s*\("#
    ),
    p!(
        "R022",
        "ptr::write",
        Critical,
        Pointer,
        r#"\bstd::ptr::write\s*\("#
    ),
    p!(
        "R023",
        "ptr::copy",
        Critical,
        Pointer,
        r#"\bstd::ptr::copy\s*\("#
    ),
    p!(
        "R024",
        "ptr::copy_nonoverlapping",
        Critical,
        Pointer,
        r#"\bcopy_nonoverlapping\s*\("#
    ),
    p!(
        "R025",
        "static mut",
        Critical,
        Unsafe,
        r#"\bstatic\s+mut\b"#
    ),
    p!(
        "R026",
        "mem::forget",
        High,
        Unsafe,
        r#"\bstd::mem::forget\s*\("#
    ),
    p!(
        "R027",
        "Box::leak",
        High,
        Performance,
        r#"\bBox::leak\s*\("#
    ),
    p!(
        "R028",
        "Arc<UnsafeCell>",
        Critical,
        Concurrency,
        r#"\bArc\s*<\s*UnsafeCell\b"#
    ),
    // Concurrency combos
    p!(
        "R029",
        "Rc<RefCell> + threads",
        Critical,
        Concurrency,
        r#"\bRc\s*<\s*RefCell\b"#,
        [r#"\bthread::spawn\s*\("#]
    ),
    p!(
        "R030",
        "Mutex::lock().unwrap()",
        High,
        Concurrency,
        r#"\bMutex::lock\s*\(\s*\)\s*\.unwrap\s*\("#
    ),
    p!(
        "R031",
        "RwLock::read().unwrap()",
        High,
        Concurrency,
        r#"\bRwLock::read\s*\(\s*\)\s*\.unwrap\s*\("#
    ),
    p!(
        "R032",
        "RwLock::write().unwrap()",
        High,
        Concurrency,
        r#"\bRwLock::write\s*\(\s*\)\s*\.unwrap\s*\("#
    ),
    p!(
        "R033",
        "thread::sleep in async fn",
        High,
        Async,
        r#"\bthread::sleep\s*\("#,
        [r#"\basync\s+fn\b"#]
    ),
    p!(
        "R034",
        "blocking fs in async fn",
        High,
        Async,
        r#"\bstd::fs::(read|write|File::open)\s*\("#,
        [r#"\basync\s+fn\b"#]
    ),
    p!("R035", "dbg! left", High, Style, r#"\bdbg!\s*\("#),
    p!("R036", "eprintln! left", High, Style, r#"\beprintln!\s*\("#),
    p!(
        "R037",
        "expect(\"TODO|FIXME\")",
        High,
        ErrorHandling,
        r#"\.expect\s*\(\s*"(?i)(todo|fixme)[^"]*"\s*\)"#
    ),
    p!(
        "R038",
        "env::var(...).unwrap()",
        High,
        ErrorHandling,
        r#"\benv::var\s*\([^)]*\)\s*\.unwrap\s*\("#
    ),
    p!(
        "R039",
        "parse().unwrap()",
        High,
        ErrorHandling,
        r#"\.parse\s*::<[^>]+>\s*\(\s*\)\s*\.unwrap\s*\(|\.parse\s*\(\s*\)\s*\.unwrap\s*\("#
    ),
    p!(
        "R040",
        "parse().expect()",
        High,
        ErrorHandling,
        r#"\.parse\s*(::?<[^>]+>)?\s*\(\s*\)\s*\.expect\s*\("#
    ),
    p!(
        "R041",
        "as numeric cast",
        High,
        Casting,
        r#"\bas\s+(u|i)(8|16|32|64|128)|\bas\s+(f32|f64|usize|isize)\b"#
    ),
    p!(
        "R042",
        "ptr.offset()",
        Critical,
        Pointer,
        r#"\.offset\s*\("#
    ),
    p!(
        "R043",
        "as_mut_ptr()",
        High,
        Pointer,
        r#"\bas_mut_ptr\s*\("#
    ),
    p!(
        "R044",
        "mem::transmute_copy",
        Critical,
        Unsafe,
        r#"\btransmute_copy\s*\("#
    ),
    p!(
        "R045",
        "JoinHandle.join().unwrap()",
        High,
        Concurrency,
        r#"\bjoin\s*\(\s*\)\s*\.unwrap\s*\("#
    ),
    p!(
        "R046",
        "tokio::spawn not awaited",
        High,
        Async,
        r#"\btokio::spawn\s*\([^)]*\)\s*;"#
    ),
    // Float traps
    p!(
        "R047",
        "compare to NaN",
        High,
        Float,
        r#"(==|!=)\s*f(32|64)::NAN"#
    ),
    p!(
        "R048",
        "direct float equality by literal",
        High,
        Float,
        r#"\b\d+\.\d+\s*==\s*\d+\.\d+\b"#
    ),
    // Security: secrets & crypto
    p!(
        "R049",
        "hard-coded API key",
        Critical,
        Security,
        r#"(?i)\b(api[_-]?key|secret|passwd|password|token)\s*=\s*"[^"]+""#
    ),
    p!(
        "R050",
        "AWS Access Key",
        Critical,
        Security,
        r#"\bAKIA[0-9A-Z]{16}\b"#
    ),
    p!(
        "R051",
        "AWS Secret Key marker",
        Critical,
        Security,
        r#"(?i)aws_secret_access_key"#
    ),
    p!(
        "R052",
        "Private Key PEM",
        Critical,
        Security,
        r#"-----BEGIN (RSA )?PRIVATE KEY-----"#
    ),
    p!(
        "R053",
        "JWT present",
        High,
        Security,
        r#"\beyJ[0-9A-Za-z_-]+\.[0-9A-Za-z_-]+\.[0-9A-Za-z_-]+\b"#
    ),
    p!(
        "R054",
        "danger_accept_invalid_certs(true)",
        Critical,
        Security,
        r#"\bdanger_accept_invalid_certs\s*\(\s*true\s*\)"#
    ),
    p!(
        "R055",
        "SslVerifyMode::NONE",
        Critical,
        Security,
        r#"\bSslVerifyMode::NONE\b"#
    ),
    p!(
        "R056",
        "dangerous()",
        High,
        Security,
        r#"\bdangerous\s*\(\s*\)"#
    ),
    p!(
        "R057",
        "Command \"sh -c\"",
        High,
        Process,
        r#"\bCommand::new\s*\(\s*"(sh|/bin/sh)"\s*\)\s*\.arg\s*\(\s*"-c""#
    ),
    p!(
        "R058",
        "Command \"cmd /C\"",
        High,
        Process,
        r#"\bCommand::new\s*\(\s*"cmd(\.exe)?"\s*\)\s*\.args?\s*\(\s*\[\s*"/C""#
    ),
    p!(
        "R059",
        "tmp file create",
        High,
        FS,
        r#"\bFile::create\s*\(\s*"/tmp/"#
    ),
    p!(
        "R060",
        "0o777 mode",
        High,
        FS,
        r#"\bOpenOptions::new\s*\(\s*\)\s*\.mode\s*\(\s*0o?777\s*\)"#
    ),
    p!(
        "R061",
        "remove_dir_all",
        High,
        FS,
        r#"\bremove_dir_all\s*\("#
    ),
    p!(
        "R062",
        "Regex::new(..).unwrap()",
        High,
        ErrorHandling,
        r#"\bRegex::new\s*\([^)]*\)\s*\.unwrap\s*\("#
    ),
    p!(
        "R063",
        "duration_since(..).unwrap()",
        High,
        Time,
        r#"\bduration_since\s*\([^)]*\)\s*\.unwrap\s*\("#
    ),
    // Deprecated
    p!("R064", "try! macro", High, Deprecated, r#"\btry!\s*\("#),
    p!(
        "R065",
        "Error::description()",
        High,
        Deprecated,
        r#"\bdescription\s*\(\s*\)"#
    ),
    // Style / leftover debug / lint masking
    p!("R066", "println! left", High, Style, r#"\bprintln!\s*\("#),
    p!("R067", "TODO comment", High, Style, r#"(?i)\bTODO\b"#),
    p!("R068", "FIXME comment", High, Style, r#"(?i)\bFIXME\b"#),
    p!(
        "R069",
        "allow(dead_code)",
        High,
        Style,
        r#"\#\!\[\s*allow\s*\(\s*dead_code\s*\)\s*\]"#
    ),
    p!(
        "R070",
        "allow(unused)",
        High,
        Style,
        r#"\#\!\[\s*allow\s*\(\s*unused[^\)]*\)\s*\]"#
    ),
    p!(
        "R071",
        "allow(clippy::unwrap_used)",
        High,
        Style,
        r#"\#\[\s*allow\s*\(\s*clippy::unwrap_used\s*\)\s*\]"#
    ),
    // Async combos
    p!(
        "R072",
        "reqwest::blocking in async",
        High,
        Async,
        r#"\breqwest::blocking::"#,
        [r#"\basync\s+fn\b"#]
    ),
    p!(
        "R073",
        "async_std::task::block_on in async",
        High,
        Async,
        r#"\basync_std::task::block_on\s*\("#,
        [r#"\basync\s+fn\b"#]
    ),
    p!(
        "R074",
        "tokio::task::block_in_place",
        High,
        Async,
        r#"\btokio::task::block_in_place\s*\("#
    ),
    // Performance / correctness
    p!(
        "R075",
        "filter_map(Some)",
        High,
        Performance,
        r#"\.filter_map\s*\(\s*\|\s*[^|]*\s*\|\s*Some\("#
    ),
    p!(
        "R076",
        "map(Some)+flatten",
        High,
        Performance,
        r#"\.map\s*\(\s*\|\s*[^|]*\s*\|\s*Some\([^)]+\)\s*\)\s*\.flatten\s*\("#
    ),
    p!(
        "R077",
        "NaiveDateTime",
        High,
        Time,
        r#"\bchrono::Naive(Date|DateTime)\b"#
    ),
    p!(
        "R078",
        "SystemTime::now()",
        High,
        Time,
        r#"\bSystemTime::now\s*\("#
    ),
    p!(
        "R079",
        "HTTP over insecure http://",
        High,
        Security,
        r#"http://[^\s")]+""#
    ),
    p!(
        "R080",
        "process::exit",
        High,
        Process,
        r#"\bstd::process::exit\s*\("#
    ),
    p!(
        "R081",
        "catch_unwind + unsafe",
        Critical,
        Unsafe,
        r#"\bcatch_unwind\s*\(\s*\|\|\s*unsafe\b"#
    ),
    p!(
        "R082",
        "futures::mpsc::unbounded()",
        High,
        Concurrency,
        r#"\bfutures::channel::mpsc::unbounded\s*\("#
    ),
    p!(
        "R083",
        "tokio::mpsc::unbounded_channel()",
        High,
        Concurrency,
        r#"\btokio::sync::mpsc::unbounded_channel\s*\("#
    ),
    p!(
        "R084",
        "std::sync::mpsc::channel()",
        High,
        Concurrency,
        r#"\bstd::sync::mpsc::channel\s*\("#
    ),
    // More error handling smells
    p!(
        "R085",
        "discard Result with `let _ =`",
        High,
        ErrorHandling,
        r#"\blet\s+_\s*=\s*[^;]+;"#
    ),
    p!(
        "R086",
        "unwrap_or_default()",
        High,
        ErrorHandling,
        r#"\.unwrap_or_default\s*\("#
    ),
    p!(
        "R087",
        "unwrap_or(Default::default())",
        High,
        ErrorHandling,
        r#"\.unwrap_or\s*\(\s*Default::default\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R088",
        "Err(_) => {}",
        High,
        ErrorHandling,
        r#"Err\s*\(\s*_\s*\)\s*=>\s*\{\s*\}"#
    ),
    p!(
        "R089",
        "if let Err(_) { }",
        High,
        ErrorHandling,
        r#"if\s+let\s+Err\s*\(\s*_\s*\)\s*=\s*[^;]+?\{\s*\}"#
    ),
    p!(
        "R090",
        "downcast_ref().unwrap()",
        High,
        ErrorHandling,
        r#"\bdowncast_ref\s*::<[^>]+>\s*\(\s*\)\s*\.unwrap\s*\("#
    ),
    // More unsafe things
    p!(
        "R091",
        "Vec::set_len",
        Critical,
        Unsafe,
        r#"\bset_len\s*\("#
    ),
    p!(
        "R092",
        "MaybeUninit::assume_init()",
        Critical,
        Unsafe,
        r#"\bassume_init\s*\("#
    ),
    p!(
        "R093",
        "unsafe impl Send",
        Critical,
        Unsafe,
        r#"\bunsafe\s+impl\s+Send\b"#
    ),
    p!(
        "R094",
        "unsafe impl Sync",
        Critical,
        Unsafe,
        r#"\bunsafe\s+impl\s+Sync\b"#
    ),
    // Casting/logic
    p!(
        "R095",
        "wrapping_add misuse",
        High,
        Casting,
        r#"\bwrapping_add\s*\("#
    ),
    p!(
        "R096",
        "wrapping_sub misuse",
        High,
        Casting,
        r#"\bwrapping_sub\s*\("#
    ),
    // Channels / joins
    p!(
        "R097",
        "recv().unwrap()",
        High,
        Concurrency,
        r#"\brecv\s*\(\s*\)\s*\.unwrap\s*\("#
    ),
    // SQL string building via format!
    p!(
        "R098",
        "format!(SQL with {})",
        High,
        Security,
        r#"\bformat!\s*\(\s*"(?i)\s*select\s+.*\{\}\s*"#
    ),
    // Randomness in security-sensitive usage (heuristic)
    p!(
        "R099",
        "rand::random::<..>()",
        High,
        Security,
        r#"\brand::random\s*::<[^>]+>\s*\("#
    ),
    // Env secrets reading without handling
    p!(
        "R100",
        "env secret read",
        High,
        Security,
        r#"env::var\s*\(\s*"(?i).*(password|token|secret).+"\s*\)"#
    ),
    // Style / API risks
    p!(
        "R101",
        "glob import",
        High,
        Style,
        r#"\buse\s+[A-Za-z0-9_:]+::\*\s*;"#
    ),
    p!(
        "R102",
        "allow(non_snake_case)",
        High,
        Style,
        r#"\#\[\s*allow\s*\(\s*non_snake_case\s*\)\s*\]"#
    ),
    p!(
        "R103",
        "allow(unused_mut)",
        High,
        Style,
        r#"\#\[\s*allow\s*\(\s*unused_mut\s*\)\s*\]"#
    ),
    // Process with interpolation (shell)
    p!(
        "R104",
        "sh -c with format!",
        High,
        Process,
        r#"\bCommand::new\s*\(\s*"(sh|/bin/sh)"\s*\)[^;]*format!\s*\("#
    ),
    // HTTP client disabling cert verify variants (another libs)
    p!(
        "R105",
        "ureq tls insecure",
        Critical,
        Security,
        r#"\b(tls|ssl).*insecure\b"#
    ),
    // panics as control-flow messages
    p!(
        "R106",
        "expect(\"never fails\")",
        High,
        Panic,
        r#"\.expect\s*\(\s*"(?i)never\s+fails""#
    ),
    // Thread sleep as retry
    p!(
        "R107",
        "sleep retry",
        High,
        Concurrency,
        r#"\bthread::sleep\s*\(\s*Duration::from_(millis|secs)\s*\("#
    ),
    // HTTP hard-coded credentials in URL
    p!(
        "R108",
        "http basic in URL",
        Critical,
        Security,
        r#"http://[^/\s:@]+:[^/\s:@]+@"#
    ),
    // Leak via mem::forget on Arc/Mutex (heuristic)
    p!(
        "R109",
        "forget on Arc",
        High,
        Performance,
        r#"\bstd::mem::forget\s*\(\s*Arc::new"#
    ),
    // Unsafe Cell direct usage
    p!(
        "R110",
        "UnsafeCell direct",
        High,
        Unsafe,
        r#"\bUnsafeCell\s*<"#
    ),
    // More async mishaps
    p!(
        "R111",
        "blocking read_to_string in async",
        High,
        Async,
        r#"\bread_to_string\s*\("#,
        [r#"\basync\s+fn\b"#]
    ),
    // Process: spawn then drop handle immediately
    p!(
        "R112",
        "spawn() dropped",
        High,
        Process,
        r#"\bstd::process::Command::new[^;]+\.spawn\s*\(\s*\)\s*; "#
    ),
    // Time: Instant::now() used for wallclock (heuristic)
    p!(
        "R113",
        "Instant::now()",
        High,
        Time,
        r#"\bInstant::now\s*\("#
    ),
    // FS: write with create(true).truncate(true) risky
    p!(
        "R114",
        "OpenOptions trunc",
        High,
        FS,
        r#"\bOpenOptions::new\s*\(\s*\)\s*\.write\s*\(\s*true\s*\)\s*\.create\s*\(\s*true\s*\)\s*\.truncate\s*\(\s*true\s*\)"#
    ),
    // SQL using string concat '+'
    p!(
        "R115",
        "SQL concat +",
        High,
        Security,
        r#""(?i)\s*select\s+.*"\s*\+\s*"#
    ),
    // Casting float->int via as
    p!(
        "R116",
        "float->int as",
        High,
        Casting,
        r#"\bas\s+(u|i)(8|16|32|64|128)\b"#
    ),
    // unwrap in Drop (heuristic: impl Drop { .. unwrap( .. ) }
    p!(
        "R117",
        "unwrap in Drop",
        High,
        ErrorHandling,
        r#"\bimpl\s+Drop\b[^}]*\.unwrap\s*\("#
    ),
    // any::downcast::<T>().unwrap()
    p!(
        "R118",
        "Box<dyn Any> unwrap",
        High,
        ErrorHandling,
        r#"\bdowncast\s*::<[^>]+>\s*\(\s*\)\s*\.unwrap\s*\("#
    ),
    // quick-and-dirty PEM private RSA key detection variant
    p!(
        "R119",
        "RSA PRIVATE KEY",
        Critical,
        Security,
        r#"-----BEGIN RSA PRIVATE KEY-----"#
    ),
    // unsafe { .. } block (generic)
    p!("R120", "unsafe block", High, Unsafe, r#"\bunsafe\s*\{"#),
    // ---------------------------------------------------------------------
    // Performance-only patterns (R121–R220)
    // ---------------------------------------------------------------------
    p!("R121", "clone()", High, Performance, r#"\.clone\s*\("#),
    p!(
        "R122",
        r#"literal.to_string()"#,
        High,
        Performance,
        r#""[^"]*"\.to_string\s*\("#
    ),
    p!(
        "R123",
        "format!(const)",
        High,
        Performance,
        r#"\bformat!\s*\(\s*"[^{}]*"\s*\)"#
    ),
    p!(
        "R124",
        "format!(one slot)",
        High,
        Performance,
        r#"\bformat!\s*\(\s*"[^"]*\{\}[^"]*"\s*,\s*[^,]+\s*\)"#
    ),
    p!(
        "R125",
        "println!(format!)",
        High,
        Performance,
        r#"\bprintln!\s*\(\s*format!\s*\("#
    ),
    p!(
        "R126",
        "Vec::new() no capacity",
        High,
        Performance,
        r#"\bVec::new\s*\("#
    ),
    p!(
        "R127",
        "String::new() no capacity",
        High,
        Performance,
        r#"\bString::new\s*\("#
    ),
    p!(
        "R128",
        "HashMap::new() no capacity",
        High,
        Performance,
        r#"\bHashMap::new\s*\("#
    ),
    p!(
        "R129",
        "BTreeMap::new()",
        High,
        Performance,
        r#"\bBTreeMap::new\s*\("#
    ),
    p!(
        "R130",
        "collect::<Vec<_>>()",
        High,
        Performance,
        r#"\.collect::<\s*Vec<[^>]*>\s*>\s*\("#
    ),
    p!(
        "R131",
        "collect::<String>()",
        High,
        Performance,
        r#"\.collect::<\s*String\s*>\s*\("#
    ),
    p!(
        "R132",
        "collect::<Vec<char>>()",
        High,
        Performance,
        r#"\.collect::<\s*Vec\s*<\s*char\s*>\s*>\s*\("#
    ),
    p!("R133", "cloned()", High, Performance, r#"\.cloned\s*\("#),
    p!(
        "R134",
        "to_owned()",
        High,
        Performance,
        r#"\.to_owned\s*\("#
    ),
    p!(
        "R135",
        "extend(vec![..])",
        High,
        Performance,
        r#"\.extend\s*\(\s*vec!\s*\["#
    ),
    p!(
        "R136",
        r#"push_str("x")"#,
        High,
        Performance,
        r#"\.push_str\s*\(\s*".{1}"\s*\)"#
    ),
    p!(
        "R137",
        r#"split(" ")"#,
        High,
        Performance,
        r#"\.split\s*\(\s*" "\s*\)"#
    ),
    p!(
        "R138",
        "trim().to_string()",
        High,
        Performance,
        r#"\.trim\s*\(\s*\)\s*\.to_string\s*\("#
    ),
    p!(
        "R139",
        r#"replace("a","b")"#,
        High,
        Performance,
        r#"\.replace\s*\(\s*".{1}"\s*,\s*".{1}"\s*\)"#
    ),
    p!(
        "R140",
        "from_utf8_lossy",
        High,
        Performance,
        r#"\bfrom_utf8_lossy\s*\("#
    ),
    p!(
        "R141",
        "map(|x| x.clone())",
        High,
        Performance,
        r#"\.map\s*\(\s*\|\s*[^|]*\s*\|\s*[A-Za-z_][A-Za-z0-9_]*\.clone\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R142",
        "map(|x| x.to_string())",
        High,
        Performance,
        r#"\.map\s*\(\s*\|\s*[^|]*\s*\|\s*[A-Za-z_][A-Za-z0-9_]*\.to_string\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R143",
        "and_then(Some)",
        High,
        Performance,
        r#"\.and_then\s*\(\s*\|\s*[^|]*\s*\|\s*Some\s*\("#
    ),
    p!(
        "R144",
        "filter(|_| true)",
        High,
        Performance,
        r#"\.filter\s*\(\s*\|\s*[^|]*\s*\|\s*true\s*\)"#
    ),
    p!(
        "R145",
        "filter(|_| false)",
        High,
        Performance,
        r#"\.filter\s*\(\s*\|\s*[^|]*\s*\|\s*false\s*\)"#
    ),
    p!(
        "R146",
        "take(usize::MAX)",
        High,
        Performance,
        r#"\.take\s*\(\s*usize::MAX\s*\)"#
    ),
    p!(
        "R147",
        "skip(0)",
        High,
        Performance,
        r#"\.skip\s*\(\s*0\s*\)"#
    ),
    p!(
        "R148",
        "nth(0)",
        High,
        Performance,
        r#"\.nth\s*\(\s*0\s*\)"#
    ),
    p!(
        "R149",
        "rev().rev()",
        High,
        Performance,
        r#"\.rev\s*\(\s*\)\s*\.rev\s*\(\s*\)"#
    ),
    p!(
        "R150",
        "drain(..).collect::<Vec<_>>()",
        High,
        Performance,
        r#"\.drain\s*\(\s*\.\.\s*\)\s*\.collect::<\s*Vec<[^>]*>\s*>\s*\("#
    ),
    p!(
        "R151",
        "iter().count()",
        High,
        Performance,
        r#"\.iter\s*\(\s*\)\s*\.count\s*\(\s*\)"#
    ),
    p!(
        "R152",
        "String::from(x.to_string())",
        High,
        Performance,
        r#"\bString::from\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\.to_string\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R153",
        "unwrap_or_else(String::new())",
        High,
        Performance,
        r#"\.unwrap_or_else\s*\(\s*\|\|\s*String::new\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R154",
        r#"unwrap_or("".to_string())"#,
        High,
        Performance,
        r#"\.unwrap_or\s*\(\s*""\.to_string\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R155",
        "chars().nth(..)",
        High,
        Performance,
        r#"\.chars\s*\(\s*\)\s*\.nth\s*\("#
    ),
    p!(
        "R156",
        "chars().count()",
        High,
        Performance,
        r#"\.chars\s*\(\s*\)\s*\.count\s*\(\s*\)"#
    ),
    p!(
        "R157",
        "contains(&String::from(..))",
        High,
        Performance,
        r#"\.contains\s*\(\s*&\s*String::from\s*\("#
    ),
    p!(
        "R158",
        "format!(..to_string..)",
        High,
        Performance,
        r#"\bformat!\s*\([^)]*to_string\s*\(\s*\)[^)]*\)"#
    ),
    p!(
        "R159",
        "Vec::with_capacity(0)",
        High,
        Performance,
        r#"\bVec::with_capacity\s*\(\s*0\s*\)"#
    ),
    p!(
        "R160",
        "String::with_capacity(0)",
        High,
        Performance,
        r#"\bString::with_capacity\s*\(\s*0\s*\)"#
    ),
    p!(
        "R161",
        "reserve(0)",
        High,
        Performance,
        r#"\.reserve\s*\(\s*0\s*\)"#
    ),
    p!(
        "R162",
        "HashMap::with_capacity(0)",
        High,
        Performance,
        r#"\bHashMap::with_capacity\s*\(\s*0\s*\)"#
    ),
    p!(
        "R163",
        "shrink_to_fit()",
        High,
        Performance,
        r#"\.shrink_to_fit\s*\(\s*\)"#
    ),
    p!(
        "R164",
        "clone().into_iter()",
        High,
        Performance,
        r#"\.clone\s*\(\s*\)\s*\.into_iter\s*\(\s*\)"#
    ),
    p!(
        "R165",
        "to_vec()",
        High,
        Performance,
        r#"\.to_vec\s*\(\s*\)"#
    ),
    p!(
        "R166",
        "Box<Vec<_>>",
        High,
        Performance,
        r#"\bBox\s*<\s*Vec<"#
    ),
    p!(
        "R167",
        "Rc<Mutex<_>>",
        High,
        Performance,
        r#"\bRc\s*<\s*Mutex<"#
    ),
    p!(
        "R168",
        "Arc<Mutex<_>>",
        High,
        Performance,
        r#"\bArc\s*<\s*Mutex<"#
    ),
    p!(
        "R170",
        "retain(|_| true)",
        High,
        Performance,
        r#"\.retain\s*\(\s*\|\s*_\s*\|\s*true\s*\)"#
    ),
    p!(
        "R171",
        "from_utf16_lossy",
        High,
        Performance,
        r#"\bfrom_utf16_lossy\s*\("#
    ),
    p!(
        "R172",
        "String::from_iter(..)",
        High,
        Performance,
        r#"\bString::from_iter\s*\("#
    ),
    p!(
        "R173",
        r#"collect::<Vec<_>>().join("")"#,
        High,
        Performance,
        r#"\.collect::<\s*Vec<[^>]*>\s*>\s*\(\s*\)\s*\.join\s*\(\s*""\s*\)"#
    ),
    p!(
        "R174",
        "unwrap_or_else(Vec::new())",
        High,
        Performance,
        r#"\.unwrap_or_else\s*\(\s*\|\|\s*Vec::new\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R175",
        "keys().collect::<Vec<_>>()",
        High,
        Performance,
        r#"\.keys\s*\(\s*\)\s*\.collect::<\s*Vec<"#
    ),
    p!(
        "R176",
        "values().collect::<Vec<_>>()",
        High,
        Performance,
        r#"\.values\s*\(\s*\)\s*\.collect::<\s*Vec<"#
    ),
    p!(
        "R177",
        "unwrap_or(vec![])",
        High,
        Performance,
        r#"\.unwrap_or\s*\(\s*vec!\s*\[\s*\]\s*\)"#
    ),
    p!(
        "R178",
        "iter().map(|x| &*x)",
        High,
        Performance,
        r#"\.iter\s*\(\s*\)\s*\.map\s*\(\s*\|\s*[^|]*\s*\|\s*&\s*\*"#
    ),
    p!(
        "R179",
        "chars().last()",
        High,
        Performance,
        r#"\.chars\s*\(\s*\)\s*\.last\s*\(\s*\)"#
    ),
    p!(
        "R180",
        "insert(0, ... char)",
        High,
        Performance,
        r#"\.insert\s*\(\s*0\s*,\s*'[^']'\s*\)"#
    ),
    p!(
        "R181",
        "remove(0)",
        High,
        Performance,
        r#"\.remove\s*\(\s*0\s*\)"#
    ),
    p!(
        "R182",
        "insert(0, ...)",
        High,
        Performance,
        r#"\.insert\s*\(\s*0\s*,\s*"#
    ),
    p!(
        "R183",
        "unwrap_or_else(Default::default())",
        High,
        Performance,
        r#"\.unwrap_or_else\s*\(\s*\|\|\s*Default::default\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R184",
        "to_string().as_str()",
        High,
        Performance,
        r#"\.to_string\s*\(\s*\)\s*\.as_str\s*\(\s*\)"#
    ),
    p!(
        "R185",
        "Vec collect → into_iter → Vec",
        High,
        Performance,
        r#"\.collect::<\s*Vec<[^>]*>\s*>\s*\(\s*\)\s*\.into_iter\s*\(\s*\)\s*\.collect::<\s*Vec<"#
    ),
    p!(
        "R186",
        "repeat(0|1)",
        High,
        Performance,
        r#"\.repeat\s*\(\s*[01]\s*\)"#
    ),
    p!(
        "R187",
        "clone_from(..)",
        High,
        Performance,
        r#"\.clone_from\s*\("#
    ),
    p!(
        "R188",
        "drain(..).for_each(drop)",
        High,
        Performance,
        r#"\.drain\s*\(\s*\.\.\s*\)\s*\.for_each\s*\(\s*drop\s*\)"#
    ),
    p!(
        "R189",
        "sort_by(partial_cmp().unwrap())",
        High,
        Performance,
        r#"\.sort_by\s*\(\s*\|\s*[^|]*\s*\|\s*[^\.]*\.partial_cmp\s*\(\s*[^)]+\s*\)\s*\.unwrap\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R190",
        "sort_by(|a,b| a.cmp(b))",
        High,
        Performance,
        r#"\.sort_by\s*\(\s*\|\s*[A-Za-z_]+,\s*[A-Za-z_]+\s*\|\s*[A-Za-z_]+\s*\.cmp\s*\(\s*[A-Za-z_]+\s*\)\s*\)"#
    ),
    p!(
        "R191",
        "sort_by_key(|x| x.clone())",
        High,
        Performance,
        r#"\.sort_by_key\s*\(\s*\|\s*[^|]*\s*\|\s*[A-Za-z_][A-Za-z0-9_]*\.clone\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R192",
        "collect::<Vec<_>>().len()",
        High,
        Performance,
        r#"\.collect::<\s*Vec<[^>]*>\s*>\s*\(\s*\)\s*\.len\s*\(\s*\)"#
    ),
    p!(
        "R193",
        "collect::<Vec<_>>().is_empty()",
        High,
        Performance,
        r#"\.collect::<\s*Vec<[^>]*>\s*>\s*\(\s*\)\s*\.is_empty\s*\(\s*\)"#
    ),
    p!("R194", "dedup()", High, Performance, r#"\.dedup\s*\(\s*\)"#),
    p!(
        "R195",
        "push_str(&String::from(..))",
        High,
        Performance,
        r#"\.push_str\s*\(\s*&\s*String::from\s*\("#
    ),
    p!(
        "R196",
        "String::from(s.as_str())",
        High,
        Performance,
        r#"\bString::from\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\.as_str\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R197",
        "iter().any(|_| true)",
        High,
        Performance,
        r#"\.iter\s*\(\s*\)\s*\.any\s*\(\s*\|\s*_\s*\|\s*true\s*\)"#
    ),
    p!(
        "R198",
        "iter().all(|_| false)",
        High,
        Performance,
        r#"\.iter\s*\(\s*\)\s*\.all\s*\(\s*\|\s*_\s*\|\s*false\s*\)"#
    ),
    p!(
        "R199",
        "position(|_| true)",
        High,
        Performance,
        r#"\.position\s*\(\s*\|\s*_\s*\|\s*true\s*\)"#
    ),
    p!(
        "R200",
        r#"format!("{}", x)"#,
        High,
        Performance,
        r#"\bformat!\s*\(\s*"\{\}"\s*,\s*[^,]+\s*\)"#
    ),
    p!(
        "R201",
        "HashSet::new()",
        High,
        Performance,
        r#"\bHashSet::new\s*\("#
    ),
    p!(
        "R202",
        "BTreeSet::new()",
        High,
        Performance,
        r#"\bBTreeSet::new\s*\("#
    ),
    p!(
        "R203",
        "mem::replace(.., Default::default())",
        High,
        Performance,
        r#"\bstd::mem::replace\s*\(\s*&mut\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*Default::default\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R204",
        "mem::replace(.., x.clone())",
        High,
        Performance,
        r#"\bstd::mem::replace\s*\(\s*&mut\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*[A-Za-z_][A-Za-z0-9_]*\.clone\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R205",
        r#"format!("{:?}", x)"#,
        High,
        Performance,
        r#"\bformat!\s*\(\s*"\{\:\?\}"\s*,\s*[^,]+\s*\)"#
    ),
    p!(
        "R206",
        "PathBuf::from(String::new())",
        High,
        Performance,
        r#"\bPathBuf::from\s*\(\s*String::new\s*\(\s*\)\s*\)"#
    ),
    p!(
        "R207",
        "Vec collect then iter()",
        High,
        Performance,
        r#"\.collect::<\s*Vec<[^>]*>\s*>\s*\(\s*\)\s*\.iter\s*\(\s*\)"#
    ),
    p!(
        "R208",
        "push_str(format!(..))",
        High,
        Performance,
        r#"\.push_str\s*\(\s*format!\s*\("#
    ),
    p!(
        "R209",
        "push(format!(..))",
        High,
        Performance,
        r#"\.push\s*\(\s*format!\s*\("#
    ),
    p!(
        "R210",
        "chars().collect::<Vec<char>>()",
        High,
        Performance,
        r#"\.chars\s*\(\s*\)\s*\.collect::<\s*Vec\s*<\s*char\s*>\s*>\s*\("#
    ),
    p!(
        "R211",
        "to_string().chars()",
        High,
        Performance,
        r#"\.to_string\s*\(\s*\)\s*\.chars\s*\(\s*\)"#
    ),
    p!(
        "R212",
        "saturating_add(0)",
        High,
        Performance,
        r#"\.saturating_add\s*\(\s*0\s*\)"#
    ),
    p!(
        "R213",
        "saturating_sub(0)",
        High,
        Performance,
        r#"\.saturating_sub\s*\(\s*0\s*\)"#
    ),
    p!(
        "R214",
        "wrapping_add(0)",
        High,
        Performance,
        r#"\.wrapping_add\s*\(\s*0\s*\)"#
    ),
    p!(
        "R215",
        "wrapping_sub(0)",
        High,
        Performance,
        r#"\.wrapping_sub\s*\(\s*0\s*\)"#
    ),
    p!(
        "R216",
        "checked_add(0)",
        High,
        Performance,
        r#"\.checked_add\s*\(\s*0\s*\)"#
    ),
    p!(
        "R217",
        "map identity",
        High,
        Performance,
        r#"\.map\s*\(\s*\|\s*x\s*\|\s*x\s*\)"#
    ),
    p!(
        "R218",
        "clone().clone()",
        High,
        Performance,
        r#"\.clone\s*\(\s*\)\s*\.clone\s*\(\s*\)"#
    ),
    p!(
        "R219",
        r#"split("")"#,
        High,
        Performance,
        r#"\.split\s*\(\s*""\s*\)"#
    ),
    p!(
        "R220",
        "collect Vec<char> (alt)",
        High,
        Performance,
        r#"\.collect::<\s*Vec\s*<\s*char\s*>\s*>\s*\("#
    ),
];

pub fn all_patterns() -> &'static [PatternDef] {
    PATTERNS
}

static REGEX_CACHE: Lazy<HashMap<&'static str, Regex>> = Lazy::new(|| {
    let mut m = HashMap::new();
    for p in all_patterns() {
        m.insert(p.id, Regex::new(p.expr).expect(p.id));
    }
    m
});

static REQUIRE_CACHE: Lazy<HashMap<&'static str, Vec<Regex>>> = Lazy::new(|| {
    let mut m = HashMap::new();
    for p in all_patterns() {
        if !p.requires_all.is_empty() {
            let v = p
                .requires_all
                .iter()
                .map(|r| Regex::new(r).expect(p.id))
                .collect::<Vec<_>>();
            m.insert(p.id, v);
        }
    }
    m
});

/// Scan a Rust source string and return all issues found.
/// - Simple, line-based matching for speed and predictability.
/// - A pattern may also require additional regexes to be present anywhere in the file (`requires_all`).
pub fn scan(source: &str) -> Vec<Issue> {
    let mut issues = Vec::new();
    let requires_ok: HashMap<&'static str, bool> = all_patterns()
        .iter()
        .map(|p| {
            let ok = if let Some(reqs) = REQUIRE_CACHE.get(p.id) {
                reqs.iter().all(|rr| rr.is_match(source))
            } else {
                true
            };
            (p.id, ok)
        })
        .collect();

    for (lineno, line) in source.lines().enumerate() {
        // We intentionally also scan comments/strings; this is a static smell finder.
        for p in all_patterns() {
            if !requires_ok[p.id] {
                continue;
            }
            let re = &REGEX_CACHE[p.id];
            for m in re.find_iter(line) {
                let col = m.start() + 1;
                issues.push(Issue {
                    pattern_id: p.id,
                    name: p.name,
                    severity: p.severity.as_str(),
                    category: p.category.as_str(),
                    line: lineno + 1,
                    col,
                    excerpt: trim_excerpt(line),
                    severity_confirmed: false,
                    confidence_score: 0.7, // Default confidence
                });
            }
        }
    }
    issues
}

/// Confirm severity of issues using additional heuristics
pub fn confirm_severity(mut issues: Vec<Issue>, file_content: &str) -> Vec<Issue> {
    for issue in &mut issues {
        let (confirmed, confidence) = validate_severity(issue, file_content);
        issue.severity_confirmed = confirmed;
        issue.confidence_score = confidence;
    }
    issues
}

/// Validate severity using context analysis
fn validate_severity(issue: &Issue, file_content: &str) -> (bool, f64) {
    let lines: Vec<&str> = file_content.lines().collect();
    let issue_line_idx = issue.line.saturating_sub(1);

    // Get context around the issue (3 lines before and after)
    let start = issue_line_idx.saturating_sub(3);
    let end = (issue_line_idx + 4).min(lines.len());
    let context = &lines[start..end];

    match issue.severity {
        "Critical" => validate_critical_severity(issue, context),
        "High" => validate_high_severity(issue, context),
        _ => (true, 0.8), // Default confidence for other severities
    }
}

/// Validate Critical severity patterns
fn validate_critical_severity(issue: &Issue, context: &[&str]) -> (bool, f64) {
    let context_text = context.join("\n").to_lowercase();

    match issue.pattern_id {
        // Memory safety issues - always critical
        "R001" | "R002" | "R003" if context_text.contains("unsafe") => (true, 0.95),

        // Panic patterns - check if they're in test code or have proper error handling
        "R007" | "R008" => {
            if context_text.contains("#[test]")
                || context_text.contains("test_")
                || context_text.contains("#[cfg(test)]")
            {
                (false, 0.3) // Lower confidence in test code
            } else if context_text.contains("todo!") || context_text.contains("unimplemented!") {
                (true, 0.9) // High confidence for development panics
            } else if context_text.contains("expect(") || context_text.contains("unwrap_or") {
                (false, 0.5) // Medium confidence if there's error handling nearby
            } else {
                (true, 0.85) // High confidence for unexpected panics
            }
        }

        // Default for other critical patterns
        _ => (true, 0.8),
    }
}

/// Validate High severity patterns
fn validate_high_severity(_issue: &Issue, context: &[&str]) -> (bool, f64) {
    let context_text = context.join("\n").to_lowercase();

    // Check if it's in test code
    if context_text.contains("#[test]")
        || context_text.contains("test_")
        || context_text.contains("#[cfg(test)]")
    {
        return (false, 0.4); // Lower confidence in test code
    }

    // Check for proper error handling context
    if context_text.contains("result")
        || context_text.contains("error")
        || context_text.contains("match")
    {
        (true, 0.75)
    } else {
        (true, 0.8)
    }
}

fn trim_excerpt(line: &str) -> String {
    const MAX: usize = 180;
    if line.len() <= MAX {
        line.to_string()
    } else {
        let mut s = line[..MAX].to_string();
        s.push('…');
        s
    }
}
