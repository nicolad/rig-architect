// Rig Analyzer Library
//
// Enhanced with 220+ pattern-based Rust bug detection and AI validation

pub mod deepseek;
pub mod false_positive_filter;
pub mod mcp;
pub mod patterns;
pub mod vector_store;

pub use deepseek::DeepSeekClient;
pub use vector_store::{classify_local_first, Decision, PatternSpec, VectorStoreManager};

// Re-export bug detection API
pub use false_positive_filter::{
    FalsePositiveFilter, SerializableIssue, ValidatedIssue, ValidationResult,
};
pub use patterns::{all_patterns, confirm_severity, scan, Category, Issue, PatternDef, Severity};

// Configuration constants (previously in config.rs)
use std::path::Path;

/// Application configuration constants
pub struct Config;

#[allow(dead_code)]
impl Config {
    /// Path to the environment file (relative to crate dir)
    pub const ENV_FILE_PATH: &'static str = ".env";

    /// Environment variable name for rig repository path
    pub const RIG_REPO_PATH_ENV: &'static str = "REPO_PATH";

    /// Default MCP server port
    pub const DEFAULT_MCP_PORT: u16 = 3000;

    /// Default bugs directory (relative name; use bugs_directory_path() for absolute)
    pub const BUGS_DIRECTORY: &'static str = "bugs";

    /// Default logs directory
    pub const LOGS_DIRECTORY: &'static str = "logs";

    /// Nautilus Trader adapters directory (Python files)
    pub const ADAPTERS_DIRECTORY: &'static str = "../nautilus_trader/adapters";

    /// Core Rust adapters directory
    /// Note: This path is relative to the nautilus-trader-rig crate directory.
    /// The adapters live at the repo root under `crates/adapters`, so from
    /// within `nautilus-trader-rig/` we must go up one level.
    pub const CORE_ADAPTERS_DIRECTORY: &'static str = "../crates/adapters";

    /// Rust adapter file extensions (without leading dot)
    pub const RUST_FILE_EXTENSIONS: &'static [&'static str] = &["rs"];

    /// Vector similarity search limit
    pub const DEFAULT_SEARCH_LIMIT: usize = 10;

    /// DeepSeek model name (using reasoner by default)
    pub const DEEPSEEK_MODEL: &'static str = "deepseek-reasoner";

    /// FastEmbed model dimension
    pub const FASTEMBED_DIMENSION: usize = 384;
}

#[allow(dead_code)]
impl Config {
    /// Absolute path to this crate's directory at compile time
    pub fn manifest_dir() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR"))
    }

    /// Get the full path to the environment file
    pub fn env_file_path() -> &'static Path {
        Path::new(Self::ENV_FILE_PATH)
    }

    /// Get the full path to the bugs directory
    pub fn bugs_directory() -> &'static Path {
        Path::new(Self::BUGS_DIRECTORY)
    }

    /// Get the full path to the logs directory
    pub fn logs_directory() -> &'static Path {
        Path::new(Self::LOGS_DIRECTORY)
    }

    /// Absolute bugs directory path, independent of current working directory
    pub fn bugs_directory_path() -> std::path::PathBuf {
        Self::manifest_dir().join("bugs")
    }

    /// Check if environment file exists
    pub fn env_file_exists() -> bool {
        Self::env_file_path().exists()
    }

    /// Check if bugs directory exists
    pub fn bugs_directory_exists() -> bool {
        Self::bugs_directory().exists()
    }

    /// Check if logs directory exists
    pub fn logs_directory_exists() -> bool {
        Self::logs_directory().exists()
    }

    /// Generate a log file path with timestamp
    pub fn generate_log_file_path() -> std::path::PathBuf {
        let timestamp = chrono::Utc::now().format("%Y-%m-%d_%H-%M-%S").to_string();
        Self::logs_directory().join(format!("nautilus_trader_rig_{}.log", timestamp))
    }

    /// Get the rig repository path from environment variable, or default
    pub fn rig_repo_path() -> std::path::PathBuf {
        if let Ok(path) = std::env::var(Self::RIG_REPO_PATH_ENV) {
            std::path::PathBuf::from(path)
        } else {
            // Default fallback: assume rig repo is adjacent to current repo
            Self::manifest_dir()
                .parent()
                .unwrap_or(Self::manifest_dir())
                .join("rig")
        }
    }

    /// Get the full path to the Rust adapters directory
    pub fn rust_adapters_directory() -> &'static Path {
        Path::new(Self::CORE_ADAPTERS_DIRECTORY)
    }

    /// Get the full path to the Rust core directory
    pub fn rust_core_directory() -> &'static Path {
        Path::new(Self::CORE_ADAPTERS_DIRECTORY)
    }

    /// Check if Rust adapters directory exists
    pub fn rust_adapters_directory_exists() -> bool {
        Self::rust_adapters_directory().exists()
    }

    /// Check if Rust core directory exists
    pub fn rust_core_directory_exists() -> bool {
        Self::rust_core_directory().exists()
    }

    /// Get all Rust adapter directories
    pub fn all_rust_adapter_directories() -> Vec<&'static Path> {
        vec![Self::rust_adapters_directory(), Self::rust_core_directory()]
    }

    /// Absolute paths for Rust adapter directories (preferred for robust execution)
    pub fn all_rust_adapter_directories_abs() -> Vec<std::path::PathBuf> {
        if std::env::var(Self::RIG_REPO_PATH_ENV).is_ok() {
            // Use rig repo from environment variable
            let rig_path = Self::rig_repo_path();
            vec![
                rig_path.join("rig-core/src"),
                rig_path.join("rig-bedrock/src"),
                rig_path.join("rig-eternalai/src"),
                rig_path.join("rig-fastembed/src"),
                rig_path.join("rig-lancedb/src"),
                rig_path.join("rig-milvus/src"),
                rig_path.join("rig-mongodb/src"),
                rig_path.join("rig-neo4j/src"),
                rig_path.join("rig-postgres/src"),
                rig_path.join("rig-qdrant/src"),
                rig_path.join("rig-s3vectors/src"),
                rig_path.join("rig-scylladb/src"),
                rig_path.join("rig-sqlite/src"),
                rig_path.join("rig-surrealdb/src"),
                rig_path.join("rig-wasm/src"),
            ]
        } else {
            // Fallback to original hardcoded paths
            let base = Self::manifest_dir();
            vec![
                base.join("../crates/adapters"),
                base.join("../crates/adapters"), // kept twice to mirror existing API semantics
            ]
        }
    }

    /// Absolute path to core adapters directory
    pub fn core_adapters_directory_abs() -> std::path::PathBuf {
        if std::env::var(Self::RIG_REPO_PATH_ENV).is_ok() {
            // Use rig repo from environment variable
            Self::rig_repo_path().join("rig-core/src")
        } else {
            // Fallback to original hardcoded path
            Self::manifest_dir().join("../crates/adapters")
        }
    }

    /// Check if rig repository path environment variable is set
    pub fn rig_repo_path_env_set() -> bool {
        std::env::var(Self::RIG_REPO_PATH_ENV).is_ok()
    }

    /// Get the current value of the rig repository path environment variable (if set)
    pub fn rig_repo_path_env_value() -> Option<String> {
        std::env::var(Self::RIG_REPO_PATH_ENV).ok()
    }

    /// Get list of supported Rust file extensions
    pub fn rust_extensions() -> &'static [&'static str] {
        Self::RUST_FILE_EXTENSIONS
    }

    /// Get Rust adapter directory path by name
    pub fn rust_adapter_path(adapter_name: &str) -> std::path::PathBuf {
        Path::new(Self::CORE_ADAPTERS_DIRECTORY).join(adapter_name)
    }
}

// ===== Logging Module (previously in logging.rs) =====

/// Centralized logging configuration for Rig Analyzer
///
/// This module provides consistent logging setup across all components with:
/// - Structured logging with tracing
/// - Environment-based log levels
/// - Color-coded output for development
/// - File output for production
/// - Performance monitoring
/// - Component-specific logging levels
use anyhow::Result;
use tracing::{debug, info, Level};
use tracing_subscriber::EnvFilter;

/// Log levels for different components
#[derive(Debug, Clone)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl From<LogLevel> for Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    }
}

/// Logging configuration for the application
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Base log level for the application
    pub level: LogLevel,
    /// Enable file logging
    pub file_logging: bool,
    /// Log file path (if file logging enabled)
    pub log_file: Option<String>,
    /// Enable colored output (for development)
    pub colored: bool,
    /// Include timestamps
    #[allow(dead_code)]
    pub timestamps: bool,
    /// Include source location (file:line)
    pub include_location: bool,
    /// Component-specific log levels
    pub component_levels: Vec<(String, LogLevel)>,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            file_logging: false,
            log_file: None,
            colored: true,
            timestamps: true,
            include_location: true,
            component_levels: vec![
                ("nautilus_trader_rig".to_string(), LogLevel::Info),
                ("ort".to_string(), LogLevel::Warn), // Reduce ONNX runtime noise
                ("hf_hub".to_string(), LogLevel::Info),
                ("rig".to_string(), LogLevel::Info),
                ("mcp".to_string(), LogLevel::Debug), // Detailed MCP logging
                ("config".to_string(), LogLevel::Debug),
                ("vector_store".to_string(), LogLevel::Info),
                ("deepseek".to_string(), LogLevel::Info),
            ],
        }
    }
}

impl LogConfig {
    /// Create a new log configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base log level
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.level = level;
        self
    }

    /// Enable file logging with specified path
    pub fn with_file_logging<P: AsRef<Path>>(mut self, log_file: P) -> Self {
        self.file_logging = true;
        self.log_file = Some(log_file.as_ref().to_string_lossy().to_string());
        self
    }

    /// Set colored output (useful for development vs production)
    pub fn with_colored(mut self, colored: bool) -> Self {
        self.colored = colored;
        self
    }

    /// Include source code location in logs
    pub fn with_location(mut self, include_location: bool) -> Self {
        self.include_location = include_location;
        self
    }

    /// Add component-specific log level
    pub fn with_component_level(mut self, component: &str, level: LogLevel) -> Self {
        self.component_levels.push((component.to_string(), level));
        self
    }

    /// Initialize the global tracing subscriber with dual output (console + file)
    pub fn init(self) -> Result<()> {
        use tracing_subscriber::layer::SubscriberExt;

        // Build the environment filter
        let mut filter = EnvFilter::from_default_env().add_directive(
            format!(
                "{}={}",
                env!("CARGO_PKG_NAME").replace('-', "_"),
                self.level_string()
            )
            .parse()?,
        );

        // Add component-specific filters
        for (component, level) in &self.component_levels {
            filter =
                filter.add_directive(format!("{}={}", component, level_string(level)).parse()?);
        }

        // Create console layer (always enabled for development)
        let console_layer = tracing_subscriber::fmt::layer()
            .with_ansi(self.colored)
            .with_target(true)
            .with_file(self.include_location)
            .with_line_number(self.include_location)
            .compact();

        // Create the subscriber with console layer
        let subscriber = tracing_subscriber::registry()
            .with(filter)
            .with(console_layer);

        // Add file layer if file logging is enabled
        if self.file_logging {
            if let Some(log_file) = &self.log_file {
                match std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(log_file)
                {
                    Ok(file) => {
                        let file_layer = tracing_subscriber::fmt::layer()
                            .with_ansi(false) // No colors in file
                            .with_target(true)
                            .with_file(true)
                            .with_line_number(true)
                            .compact()
                            .with_writer(file);

                        let subscriber = subscriber.with(file_layer);
                        tracing::subscriber::set_global_default(subscriber)?;

                        info!("🚀 Logging system initialized with console and file output");
                        debug!("Log file: {}", log_file);
                    }
                    Err(e) => {
                        tracing::subscriber::set_global_default(subscriber)?;
                        info!("🚀 Logging system initialized with console output only");
                        eprintln!("❌ Failed to open log file '{}': {}", log_file, e);
                        eprintln!("Continuing with console logging only");
                    }
                }
            } else {
                tracing::subscriber::set_global_default(subscriber)?;
                info!("🚀 Logging system initialized with console output only");
            }
        } else {
            tracing::subscriber::set_global_default(subscriber)?;
            info!("🚀 Logging system initialized with console output only");
        }

        debug!("Log configuration: {:?}", self);

        Ok(())
    }

    fn level_string(&self) -> &'static str {
        level_string(&self.level)
    }
}

fn level_string(level: &LogLevel) -> &'static str {
    match level {
        LogLevel::Trace => "trace",
        LogLevel::Debug => "debug",
        LogLevel::Info => "info",
        LogLevel::Warn => "warn",
        LogLevel::Error => "error",
    }
}

/// Initialize logging with default configuration
#[allow(dead_code)]
pub fn init_default_logging() -> Result<()> {
    LogConfig::default().init()
}

/// Initialize logging for development (verbose, colored, with file output)
pub fn init_dev_logging() -> Result<()> {
    // Create logs directory if it doesn't exist
    if !Config::logs_directory_exists() {
        std::fs::create_dir_all(Config::logs_directory())?;
        println!("📁 Created logs directory: {:?}", Config::logs_directory());
    }

    // Generate timestamped log file path
    let log_file = Config::generate_log_file_path();
    println!("📝 Log file will be created at: {:?}", log_file);

    // Test file creation to ensure path is writable
    if let Some(parent) = log_file.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
            println!("📁 Created parent directories for log file");
        }
    }

    LogConfig::new()
        .with_level(LogLevel::Debug)
        .with_colored(true)
        .with_location(true)
        .with_file_logging(log_file)
        .with_component_level("nautilus_trader_rig", LogLevel::Debug)
        .with_component_level("mcp", LogLevel::Debug)
        .with_component_level("config", LogLevel::Debug)
        .init()
}

/// Initialize logging for production (concise, file-based)
#[allow(dead_code)]
pub fn init_prod_logging<P: AsRef<Path>>(log_file: P) -> Result<()> {
    LogConfig::new()
        .with_level(LogLevel::Info)
        .with_colored(false)
        .with_location(false)
        .with_file_logging(log_file)
        .with_component_level("ort", LogLevel::Error) // Minimize ONNX noise in prod
        .init()
}

// Structured logging macros for consistent formatting
/// Log file processing operations
#[macro_export]
macro_rules! log_file_processing {
    ($level:ident, $action:expr, $file:expr) => {
        tracing::$level!("📄 {} file: {}", $action, $file);
    };
    ($level:ident, $action:expr, $file:expr, $size:expr) => {
        tracing::$level!("📄 {} file: {} ({} bytes)", $action, $file, $size);
    };
}

/// Log directory operations  
#[macro_export]
macro_rules! log_directory_op {
    ($level:ident, $action:expr, $dir:expr) => {
        tracing::$level!("📁 {} directory: {:?}", $action, $dir);
    };
    ($level:ident, $action:expr, $dir:expr, $count:expr) => {
        tracing::$level!("📁 {} directory: {:?} ({} items)", $action, $dir, $count);
    };
}

/// Log configuration operations
#[allow(unused_macros)]
macro_rules! log_config_op {
    ($level:ident, $action:expr, $component:expr) => {
        tracing::$level!("🔧 {} {}", $action, $component);
    };
    ($level:ident, $action:expr, $component:expr, $value:expr) => {
        tracing::$level!("🔧 {} {}: {}", $action, $component, $value);
    };
}

/// Log network/MCP operations
#[macro_export]
macro_rules! log_mcp_op {
    ($level:ident, $action:expr, $details:expr) => {
        tracing::$level!("🌐 MCP {}: {}", $action, $details);
    };
    ($level:ident, $action:expr) => {
        tracing::$level!("🌐 MCP {}", $action);
    };
}

/// Log performance metrics
#[allow(unused_macros)]
macro_rules! log_performance {
    ($level:ident, $operation:expr, $duration:expr) => {
        tracing::$level!("⏱️ {} took: {:?}", $operation, $duration);
    };
    ($level:ident, $operation:expr, $duration:expr, $count:expr) => {
        tracing::$level!("⏱️ {} took: {:?} ({} items)", $operation, $duration, $count);
    };
}
