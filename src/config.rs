use std::env;

/// Configuration settings for the AI Architecture Improver
#[derive(Debug, Clone)]
pub struct Config {
    pub max_changed_lines: usize,
    pub branch_prefix: String,
    pub min_audit_avg: f32,
    pub strict_safety: bool,
    pub include: Option<String>,
    pub exclude: Option<String>,
    pub cron_schedule: String,
}

impl Config {
    /// Load configuration from environment variables
    /// Returns a Config instance with values parsed from environment variables or defaults
    pub fn from_env() -> Self {
        Self {
            max_changed_lines: env::var("ARCH_MAX_LINES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(32),
            branch_prefix: env::var("ARCH_BRANCH_PREFIX")
                .unwrap_or_else(|_| "auto/arch-improvement".to_string()),
            min_audit_avg: env::var("ARCH_MIN_SCORE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.88),
            strict_safety: env::var("ARCH_STRICT_SAFETY")
                .map(|s| s == "true")
                .unwrap_or(true),
            include: env::var("ARCH_INCLUDE").ok(),
            exclude: env::var("ARCH_EXCLUDE").ok(),
            cron_schedule: env::var("ARCH_CRON_SCHEDULE")
                .unwrap_or_else(|_| "0 */1 * * * *".to_string()), // Default: every 1 minute
        }
    }

    /// Get default configuration with sensible defaults
    pub fn default() -> Self {
        Self {
            max_changed_lines: 32,
            branch_prefix: "auto/arch-improvement".to_string(),
            min_audit_avg: 0.88,
            strict_safety: true,
            include: None,
            exclude: None,
            cron_schedule: "0 */5 * * * *".to_string(), // Every 5 minutes
        }
    }

    /// Validate the cron schedule format
    pub fn validate_cron_schedule(&self) -> Result<(), String> {
        use std::str::FromStr;
        use apalis_cron::Schedule;
        
        Schedule::from_str(&self.cron_schedule)
            .map_err(|e| format!("Invalid cron schedule '{}': {}", self.cron_schedule, e))?;
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.max_changed_lines, 32);
        assert_eq!(config.branch_prefix, "auto/arch-improvement");
        assert_eq!(config.cron_schedule, "0 */5 * * * *");
        assert!(config.validate_cron_schedule().is_ok());
    }

    #[test]
    fn test_cron_schedule_validation() {
        let mut config = Config::default();
        
        // Valid schedule
        config.cron_schedule = "0 0 12 * * *".to_string(); // Daily at noon
        assert!(config.validate_cron_schedule().is_ok());
        
        // Invalid schedule
        config.cron_schedule = "invalid cron".to_string();
        assert!(config.validate_cron_schedule().is_err());
    }
}
