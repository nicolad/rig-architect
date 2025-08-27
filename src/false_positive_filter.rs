// False Positive Detection System
//
// This module uses DeepSeek AI to validate pattern matches and filter out false positives
// by analyzing the context and semantics of detected issues.

use crate::deepseek::DeepSeekClient;
use crate::patterns::Issue;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, warn};

// Rig-based imports
use rig::client::{CompletionClient, ProviderClient};
use rig::extractor::Extractor;
use rig::providers::deepseek;
use schemars::JsonSchema;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_false_positive: bool,
    pub confidence: f32, // 0.0 to 1.0
    pub reasoning: String,
    pub suggested_action: SuggestedAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestedAction {
    Ignore,
    Review,
    FixImmediately,
    MonitorForPatterns,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsePositivePattern {
    pub pattern_id: String,
    pub context_keywords: Vec<String>,
    pub common_false_positives: Vec<String>,
    pub confidence_threshold: f32,
}

pub struct FalsePositiveFilter {
    deepseek_client: Option<DeepSeekClient>,
    known_patterns: HashMap<String, FalsePositivePattern>,
    validation_cache: HashMap<String, ValidationResult>,
}

impl FalsePositiveFilter {
    pub fn new(deepseek_client: Option<DeepSeekClient>) -> Self {
        let mut filter = Self {
            deepseek_client,
            known_patterns: HashMap::new(),
            validation_cache: HashMap::new(),
        };

        filter.initialize_known_patterns();
        filter
    }

    fn initialize_known_patterns(&mut self) {
        // Environment Variable Patterns - Reading secrets from env is safe practice
        self.known_patterns.insert(
            "R051".to_string(),
            FalsePositivePattern {
                pattern_id: "R051".to_string(),
                context_keywords: vec![
                    "env::var(".to_string(),
                    "std::env::var(".to_string(),
                    "dotenvy::var(".to_string(),
                    "dotenv::var(".to_string(),
                    "config::get(".to_string(),
                    "from_env()".to_string(),
                    ".expect(".to_string(),
                    ".unwrap_or(".to_string(),
                ],
                common_false_positives: vec![
                    "env::var(\"".to_string(),
                    "std::env::var(\"".to_string(),
                    "API_KEY\")".to_string(),
                    "SECRET_KEY\")".to_string(),
                    "PASSWORD\")".to_string(),
                    "TOKEN\")".to_string(),
                    ".expect(\"".to_string(),
                ],
                confidence_threshold: 0.95,
            },
        );

        // Test Code Patterns - Many patterns are acceptable in test code
        self.known_patterns.insert(
            "R001".to_string(),
            FalsePositivePattern {
                pattern_id: "R001".to_string(),
                context_keywords: vec![
                    "#[test]".to_string(),
                    "#[should_panic]".to_string(),
                    "mod tests".to_string(),
                    "#[cfg(test)]".to_string(),
                    "test_".to_string(),
                    "_test".to_string(),
                    "tests/".to_string(),
                ],
                common_false_positives: vec![
                    "panic!(\"test".to_string(),
                    "panic!(\"Test".to_string(),
                    "panic!(\"Should".to_string(),
                    "panic!(\"Expected".to_string(),
                ],
                confidence_threshold: 0.9,
            },
        );

        // Assertion Patterns in Test Code
        self.known_patterns.insert(
            "R005".to_string(),
            FalsePositivePattern {
                pattern_id: "R005".to_string(),
                context_keywords: vec![
                    "#[test]".to_string(),
                    "#[cfg(test)]".to_string(),
                    "mod tests".to_string(),
                    "fn test_".to_string(),
                    "assert_eq!".to_string(),
                    "assert_ne!".to_string(),
                ],
                common_false_positives: vec![
                    "assert_eq!(".to_string(),
                    "assert_ne!(".to_string(),
                    "assert!(".to_string(),
                ],
                confidence_threshold: 0.8,
            },
        );

        // Intentional Unsafe Code Patterns - Safe when properly documented or following official patterns
        self.known_patterns.insert(
            "R013".to_string(),
            FalsePositivePattern {
                pattern_id: "R013".to_string(),
                context_keywords: vec![
                    "// SAFETY:".to_string(),
                    "// Safety:".to_string(),
                    "/// # Safety".to_string(),
                    "unsafe {".to_string(),
                    "transmute".to_string(),
                    "ptr::".to_string(),
                    "std::mem::".to_string(),
                    "sqlite3_auto_extension".to_string(),
                    "sqlite3_vec_init".to_string(),
                    "sqlite_vec::".to_string(),
                    "rusqlite::ffi::".to_string(),
                    "register_auto_extension".to_string(),
                    "extension".to_string(),
                    "C entrypoint".to_string(),
                    "library integration".to_string(),
                    "official example".to_string(),
                    "recommended pattern".to_string(),
                ],
                common_false_positives: vec![
                    "// SAFETY: ".to_string(),
                    "// This is safe because".to_string(),
                    "/// # Safety".to_string(),
                    "mem::transmute".to_string(),
                    "sqlite3_auto_extension(Some(std::mem::transmute".to_string(),
                    "std::mem::transmute(sqlite3_vec_init".to_string(),
                    "transmute::<*const (), SqliteExtensionFn>".to_string(),
                    "sqlite3_auto_extension(".to_string(),
                    "register_auto_extension(".to_string(),
                ],
                confidence_threshold: 0.85,
            },
        );

        // Development/Placeholder Patterns - Common in WIP code
        self.known_patterns.insert(
            "R002".to_string(),
            FalsePositivePattern {
                pattern_id: "R002".to_string(),
                context_keywords: vec![
                    "// TODO:".to_string(),
                    "/* TODO".to_string(),
                    "todo!(".to_string(),
                    "unimplemented!(".to_string(),
                    "unreachable!(".to_string(),
                    "#[allow(".to_string(),
                ],
                common_false_positives: vec![
                    "todo!(\"".to_string(),
                    "unimplemented!(\"".to_string(),
                    "// TODO: implement".to_string(),
                    "// TODO: add".to_string(),
                ],
                confidence_threshold: 0.6,
            },
        );

        // Intentional Ignore Patterns - Explicit discard is often correct
        self.known_patterns.insert(
            "R085".to_string(),
            FalsePositivePattern {
                pattern_id: "R085".to_string(),
                context_keywords: vec![
                    "let _ =".to_string(),
                    "tuple".to_string(),
                    "struct".to_string(),
                    "len()".to_string(),
                    "write!(".to_string(),
                    "writeln!(".to_string(),
                    "print!(".to_string(),
                    "println!(".to_string(),
                ],
                common_false_positives: vec![
                    "let _ = write!(".to_string(),
                    "let _ = writeln!(".to_string(),
                    "let _ = print!(".to_string(),
                    "let _ = println!(".to_string(),
                    "let _ = tuple".to_string(),
                    "let _ = struct".to_string(),
                ],
                confidence_threshold: 0.7,
            },
        );

        // Cryptographic Patterns - Often legitimate in crypto code
        self.known_patterns.insert(
            "R045".to_string(),
            FalsePositivePattern {
                pattern_id: "R045".to_string(),
                context_keywords: vec![
                    "use ring::".to_string(),
                    "use openssl::".to_string(),
                    "use rustls::".to_string(),
                    "Cipher".to_string(),
                    "Encrypt".to_string(),
                    "Decrypt".to_string(),
                    "Hash".to_string(),
                    "crypto::".to_string(),
                ],
                common_false_positives: vec![
                    "AES".to_string(),
                    "SHA256".to_string(),
                    "RSA".to_string(),
                    "HMAC".to_string(),
                ],
                confidence_threshold: 0.8,
            },
        );

        // File System Patterns - Often legitimate in system utilities
        self.known_patterns.insert(
            "R070".to_string(),
            FalsePositivePattern {
                pattern_id: "R070".to_string(),
                context_keywords: vec![
                    "std::fs::".to_string(),
                    "File::open(".to_string(),
                    "File::create(".to_string(),
                    "Path::new(".to_string(),
                    "PathBuf::".to_string(),
                    "config".to_string(),
                    "temp".to_string(),
                    "cache".to_string(),
                ],
                common_false_positives: vec![
                    "/tmp/".to_string(),
                    "/var/tmp/".to_string(),
                    "config.toml".to_string(),
                    ".cache/".to_string(),
                ],
                confidence_threshold: 0.6,
            },
        );

        // Network/Process Patterns - Common in system utilities
        self.known_patterns.insert(
            "R080".to_string(),
            FalsePositivePattern {
                pattern_id: "R080".to_string(),
                context_keywords: vec![
                    "Command::new(".to_string(),
                    "process::".to_string(),
                    "spawn(".to_string(),
                    "tokio::".to_string(),
                    "async".to_string(),
                    "await".to_string(),
                    "server".to_string(),
                    "client".to_string(),
                ],
                common_false_positives: vec![
                    "Command::new(\"ls\")".to_string(),
                    "Command::new(\"git\")".to_string(),
                    "localhost".to_string(),
                    "127.0.0.1".to_string(),
                ],
                confidence_threshold: 0.7,
            },
        );
    }

    pub async fn validate_issue(
        &mut self,
        issue: &Issue,
        file_content: &str,
    ) -> Result<ValidationResult> {
        let cache_key = format!("{}:{}:{}", issue.pattern_id, issue.line, issue.excerpt);

        // Check cache first
        if let Some(cached_result) = self.validation_cache.get(&cache_key) {
            debug!("Using cached validation result for {}", issue.pattern_id);
            return Ok(cached_result.clone());
        }

        // Quick heuristic check
        if let Some(quick_result) = self.quick_heuristic_check(issue, file_content) {
            self.validation_cache
                .insert(cache_key.clone(), quick_result.clone());
            return Ok(quick_result);
        }

        // DeepSeek validation
        let validation_result = if let Some(deepseek) = &self.deepseek_client {
            self.deepseek_validation(issue, file_content, deepseek)
                .await?
        } else {
            // Fallback to pattern-based validation
            self.pattern_based_validation(issue, file_content)
        };

        // Cache the result
        self.validation_cache
            .insert(cache_key, validation_result.clone());
        Ok(validation_result)
    }

    fn quick_heuristic_check(&self, issue: &Issue, file_content: &str) -> Option<ValidationResult> {
        if let Some(pattern) = self.known_patterns.get(issue.pattern_id) {
            let issue_line = file_content.lines().nth(issue.line.saturating_sub(1))?;
            let context_lines: Vec<&str> = file_content
                .lines()
                .skip(issue.line.saturating_sub(6))
                .take(10)
                .collect();

            let context_text = context_lines.join("\n");

            // Check for contextual keywords that indicate safe usage
            let has_safe_context = pattern
                .context_keywords
                .iter()
                .any(|keyword| context_text.contains(keyword) || issue_line.contains(keyword));

            // Check for known false positive patterns
            let matches_false_positive = pattern.common_false_positives.iter().any(|fp_pattern| {
                issue.excerpt.contains(fp_pattern) || issue_line.contains(fp_pattern)
            });

            // Generic context analysis
            let context_score = self.analyze_context_safety(issue, &context_text, issue_line);

            // Determine if this is likely a false positive
            if has_safe_context || matches_false_positive || context_score > 0.7 {
                let confidence = if has_safe_context && matches_false_positive {
                    pattern.confidence_threshold
                } else if has_safe_context || matches_false_positive {
                    pattern.confidence_threshold * 0.8
                } else {
                    context_score
                };

                let reasoning = self.generate_reasoning(
                    issue,
                    has_safe_context,
                    matches_false_positive,
                    context_score,
                );

                return Some(ValidationResult {
                    is_false_positive: true,
                    confidence,
                    reasoning,
                    suggested_action: if confidence > 0.8 {
                        SuggestedAction::Ignore
                    } else {
                        SuggestedAction::Review
                    },
                });
            }

            // Special pattern-specific checks
            if let Some(specific_result) =
                self.pattern_specific_checks(issue, issue_line, &context_text)
            {
                return Some(specific_result);
            }
        }

        None
    }

    fn analyze_context_safety(&self, _issue: &Issue, context: &str, issue_line: &str) -> f32 {
        let mut safety_score: f32 = 0.0;

        // Check for common safe patterns
        let safe_indicators = [
            // Environment variable reading
            ("env::var(", 0.9),
            ("std::env::var(", 0.9),
            ("dotenvy::", 0.8),
            ("config::", 0.7),
            // Test context
            ("#[test]", 0.9),
            ("#[cfg(test)]", 0.9),
            ("mod tests", 0.9),
            ("fn test_", 0.8),
            ("#[should_panic]", 0.9),
            // Documentation/comments indicating safety
            ("// SAFETY:", 0.8),
            ("/// # Safety", 0.8),
            ("// This is safe", 0.7),
            ("// Safe because", 0.7),
            // Library integration patterns (often from official docs)
            ("sqlite3_auto_extension", 0.85),
            ("register_auto_extension", 0.85),
            ("sqlite3_vec_init", 0.9),
            ("sqlite_vec::", 0.8),
            ("rusqlite::ffi::", 0.8),
            ("C entrypoint", 0.8),
            ("extension", 0.7),
            ("official", 0.7),
            ("recommended", 0.7),
            ("documentation", 0.7),
            ("library", 0.6),
            ("crate", 0.6),
            // Example/demo context
            ("example", 0.7),
            ("demo", 0.7),
            ("tutorial", 0.7),
            ("guide", 0.6),
            ("usage", 0.6),
            // Intentional patterns
            ("// TODO:", 0.6),
            ("// FIXME:", 0.6),
            ("// NOTE:", 0.5),
            ("// Intentionally", 0.7),
            // Standard library safe patterns
            ("std::fs::", 0.6),
            ("Path::new(", 0.6),
            ("write!(", 0.8),
            ("println!(", 0.8),
        ];

        for (indicator, score) in safe_indicators {
            if context.contains(indicator) || issue_line.contains(indicator) {
                safety_score = safety_score.max(score);
            }
        }

        // Check for development/example context
        let dev_indicators = ["example", "demo", "test", "mock", "stub", "placeholder"];
        for indicator in dev_indicators {
            if context.to_lowercase().contains(indicator) {
                safety_score = safety_score.max(0.6);
            }
        }

        safety_score
    }

    fn generate_reasoning(
        &self,
        issue: &Issue,
        has_safe_context: bool,
        matches_false_positive: bool,
        context_score: f32,
    ) -> String {
        let mut reasons = Vec::new();

        if has_safe_context {
            reasons.push(
                "Code appears in safe context (test, documented unsafe, env var reading, etc.)"
                    .to_string(),
            );
        }

        if matches_false_positive {
            reasons.push("Matches known false positive pattern".to_string());
        }

        if context_score > 0.7 {
            reasons.push(format!(
                "Context analysis indicates safe usage (score: {:.2})",
                context_score
            ));
        }

        match issue.pattern_id {
            "R051" => {
                reasons.push("Environment variable reading is standard secure practice".to_string())
            }
            "R001" | "R002" | "R003" => reasons.push(
                "Panic/TODO patterns are often acceptable in test or development code".to_string(),
            ),
            "R013" => {
                reasons.push("Unsafe code may be intentional and properly documented".to_string())
            }
            "R085" => reasons.push(
                "Intentional discard patterns are often correct for non-Result types".to_string(),
            ),
            _ => {}
        }

        if reasons.is_empty() {
            format!(
                "Pattern {} detected but context suggests low risk",
                issue.pattern_id
            )
        } else {
            reasons.join("; ")
        }
    }

    fn pattern_specific_checks(
        &self,
        issue: &Issue,
        issue_line: &str,
        context: &str,
    ) -> Option<ValidationResult> {
        match issue.pattern_id {
            // R013: Unsafe patterns - Check for library integration patterns
            "R013" => {
                // SQLite extension patterns (official recommended usage)
                if (issue_line.contains("sqlite3_auto_extension")
                    || issue_line.contains("register_auto_extension")
                    || context.contains("sqlite3_vec_init")
                    || context.contains("sqlite_vec::"))
                    && (issue_line.contains("std::mem::transmute")
                        || issue_line.contains("transmute"))
                {
                    return Some(ValidationResult {
                        is_false_positive: true,
                        confidence: 0.95,
                        reasoning:
                            "SQLite extension registration following official documentation pattern"
                                .to_string(),
                        suggested_action: SuggestedAction::Ignore,
                    });
                }

                // General documented unsafe patterns
                if context.contains("// SAFETY:")
                    || context.contains("/// # Safety")
                    || context.contains("// This is safe")
                {
                    return Some(ValidationResult {
                        is_false_positive: true,
                        confidence: 0.8,
                        reasoning: "Unsafe code with proper safety documentation".to_string(),
                        suggested_action: SuggestedAction::Review,
                    });
                }
            }

            // R085: Check if it's actually discarding a Result
            "R085" => {
                if !issue_line.contains("Result<") && !context.contains("-> Result<") {
                    return Some(ValidationResult {
                        is_false_positive: true,
                        confidence: 0.8,
                        reasoning: "let _ = pattern not applied to Result type".to_string(),
                        suggested_action: SuggestedAction::Ignore,
                    });
                }
            }

            // R051: Environment variable patterns
            "R051" => {
                if issue_line.contains("env::var(") || issue_line.contains("std::env::var(") {
                    return Some(ValidationResult {
                        is_false_positive: true,
                        confidence: 0.95,
                        reasoning: "Reading environment variable is standard secure practice"
                            .to_string(),
                        suggested_action: SuggestedAction::Ignore,
                    });
                }
            }

            _ => {}
        }

        None
    }

    async fn deepseek_validation(
        &self,
        issue: &Issue,
        file_content: &str,
        deepseek: &DeepSeekClient,
    ) -> Result<ValidationResult> {
        let context_lines: Vec<&str> = file_content
            .lines()
            .skip(issue.line.saturating_sub(10))
            .take(20)
            .collect();

        let context = context_lines.join("\n");

        let validation_prompt = format!(
            r#"Analyze this Rust code pattern detection result for false positives:

Pattern ID: {}
Pattern Name: {}
Detected Issue: {}
Line {}: {}

Code Context:
```rust
{}
```

Please analyze if this is a FALSE POSITIVE by considering:
1. Is this in test code where the pattern might be acceptable?
2. Is the pattern being used safely in this context?
3. Is this a common development practice that's generally safe?
4. Does the surrounding code provide proper error handling?

Respond in this exact format:
FALSE_POSITIVE: [yes|no]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation]
ACTION: [IGNORE|REVIEW|FIX_IMMEDIATELY|MONITOR]

Be conservative - only mark as false positive if you're confident it's safe."#,
            issue.pattern_id,
            issue.name,
            issue.excerpt,
            issue.line,
            file_content
                .lines()
                .nth(issue.line.saturating_sub(1))
                .unwrap_or(""),
            context
        );

        match deepseek.analyze_code(&validation_prompt).await {
            Ok(response) => {
                debug!("DeepSeek validation response: {}", response);
                self.parse_deepseek_response(&response)
            }
            Err(e) => {
                warn!("DeepSeek validation failed: {}", e);
                // Fallback to conservative validation
                Ok(ValidationResult {
                    is_false_positive: false,
                    confidence: 0.5,
                    reasoning: format!("Could not validate with DeepSeek: {}", e),
                    suggested_action: SuggestedAction::Review,
                })
            }
        }
    }

    fn parse_deepseek_response(&self, response: &str) -> Result<ValidationResult> {
        let is_false_positive = response
            .lines()
            .find(|line| line.starts_with("FALSE_POSITIVE:"))
            .and_then(|line| line.split(':').nth(1))
            .map(|s| s.trim().to_lowercase() == "yes")
            .unwrap_or(false);

        let confidence = response
            .lines()
            .find(|line| line.starts_with("CONFIDENCE:"))
            .and_then(|line| line.split(':').nth(1))
            .and_then(|s| s.trim().parse::<f32>().ok())
            .unwrap_or(0.5);

        let reasoning = response
            .lines()
            .find(|line| line.starts_with("REASONING:"))
            .map(|line| {
                line.split(':')
                    .skip(1)
                    .collect::<Vec<_>>()
                    .join(":")
                    .trim()
                    .to_string()
            })
            .unwrap_or_else(|| "No reasoning provided".to_string());

        let suggested_action = response
            .lines()
            .find(|line| line.starts_with("ACTION:"))
            .and_then(|line| line.split(':').nth(1))
            .map(|s| match s.trim().to_uppercase().as_str() {
                "IGNORE" => SuggestedAction::Ignore,
                "FIX_IMMEDIATELY" => SuggestedAction::FixImmediately,
                "MONITOR" => SuggestedAction::MonitorForPatterns,
                _ => SuggestedAction::Review,
            })
            .unwrap_or(SuggestedAction::Review);

        Ok(ValidationResult {
            is_false_positive,
            confidence,
            reasoning,
            suggested_action,
        })
    }

    fn pattern_based_validation(&self, issue: &Issue, file_content: &str) -> ValidationResult {
        // Fallback pattern-based validation when DeepSeek is not available
        match issue.pattern_id {
            "R005" | "R006" | "R007" | "R008" => {
                // Assert patterns in test context
                let context = file_content
                    .lines()
                    .skip(issue.line.saturating_sub(20))
                    .take(40)
                    .collect::<Vec<_>>()
                    .join("\n");

                if context.contains("#[test]")
                    || context.contains("#[cfg(test)]")
                    || context.contains("mod tests")
                {
                    ValidationResult {
                        is_false_positive: true,
                        confidence: 0.8,
                        reasoning: "Assert pattern detected in test context".to_string(),
                        suggested_action: SuggestedAction::Ignore,
                    }
                } else {
                    ValidationResult {
                        is_false_positive: false,
                        confidence: 0.7,
                        reasoning: "Assert pattern outside test context".to_string(),
                        suggested_action: SuggestedAction::Review,
                    }
                }
            }
            "R085" => {
                // let _ = pattern validation
                let line = file_content
                    .lines()
                    .nth(issue.line.saturating_sub(1))
                    .unwrap_or("");
                if line.contains("Result<") || line.contains("()") {
                    ValidationResult {
                        is_false_positive: false,
                        confidence: 0.8,
                        reasoning: "Potentially discarding Result type".to_string(),
                        suggested_action: SuggestedAction::Review,
                    }
                } else {
                    ValidationResult {
                        is_false_positive: true,
                        confidence: 0.7,
                        reasoning: "let _ = used with non-Result type".to_string(),
                        suggested_action: SuggestedAction::Ignore,
                    }
                }
            }
            _ => ValidationResult {
                is_false_positive: false,
                confidence: 0.5,
                reasoning: "No specific validation rule available".to_string(),
                suggested_action: SuggestedAction::Review,
            },
        }
    }

    pub fn get_validation_stats(&self) -> (usize, usize) {
        let total = self.validation_cache.len();
        let false_positives = self
            .validation_cache
            .values()
            .filter(|v| v.is_false_positive)
            .count();
        (total, false_positives)
    }
}

// Serializable version of Issue that owns its strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableIssue {
    pub pattern_id: String,
    pub name: String,
    pub severity: String,
    pub category: String,
    pub line: usize,
    pub col: usize,
    pub excerpt: String,
}

impl From<Issue> for SerializableIssue {
    fn from(issue: Issue) -> Self {
        Self {
            pattern_id: issue.pattern_id.to_string(),
            name: issue.name.to_string(),
            severity: issue.severity.to_string(),
            category: issue.category.to_string(),
            line: issue.line,
            col: issue.col,
            excerpt: issue.excerpt,
        }
    }
}

// Enhanced Issue with validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedIssue {
    pub issue: SerializableIssue,
    pub is_valid: bool,
    pub confidence: f32,
    pub ai_reasoning: String,
    pub validation_method: String,
}

impl ValidatedIssue {
    pub fn new(issue: Issue, validation: ValidationResult) -> Self {
        let is_valid = !validation.is_false_positive;
        let confidence = validation.confidence;

        Self {
            issue: issue.into(),
            is_valid,
            confidence,
            ai_reasoning: validation.reasoning,
            validation_method: format!("{:?}", validation.suggested_action),
        }
    }
}

// ===== Rig-based Enhanced FP Validator =====

/// Minimal view of a finding (keeps module decoupled from internal Issue type).
#[derive(Debug)]
pub struct IssueView<'a> {
    pub pattern_id: &'a str,
    pub name: &'a str,
    pub category: &'a str,
    pub severity: &'a str,
    pub relative_path: &'a str,
    pub line: usize, // 1-based
    pub col: usize,  // 1-based
    pub excerpt: &'a str,
}

/// Structured decision the model must submit (strict JSON).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FpDecision {
    pub is_false_positive: bool,
    /// Confidence ∈ [0.0, 1.0]
    pub confidence: f32,
    /// One short, concrete reason (<= ~200 chars)
    pub reason: Option<String>,
    /// Optional refinement fields you can log or ignore
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub signals: Option<Vec<String>>,
}

/// Optional similar-example your caller can provide for the second pass.
#[derive(Debug, Clone, Serialize)]
pub struct Neighbor {
    pub id: String,
    pub score: f32,              // similarity score
    pub label: Option<String>,   // e.g. "true_bug" | "benign" | "prior_fp"
    pub snippet: Option<String>, // short code/evidence
}

/// Enhanced validation result with Rig context
#[derive(Debug, Clone)]
pub struct RigValidation {
    pub is_false_positive: bool,
    pub confidence: f32,   // [0.0, 1.0]
    pub reasoning: String, // short reason
    /// Optional enrichment if you want to surface context in logs
    pub decided_by: &'static str, // "rig_extractor_first" | "rig_extractor_second" | "rig_error"
}

/// The Rig-based validator: build once; call `validate()` per issue.
pub struct RigFpValidator {
    first: Extractor<deepseek::CompletionModel, FpDecision>,
    second: Extractor<deepseek::CompletionModel, FpDecision>,
    /// If the model calls FP with confidence below this, we ignore it (fail-open).
    ai_confidence_floor: f32,
    /// Borderline band that triggers a second pass.
    borderline_low: f32,
    borderline_high: f32,
    /// Lines of context around the hit for the second pass.
    context_radius: usize,
    /// How many neighbors you plan to pass in for second pass.
    neighbors_k: usize,
}

impl RigFpValidator {
    /// Build with reasonable defaults; override via env in your caller if you like.
    pub fn new(
        ai_confidence_floor: f32,
        borderline_low: f32,
        borderline_high: f32,
        context_radius: usize,
        neighbors_k: usize,
    ) -> Result<Self> {
        let client = deepseek::Client::from_env();

        let first = client
            .extractor::<FpDecision>(deepseek::DEEPSEEK_REASONER)
            .preamble(
                "You are a precise Rust static-analysis validator. \
                 Decide ONLY whether the flagged match is a FALSE POSITIVE. \
                 Return strictly via the submit tool that matches the provided schema. \
                 Be conservative: if uncertain, lower confidence.",
            )
            .build();

        let second = client
            .extractor::<FpDecision>(deepseek::DEEPSEEK_REASONER)
            .preamble(
                "Second pass re-evaluation with more evidence. \
                 You MUST reconsider the first assessment using the expanded context \
                 and any similar examples. Return ONLY one submit tool call.",
            )
            .build();

        Ok(Self {
            first,
            second,
            ai_confidence_floor: ai_confidence_floor.clamp(0.0, 1.0),
            borderline_low: borderline_low.clamp(0.0, 1.0),
            borderline_high: borderline_high.clamp(0.0, 1.0),
            context_radius: context_radius.max(3).min(200),
            neighbors_k: neighbors_k.min(20),
        })
    }

    /// Primary entry point. Provide neighbors when you have them; pass &[] otherwise.
    pub async fn validate(
        &self,
        issue: &IssueView<'_>,
        file_text_utf8: &str,
        repo_branch: &str,
        commit_hash: &str,
        neighbors: &[Neighbor],
    ) -> RigValidation {
        // ---------- First pass (cheap) ----------
        let first_prompt = format!(
            "Repository context:
- branch: {repo_branch}
- commit: {commit_hash}
- file: {path}
- location: L{line}:C{col}
- severity: {sev}
- category: {cat}
- pattern_id: {pid}
- finding: {name}

Flagged excerpt (verbatim):
```

{ex}

```

Decide if this is a FALSE POSITIVE. \
Return submit {{is_false_positive, confidence, reason, category?, signals?}}. \
Nothing else.",
            path = issue.relative_path,
            line = issue.line,
            col = issue.col,
            sev = issue.severity,
            cat = issue.category,
            pid = issue.pattern_id,
            name = issue.name,
            ex = issue.excerpt
        );

        let mut decided_by = "rig_extractor_first";
        let first_res = self.first.extract(&first_prompt).await;

        // On extractor failure, fail-open (non-FP, 0.0 conf) so we don't hide criticals.
        let mut dec = match first_res {
            Ok(v) => v,
            Err(e) => {
                return RigValidation {
                    is_false_positive: false,
                    confidence: 0.0,
                    reasoning: format!("Extractor error (first pass): {e}"),
                    decided_by: "rig_error",
                }
            }
        };

        dec.confidence = dec.confidence.clamp(0.0, 1.0);

        // If confident FP and above the floor, accept early.
        if dec.is_false_positive && dec.confidence >= self.ai_confidence_floor {
            return RigValidation {
                is_false_positive: true,
                confidence: dec.confidence,
                reasoning: dec.reason.unwrap_or_else(|| "No reason provided".into()),
                decided_by,
            };
        }

        // If clearly non-FP (confidence high), keep it.
        if !dec.is_false_positive && dec.confidence >= self.borderline_high {
            return RigValidation {
                is_false_positive: false,
                confidence: dec.confidence,
                reasoning: dec.reason.unwrap_or_else(|| "No reason provided".into()),
                decided_by,
            };
        }

        // ---------- Borderline second pass ----------
        if dec.confidence >= self.borderline_low && dec.confidence < self.borderline_high {
            decided_by = "rig_extractor_second";
            let window = self.context_window(file_text_utf8, issue.line, self.context_radius);
            let mut neighbor_block = String::new();
            if !neighbors.is_empty() {
                neighbor_block.push_str("Similar examples (caller-provided):\n");
                for (_i, n) in neighbors.iter().take(self.neighbors_k).enumerate() {
                    neighbor_block.push_str(&format!(
                        "- [{}] score={:.3} label={}\n{}\n",
                        n.id,
                        n.score,
                        n.label.as_deref().unwrap_or("n/a"),
                        n.snippet
                            .as_deref()
                            .unwrap_or("[no snippet provided]")
                            .lines()
                            .take(12)
                            .collect::<Vec<_>>()
                            .join("\n")
                    ));
                }
            }

            let second_prompt = format!(
                "Re-evaluate the finding with more context.

File: {path}
Location: L{line}:C{col}
Pattern: {pid} — {name} [{cat}, severity={sev}]

=== Surrounding code (±{rad} lines) ===
{window}
=== End surrounding code ===

{neighbor_block}

Task:
Return a single submit tool call with fields:
- is_false_positive (bool)
- confidence [0.0, 1.0]
- reason (short, concrete)
- category? (optional)
- signals? (optional)",
                path = issue.relative_path,
                line = issue.line,
                col = issue.col,
                pid = issue.pattern_id,
                name = issue.name,
                cat = issue.category,
                sev = issue.severity,
                rad = self.context_radius,
                window = window,
                neighbor_block = neighbor_block
            );

            match self.second.extract(&second_prompt).await {
                Ok(mut v2) => {
                    v2.confidence = v2.confidence.clamp(0.0, 1.0);
                    if v2.is_false_positive && v2.confidence >= self.ai_confidence_floor {
                        return RigValidation {
                            is_false_positive: true,
                            confidence: v2.confidence,
                            reasoning: v2.reason.unwrap_or_else(|| "No reason provided".into()),
                            decided_by,
                        };
                    } else {
                        return RigValidation {
                            is_false_positive: false,
                            confidence: v2.confidence.max(dec.confidence),
                            reasoning: v2
                                .reason
                                .or(dec.reason)
                                .unwrap_or_else(|| "No reason provided".into()),
                            decided_by,
                        };
                    }
                }
                Err(e) => {
                    return RigValidation {
                        is_false_positive: false,
                        confidence: dec.confidence,
                        reasoning: format!("Extractor error (second pass): {e}"),
                        decided_by: "rig_error",
                    };
                }
            }
        }

        // ---------- Default: keep first verdict (not confident enough to auto-FP) ----------
        RigValidation {
            is_false_positive: false,
            confidence: dec.confidence,
            reasoning: dec.reason.unwrap_or_else(|| "No reason provided".into()),
            decided_by,
        }
    }

    fn context_window(&self, src: &str, line_1_based: usize, radius: usize) -> String {
        let mut out = String::new();
        let lines: Vec<&str> = src.lines().collect();
        if lines.is_empty() {
            return out;
        }
        let idx = line_1_based.saturating_sub(1);
        let start = idx.saturating_sub(radius);
        let end = (idx + radius + 1).min(lines.len());
        for (i, l) in lines[start..end].iter().enumerate() {
            out.push_str(&format!("{:>6} | {}\n", start + i + 1, l));
        }
        out
    }
}
