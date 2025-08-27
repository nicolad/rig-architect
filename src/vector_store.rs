//! Vector Store Manager for bug pattern similarity search using FastEmbed
//!
//! This module uses FastEmbed for local embeddings (no API key required) with in-memory or SQLite storage.
//! Embeddings are generated locally using the AllMiniLML6V2 model.
//! Also includes local-first classification functionality with LLM fallback.
//! Contains FastEmbed integration for testing and reference implementations.

use crate::deepseek::DeepSeekClient;
use anyhow::{anyhow, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, trace, warn};

use fastembed::{
    read_file_to_bytes, EmbeddingModel as FastembedModel, Pooling,
    TextEmbedding as FastembedTextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use rig::{
    embeddings::EmbeddingsBuilder,
    vector_store::{
        in_memory_store::InMemoryVectorStore, request::VectorSearchRequest, VectorStoreIndex,
    },
    Embed,
};
use rig_fastembed::EmbeddingModel;

// Bug pattern document that will be embedded and searched
#[derive(Embed, Clone, Deserialize, Debug, Serialize, Eq, PartialEq, Default)]
pub struct BugPatternDocument {
    pub id: String,
    pub category: String,
    pub severity: u8,
    // The content field will be used to generate embeddings
    #[embed]
    pub content: String,
}

// FastEmbed testing - Shape of data that needs to be RAG'ed.
// The definition field will be used to generate embeddings.
#[derive(Embed, Clone, Deserialize, Debug, Serialize, Eq, PartialEq, Default)]
struct WordDefinition {
    id: String,
    word: String,
    #[embed]
    definitions: Vec<String>,
}

pub struct VectorStoreManager {
    pub vector_store: InMemoryVectorStore<BugPatternDocument>,
    pub embedding_model: EmbeddingModel,
    pub documents: Vec<BugPatternDocument>,
}

impl VectorStoreManager {
    /// Initialize vector store with FastEmbed local embeddings
    pub async fn new() -> Result<Self> {
        info!("🔧 Initializing vector store with FastEmbed local embeddings...");
        debug!("Starting vector store initialization process");

        // Try to use pre-downloaded model first, fall back to automatic download
        trace!("Creating FastEmbed embedding model");
        let embedding_model = Self::create_embedding_model().await?;
        info!("✅ FastEmbed embedding model created successfully");

        debug!("Initializing in-memory vector store");
        let mut manager = Self {
            vector_store: InMemoryVectorStore::default(),
            embedding_model,
            documents: Vec::new(),
        };

        // Initialize with default patterns
        debug!("Loading default bug patterns into vector store");
        manager.initialize_default_patterns().await?;

        info!("✅ Vector store initialized successfully with FastEmbed");
        debug!(
            "Vector store ready with {} patterns loaded",
            manager.documents.len()
        );
        Ok(manager)
    }

    /// Create embedding model - try local files first, then auto-download
    async fn create_embedding_model() -> Result<EmbeddingModel> {
        let model_dir = Path::new("./models/Qdrant--all-MiniLM-L6-v2-onnx/snapshots");

        if model_dir.exists() {
            info!(
                "Loading FastEmbed model from local directory: {}",
                model_dir.display()
            );
            debug!(
                "Model directory path: {}",
                model_dir
                    .canonicalize()
                    .unwrap_or_else(|_| model_dir.to_path_buf())
                    .display()
            );
            Self::load_local_model(model_dir).await
        } else {
            info!(
                "Local model not found at {}, using automatic download",
                model_dir.display()
            );
            debug!("Will download model to default cache location");
            Self::load_auto_model().await
        }
    }

    /// Load model from local files
    async fn load_local_model(model_dir: &Path) -> Result<EmbeddingModel> {
        debug!(
            "Loading FastEmbed model files from: {}",
            model_dir.display()
        );

        // Get model info
        let test_model_info =
            FastembedTextEmbedding::get_model_info(&FastembedModel::AllMiniLML6V2)?;

        // Load model files
        let onnx_path = model_dir.join("model.onnx");
        debug!("Loading ONNX model file: {}", onnx_path.display());
        let onnx_file = read_file_to_bytes(&onnx_path)?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("config.json");
        let special_tokens_path = model_dir.join("special_tokens_map.json");
        let tokenizer_config_path = model_dir.join("tokenizer_config.json");

        debug!("Loading tokenizer files: tokenizer={}, config={}, special_tokens={}, tokenizer_config={}", 
               tokenizer_path.display(), config_path.display(), special_tokens_path.display(), tokenizer_config_path.display());

        let tokenizer_files = TokenizerFiles {
            tokenizer_file: read_file_to_bytes(&tokenizer_path)?,
            config_file: read_file_to_bytes(&config_path)?,
            special_tokens_map_file: read_file_to_bytes(&special_tokens_path)?,
            tokenizer_config_file: read_file_to_bytes(&tokenizer_config_path)?,
        };

        // Create embedding model
        let user_defined_model =
            UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files).with_pooling(Pooling::Mean);

        let embedding_model = EmbeddingModel::new_from_user_defined(
            user_defined_model,
            384, // Dimension for AllMiniLML6V2
            test_model_info,
        );

        info!(
            "✅ Successfully loaded local FastEmbed model from {}",
            model_dir.display()
        );
        Ok(embedding_model)
    }

    /// Load model with automatic download
    async fn load_auto_model() -> Result<EmbeddingModel> {
        info!("Downloading FastEmbed model automatically...");
        debug!("Using automatic download for FastembedModel::AllMiniLML6V2");

        // Create init options for the FastEmbed model
        let init_options = fastembed::InitOptions::new(FastembedModel::AllMiniLML6V2);

        // First create the FastEmbed model instance
        let _fastembed_model = FastembedTextEmbedding::try_new(init_options)?;

        // Get the model info
        let model_info = FastembedTextEmbedding::get_model_info(&FastembedModel::AllMiniLML6V2)?;

        // Create the rig embedding model - use reference to the model enum, not the instance
        let embedding_model = EmbeddingModel::new(&FastembedModel::AllMiniLML6V2, model_info.dim);
        info!("✅ Successfully downloaded and initialized FastEmbed model");
        Ok(embedding_model)
    }

    /// Initialize with default bug patterns
    async fn initialize_default_patterns(&mut self) -> Result<()> {
        let default_patterns = vec![
            BugPatternDocument {
                id: "auth_bypass_001".to_string(),
                category: "AUTHENTICATION".to_string(),
                severity: 9,
                content: "Authentication bypass vulnerability in adapters: API key validation missing, weak token verification, credential exposure in logs, unauthorized access to privileged functions".to_string(),
            },
            BugPatternDocument {
                id: "rate_limit_bypass_001".to_string(),
                category: "RATE_LIMITING".to_string(),
                severity: 8,
                content: "Rate limiting bypass in exchange adapters: Multiple connection exploitation, request queue overflow, throttling mechanism failure, DoS attack vectors".to_string(),
            },
            BugPatternDocument {
                id: "websocket_security_001".to_string(),
                category: "WEBSOCKET".to_string(),
                severity: 7,
                content: "WebSocket security vulnerabilities: Unvalidated message injection, connection hijacking, authentication after connect, message replay attacks".to_string(),
            },
            BugPatternDocument {
                id: "order_execution_race_001".to_string(),
                category: "EXECUTION".to_string(),
                severity: 9,
                content: "Order execution race conditions: Concurrent order processing, double execution vulnerability, state synchronization issues, atomic operation failures".to_string(),
            },
            BugPatternDocument {
                id: "data_validation_001".to_string(),
                category: "VALIDATION".to_string(),
                severity: 6,
                content: "Market data validation failures: Unvalidated price feeds, malformed data processing, type conversion vulnerabilities, data integrity checks missing".to_string(),
            },
            BugPatternDocument {
                id: "connection_pool_001".to_string(),
                category: "CONNECTION".to_string(),
                severity: 5,
                content: "Connection pool issues: Pool exhaustion, resource leaks, timeout handling, connection state management, cleanup failures".to_string(),
            },
            BugPatternDocument {
                id: "memory_leak_001".to_string(),
                category: "MEMORY".to_string(),
                severity: 7,
                content: "Memory management issues: Memory leaks in adapters, resource cleanup failures, garbage collection problems, buffer overflows".to_string(),
            },
            BugPatternDocument {
                id: "performance_degradation_001".to_string(),
                category: "PERFORMANCE".to_string(),
                severity: 6,
                content: "Performance issues: Slow response times, high latency, CPU bottlenecks, inefficient algorithms, blocking operations".to_string(),
            },
            BugPatternDocument {
                id: "error_handling_001".to_string(),
                category: "ERROR_HANDLING".to_string(),
                severity: 8,
                content: "Error handling problems: Unhandled exceptions, silent failures, improper error propagation, missing error logging".to_string(),
            },
            BugPatternDocument {
                id: "configuration_001".to_string(),
                category: "CONFIGURATION".to_string(),
                severity: 7,
                content: "Configuration vulnerabilities: Hardcoded secrets, missing environment variables, insecure defaults, configuration injection".to_string(),
            },
        ];

        // Create embeddings using EmbeddingsBuilder
        let embeddings = EmbeddingsBuilder::new(self.embedding_model.clone())
            .documents(default_patterns.clone())?
            .build()
            .await?;

        // Create vector store from documents
        self.vector_store =
            InMemoryVectorStore::from_documents_with_id_f(embeddings, |doc| doc.id.clone());

        self.documents = default_patterns;

        info!(
            "Added {} default bug patterns to vector store with FastEmbed",
            self.documents.len()
        );
        Ok(())
    }

    /// Perform similarity search using FastEmbed
    pub async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>> {
        info!(
            "🔍 Performing vector similarity search for query: '{}'",
            query
        );
        debug!("Search parameters: query='{}', limit={}", query, limit);

        // Create index for searching
        let index = self
            .vector_store
            .clone()
            .index(self.embedding_model.clone());

        let req = VectorSearchRequest::builder()
            .query(query)
            .samples(limit as u64)
            .build()?;

        debug!("Executing vector search against embedded patterns...");
        // Query the index
        let results = index
            .top_n::<BugPatternDocument>(req)
            .await?
            .into_iter()
            .map(|(score, _id, doc)| {
                debug!(
                    "Found similar pattern: id='{}', category='{}', score={:.4}",
                    doc.id, doc.category, score
                );
                serde_json::json!({
                    "id": doc.id,
                    "content": doc.content,
                    "category": doc.category,
                    "severity": doc.severity,
                    "score": score
                })
            })
            .collect::<Vec<_>>();

        info!(
            "✅ Vector search completed: found {} similar patterns for query '{}'",
            results.len(),
            query
        );
        if results.is_empty() {
            debug!("No similar patterns found for query: '{}'", query);
        } else {
            debug!(
                "Top result: {} (score: {:.4})",
                results[0]["id"].as_str().unwrap_or("unknown"),
                results[0]["score"].as_f64().unwrap_or(0.0)
            );
        }
        Ok(results)
    }

    /// Add a new bug pattern to the vector store using FastEmbed
    pub async fn add_pattern(&mut self, pattern: BugPatternDocument) -> Result<()> {
        info!(
            "📄 Adding bug pattern to vector store: id='{}', category='{}'",
            pattern.id, pattern.category
        );
        debug!(
            "Pattern details: severity={}, content_length={}",
            pattern.severity,
            pattern.content.len()
        );
        // Create embeddings for the new pattern
        let embeddings = EmbeddingsBuilder::new(self.embedding_model.clone())
            .documents(vec![pattern.clone()])?
            .build()
            .await?;

        // Add to vector store
        self.vector_store
            .add_documents_with_id_f(embeddings, |doc| doc.id.clone());
        self.documents.push(pattern);

        Ok(())
    }

    /// Get pattern by ID
    pub async fn get_pattern_by_id(&self, pattern_id: &str) -> Result<Option<BugPatternDocument>> {
        Ok(self.documents.iter().find(|p| p.id == pattern_id).cloned())
    }

    /// Get all patterns in a specific category
    pub async fn get_patterns_by_category(
        &self,
        category: &str,
    ) -> Result<Vec<BugPatternDocument>> {
        Ok(self
            .documents
            .iter()
            .filter(|p| p.category == category)
            .cloned()
            .collect())
    }

    /// Get patterns by severity level
    pub async fn get_patterns_by_severity(
        &self,
        min_severity: u8,
    ) -> Result<Vec<BugPatternDocument>> {
        Ok(self
            .documents
            .iter()
            .filter(|p| p.severity >= min_severity)
            .cloned()
            .collect())
    }
}

// ========== CLASSIFICATION FUNCTIONALITY ==========

/// A pattern label + examples. Provide 3-8 concise examples per label for best results.
#[derive(Debug, Clone)]
pub struct PatternSpec {
    pub label: String,
    pub description: Option<String>,
    pub examples: Vec<String>,
}

/// Document we embed & store in the vector index for classification.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PatternExample {
    pub id: String,
    pub label: String,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LlmDecision {
    /// Either one of the provided labels, or "NONE"
    pub label: String,
    /// 0.0..=1.0 model-estimated confidence
    pub confidence: f32,
    /// Short human-readable rationale
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum Decision {
    Match {
        label: String,
        score: f32,
        reason: String,
        via: &'static str, // "embeddings" | "deepseek"
    },
    NoMatch {
        score: f32,     // best embedding score we saw
        reason: String, // why we declined
    },
}

// --------- Tunables ---------

const TOP_K: usize = 6; // search this many nearest examples
const MIN_SCORE: f64 = 0.68; // absolute floor for a confident local match (cosine)
const MARGIN: f64 = 0.04; // how much top label must beat the runner-up

// --------- Simple pattern index using basic similarity ---------

struct PatternIndex {
    examples: Vec<PatternExample>,
    labels: Vec<String>,
}

async fn build_index(specs: &[PatternSpec]) -> Result<PatternIndex> {
    // Prepare documents (examples)
    let mut examples: Vec<PatternExample> = Vec::new();
    for (li, spec) in specs.iter().enumerate() {
        for (ei, ex) in spec.examples.iter().enumerate() {
            examples.push(PatternExample {
                id: format!("L{li}_E{ei}"),
                label: spec.label.clone(),
                text: ex.trim().to_owned(),
                embedding: None, // We'll use simple text matching for now
            });
        }
    }

    Ok(PatternIndex {
        examples,
        labels: specs.iter().map(|s| s.label.clone()).collect(),
    })
}

// --------- Simple text similarity (placeholder for embeddings) ---------

fn simple_similarity(text1: &str, text2: &str) -> f64 {
    // Very basic similarity based on common words
    let text1_lower = text1.to_lowercase();
    let text2_lower = text2.to_lowercase();
    let words1: std::collections::HashSet<&str> = text1_lower.split_whitespace().collect();
    let words2: std::collections::HashSet<&str> = text2_lower.split_whitespace().collect();

    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

async fn coarse_match(
    index: &PatternIndex,
    input: &str,
) -> Result<(Option<(String, f64, String)>, f64, Vec<(String, f64)>)> {
    // Calculate similarity scores for all examples
    let mut scores: Vec<(f64, &PatternExample)> = index
        .examples
        .iter()
        .map(|example| {
            let score = simple_similarity(input, &example.text);
            (score, example)
        })
        .collect();

    // Sort by score descending
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    if scores.is_empty() {
        return Ok((None, 0.0, vec![]));
    }

    // Take top K results
    let top_results: Vec<(f64, &PatternExample)> = scores.into_iter().take(TOP_K).collect();

    // Aggregate best score per label and keep an example for rationale
    let mut best: HashMap<String, (f64, String)> = HashMap::new();
    let mut all_best: Vec<(String, f64)> = vec![];

    for (score, example) in top_results {
        match best.get(&example.label) {
            Some(&(existing, _)) if existing >= score => {}
            _ => {
                best.insert(example.label.clone(), (score, example.text.clone()));
            }
        }
    }

    // Rank labels by their best example score
    let mut label_scores: Vec<(String, f64, String)> = best
        .into_iter()
        .map(|(label, (s, ex))| (label, s, ex))
        .collect();

    label_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_score = label_scores.first().map(|x| x.1).unwrap_or(0.0);
    let top: Option<(String, f64, String)> = label_scores.first().cloned();

    // For auditability, return the top-N label scores as well
    for (label, score, _ex) in label_scores.iter().take(5) {
        all_best.push((label.clone(), *score));
    }

    Ok((top, top_score, all_best))
}

// --------- DeepSeek tie-break (JSON-only, parsed with serde) ---------

fn extract_json_block(s: &str) -> Option<&str> {
    // Be robust to models that wrap JSON in prose/fences.
    let start = s.find('{')?;
    let end = s.rfind('}')?;
    if end > start {
        Some(&s[start..=end])
    } else {
        None
    }
}

async fn llm_tiebreak_with_deepseek(
    ds: &DeepSeekClient,
    labels: &[String],
    input: &str,
    descriptions: &HashMap<String, String>,
) -> Result<LlmDecision> {
    let label_list = labels.join(", ");
    let mut label_table = String::new();
    for l in labels {
        let desc = descriptions.get(l).map(|s| s.as_str()).unwrap_or("");
        if desc.is_empty() {
            label_table.push_str(&format!("- {l}\n"));
        } else {
            label_table.push_str(&format!("- {l}: {desc}\n"));
        }
    }

    // Strong JSON-only instruction; model-agnostic.
    let prompt = format!(
        "You are a STRICT classifier. Your job is to assign the input to ONE label or \"NONE\".\n\
         Allowed labels: {label_list}\n\
         Label overview:\n{label_table}\n\
         RULES:\n\
         - Consider semantics, not substrings.\n\
         - Prefer precision over recall.\n\
         - If nothing fits well, use label \"NONE\".\n\
         - OUTPUT ONLY VALID JSON (no backticks, no explanations), matching this schema exactly:\n\
         {{\"label\": string, \"confidence\": number (0..1), \"reason\": string}}\n\
         Now classify the INPUT below.\n\n\
         INPUT:\n{input}\n"
    );

    let raw = ds
        .prompt_with_context(&prompt, "LocalFirst-Classifier")
        .await?;

    // Try direct parse; then fallback to extracting a JSON block.
    match serde_json::from_str::<LlmDecision>(&raw) {
        Ok(dec) => Ok(dec),
        Err(_) => {
            let json = extract_json_block(&raw).ok_or_else(|| {
                anyhow!("DeepSeek returned non-JSON answer and no JSON block could be found")
            })?;
            let dec: LlmDecision = serde_json::from_str(json)?;
            Ok(dec)
        }
    }
}

/// Local-first classifier:
/// 1) try embeddings-only; 2) if ambiguous, consult DeepSeek via your client wrapper.
pub async fn classify_local_first(
    specs: &[PatternSpec],
    input: &str,
    deepseek: Option<&DeepSeekClient>, // None => skip LLM
) -> Result<Decision> {
    let index = build_index(specs).await?;

    // Build map of label -> description for the LLM step
    let mut descs: HashMap<String, String> = HashMap::new();
    for s in specs {
        if let Some(d) = &s.description {
            descs.insert(s.label.clone(), d.clone());
        }
    }

    // Coarse local-only match
    let (top, top_score, ranked) = coarse_match(&index, input).await?;

    if let Some((label, score, example)) = top {
        // Grab runner-up
        let second = ranked.get(1).map(|(_, s)| *s).unwrap_or(0.0);

        info!("coarse ranked: {:?}", ranked);

        // Accept if the top label is clearly best
        if score >= MIN_SCORE && (score - second) >= MARGIN {
            return Ok(Decision::Match {
                label,
                score: score as f32,
                reason: format!(
                    "Local embeddings agreed confidently (score={:.3}, margin over next={:.3}). Example: {}",
                    score, score - second, example
                ),
                via: "embeddings",
            });
        }
    }

    // Otherwise, ambiguous; try DeepSeek if provided
    if let Some(ds) = deepseek {
        match llm_tiebreak_with_deepseek(ds, &index.labels, input, &descs).await {
            Ok(dec) => {
                if dec.label == "NONE" {
                    return Ok(Decision::NoMatch {
                        score: top_score as f32,
                        reason: format!("DeepSeek declined to classify. LLM says: {}", dec.reason),
                    });
                } else {
                    return Ok(Decision::Match {
                        label: dec.label,
                        score: dec.confidence,
                        reason: dec.reason,
                        via: "deepseek",
                    });
                }
            }
            Err(e) => {
                warn!("DeepSeek tie-break failed: {e:?}");
                // Fall back to "no match" but report embedding best-score
                return Ok(Decision::NoMatch {
                    score: top_score as f32,
                    reason: "Ambiguous by embeddings and DeepSeek tie-break unavailable."
                        .to_string(),
                });
            }
        }
    }

    // LLM not allowed / not provided
    Ok(Decision::NoMatch {
        score: top_score as f32,
        reason: "Ambiguous by embeddings; DeepSeek tie-break disabled.".to_string(),
    })
}

/// Demo function that can be called from main or tests
pub async fn run_classification_demo() -> Result<()> {
    info!("🧠 Running classification demo...");

    // NOTE: The DeepSeek step is optional; if you don't set DEEPSEEK_API_KEY,
    // we'll run embeddings-only and return NoMatch on ambiguity.
    let deepseek = DeepSeekClient::from_env().ok();

    // Example patterns (replace with your own domain)
    let specs = vec![
        PatternSpec {
            label: "ORDER_ID".to_string(),
            description: Some(
                "Alphanumeric order identifiers like ORD-12345, PO#998877".to_string(),
            ),
            examples: vec![
                "Please check order ORD-12345 status".to_string(),
                "po#998877 needs an address update".to_string(),
                "ship this as order 77-ABC-432".to_string(),
            ],
        },
        PatternSpec {
            label: "ERROR_LOG".to_string(),
            description: Some("Application errors with codes/messages".to_string()),
            examples: vec![
                "ERROR [E042] null pointer at module:auth".to_string(),
                "FATAL: db timeout code=DBT-504".to_string(),
                "warn: retrying connection, code=ECONNRESET".to_string(),
            ],
        },
        PatternSpec {
            label: "GREETING".to_string(),
            description: Some("Human salutations".to_string()),
            examples: vec![
                "hey team, quick question".to_string(),
                "hello there!".to_string(),
                "good morning everyone".to_string(),
            ],
        },
    ];

    let inputs = vec![
        "Can you re-route PO#998877 to warehouse B?",
        "FATAL: db timeout code=DBT-504 after 30s",
        "morning folks — deploy passed ✅",
        "unknown shape: maybe a tracking ref 99-XY but not sure",
    ];

    for text in inputs {
        let decision = classify_local_first(&specs, text, deepseek.as_ref()).await?;
        info!("\nINPUT: {text}");
        match decision {
            Decision::Match {
                label,
                score,
                reason,
                via,
            } => {
                info!("→ MATCH: {label}  (score {:.3}, via {via})", score);
                info!("  reason: {reason}");
            }
            Decision::NoMatch { score, reason } => {
                info!("→ NO MATCH (best local score {:.3})", score);
                info!("  reason: {reason}");
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vector_store_creation() {
        // FastEmbed doesn't require API keys - runs locally
        let manager = VectorStoreManager::new().await.unwrap();
        let results = manager
            .similarity_search("authentication", 3)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_add_pattern() {
        // FastEmbed doesn't require API keys - runs locally
        let mut manager = VectorStoreManager::new().await.unwrap();

        let new_pattern = BugPatternDocument {
            id: "test_pattern_001".to_string(),
            category: "TEST".to_string(),
            severity: 5,
            content: "Test pattern for unit testing: This is a test vulnerability pattern"
                .to_string(),
        };

        manager.add_pattern(new_pattern).await.unwrap();
    }

    #[tokio::test]
    async fn test_similarity_search() {
        let manager = VectorStoreManager::new().await.unwrap();

        // Test searching for authentication-related issues
        let results = manager
            .similarity_search("API key security authentication", 3)
            .await
            .unwrap();
        assert!(!results.is_empty());

        // The first result should be related to authentication
        let first_result = &results[0];
        assert!(
            first_result["category"]
                .as_str()
                .unwrap_or("")
                .contains("AUTHENTICATION")
                || first_result["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_lowercase()
                    .contains("authentication")
        );
    }

    #[tokio::test]
    async fn test_category_filtering() {
        let manager = VectorStoreManager::new().await.unwrap();
        let auth_patterns = manager
            .get_patterns_by_category("AUTHENTICATION")
            .await
            .unwrap();
        assert!(!auth_patterns.is_empty());
    }

    #[tokio::test]
    async fn test_severity_filtering() {
        let manager = VectorStoreManager::new().await.unwrap();
        let critical_patterns = manager.get_patterns_by_severity(8).await.unwrap();
        assert!(!critical_patterns.is_empty());
    }
}

#[allow(dead_code)]
pub async fn run_fastembed_test() -> Result<()> {
    // Get model info
    let test_model_info =
        FastembedTextEmbedding::get_model_info(&FastembedModel::AllMiniLML6V2).unwrap();

    // Set up model directory
    let model_dir = Path::new("./models/Qdrant--all-MiniLM-L6-v2-onnx/snapshots");
    println!("Loading model from: {model_dir:?}");

    let embedding_model = if model_dir.exists() {
        // Load model files
        let onnx_file = read_file_to_bytes(&model_dir.join("model.onnx"))
            .expect("Could not read model.onnx file");

        let tokenizer_files = TokenizerFiles {
            tokenizer_file: read_file_to_bytes(&model_dir.join("tokenizer.json"))
                .expect("Could not read tokenizer.json"),
            config_file: read_file_to_bytes(&model_dir.join("config.json"))
                .expect("Could not read config.json"),
            special_tokens_map_file: read_file_to_bytes(&model_dir.join("special_tokens_map.json"))
                .expect("Could not read special_tokens_map.json"),
            tokenizer_config_file: read_file_to_bytes(&model_dir.join("tokenizer_config.json"))
                .expect("Could not read tokenizer_config.json"),
        };

        // Create embedding model
        let user_defined_model =
            UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files).with_pooling(Pooling::Mean);

        EmbeddingModel::new_from_user_defined(user_defined_model, 384, test_model_info)
    } else {
        println!("Model directory not found, using automatic download");
        // Create init options for the FastEmbed model
        let init_options = fastembed::InitOptions::new(FastembedModel::AllMiniLML6V2);

        // First create the FastEmbed model instance
        let _fastembed_model = FastembedTextEmbedding::try_new(init_options)?;

        // Get the model info
        let model_info = FastembedTextEmbedding::get_model_info(&FastembedModel::AllMiniLML6V2)?;

        // Create the rig embedding model - use reference to the model enum, not the instance
        EmbeddingModel::new(&FastembedModel::AllMiniLML6V2, model_info.dim)
    };

    // Create documents
    let documents = vec![
        WordDefinition {
            id: "doc0".to_string(),
            word: "flurbo".to_string(),
            definitions: vec![
                "A green alien that lives on cold planets.".to_string(),
                "A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
            ]
        },
        WordDefinition {
            id: "doc1".to_string(),
            word: "glarb-glarb".to_string(),
            definitions: vec![
                "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
            ]
        },
        WordDefinition {
            id: "doc2".to_string(),
            word: "linglingdong".to_string(),
            definitions: vec![
                "A term used by inhabitants of the sombrero galaxy to describe humans.".to_string(),
                "A rare, mystical instrument crafted by the ancient monks of the Nebulon Mountain Ranges on the planet Quarm.".to_string()
            ]
        },
    ];

    // Create embeddings using EmbeddingsBuilder
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(documents)?
        .build()
        .await?;

    // Create vector store
    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |doc| doc.id.clone());
    let index = vector_store.index(embedding_model);

    let query =
        "I need to buy something in a fictional universe. What type of money can I use for this?";

    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build()?;

    let results = index
        .top_n::<WordDefinition>(req)
        .await?
        .into_iter()
        .map(|(score, id, doc)| (score, id, doc.word))
        .collect::<Vec<_>>();

    println!("Results: {results:?}");

    Ok(())
}

// Additional tests from the original fastembed.rs file
#[cfg(test)]
mod fastembed_tests {
    use super::*;

    #[tokio::test]
    async fn test_fastembed_integration() {
        run_fastembed_test().await.unwrap();
    }
}
