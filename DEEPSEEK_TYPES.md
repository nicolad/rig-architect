# DeepSeek API Type Generation System

## Overview

This document describes the canonical Rust type generation system implemented for the DeepSeek API integration.

## Architecture

### Generated Types Location

- **Primary types**: `src/deepseek/types.rs` (auto-generated)
- **Template**: `templates/types_template.rs` (source template)
- **Generator**: `bin/generate_types.rs` (generation script)

### Key Features

1. **Automatic Type Generation**
   - Run `cargo run --bin generate_types` to regenerate types
   - Based on actual DeepSeek API specification
   - Includes comprehensive serde annotations

2. **Comprehensive Type Coverage**
   - `ChatCompletionRequest` - Request payload with optional fields
   - `ChatCompletionResponse` - Complete response structure
   - `Message` - Chat messages with role and content
   - `Tool` & `Function` - Function calling support
   - `Usage` - Token usage with context caching support
   - `ErrorResponse` & `ErrorDetail` - Error handling types

3. **Context Caching Support**
   - `prompt_cache_hit_tokens` - Cached tokens (0.1 yuan per million)
   - `prompt_cache_miss_tokens` - Non-cached tokens (1 yuan per million)

4. **Model Constants**
   - `models::DEEPSEEK_CHAT` - Standard chat model
   - `models::DEEPSEEK_REASONER` - Enhanced reasoning model

## Module Structure

```
src/deepseek/
├── mod.rs           # Module exports and organization
├── client.rs        # API client implementation
└── types.rs         # Generated type definitions (auto-generated)
```

## Usage

### Generating Types

```bash
cargo run --bin generate_types
```

### Using Types in Code

```rust
use crate::deepseek::types::*;

let request = ChatCompletionRequest {
    model: models::DEEPSEEK_CHAT.to_string(),
    messages: vec![
        Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }
    ],
    max_tokens: Some(100),
    temperature: Some(0.7),
    top_p: None,
    stream: false,
    tools: None,
};
```

## Testing

All types include comprehensive test coverage:

- Serialization/deserialization tests
- Context caching field handling
- Optional field skipping
- Error case handling

Run tests with:

```bash
cargo test types
```

## Integration

The generated types are fully integrated with:

- DeepSeek API client (`src/deepseek/client.rs`)
- Integration tests (`tests/integration_deepseek.rs`)
- Main application architecture analysis features

## Benefits

1. **Type Safety**: Compile-time guarantees for API interactions
2. **Maintainability**: Single source of truth for type definitions
3. **API Compliance**: Generated from actual API specification
4. **Context Caching**: Built-in support for DeepSeek's caching features
5. **Extensibility**: Easy to add new types as API evolves

## Next Steps

- Types can be regenerated anytime with the generation script
- Template can be updated for API changes
- Additional types can be added to the template as needed
- Integration tests validate real API compatibility
