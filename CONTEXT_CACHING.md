# DeepSeek Context Caching Integration

This document explains how the AI Architecture Improver leverages DeepSeek's context caching functionality to reduce API costs and improve performance.

## Overview

DeepSeek automatically caches conversation prefixes on disk, providing significant cost savings:

- **Cache Hit**: 0.1 yuan per million tokens (90% discount)  
- **Cache Miss**: 1.0 yuan per million tokens (standard rate)

## Implementation

### 1. Enhanced DeepSeek Client (`src/deepseek.rs`)

- **Usage Tracking**: Extended `Usage` structure to track `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`
- **Cache Logging**: Automatic logging of cache hit statistics and efficiency percentages
- **Context Optimization**: Agent structure designed to maintain cache-friendly message ordering

### 2. Architecture Agent with Few-Shot Examples

The `architecture_agent()` method creates cache-optimized agents with pre-configured architectural improvement examples:

```rust
let mut improver = ds
    .architecture_agent(chat_model) // Cache-optimized with examples
    .preamble(&improver_preamble(...))
    .build();
```

Few-shot examples include common Rust architectural patterns:

- Iterator optimization techniques
- Builder pattern implementations  
- Error handling improvements
- Encapsulation best practices

### 3. Message Ordering for Cache Optimization

The system maintains consistent message ordering to maximize cache hits:

1. **System prompt** (preamble)
2. **Base context** (few-shot examples)
3. **Conversation history**
4. **Current user message**

This ordering ensures that the system prompt + few-shot examples form a consistent cache-friendly prefix across all conversations.

## Benefits

### Cost Reduction

- Up to 90% reduction in token costs for repeated architectural patterns
- Consistent prefix reuse across multiple improvement cycles
- Automatic cache optimization without code changes

### Performance

- Faster response times for cache hits
- Reduced computational overhead for repeated patterns
- Improved consistency in architectural recommendations

### Monitoring

- Automatic logging of cache hit rates
- Cache efficiency percentage tracking
- Performance metrics for optimization tuning

## Example Cache Hit Scenario

**First Request**: Complete system prompt + few-shot examples + new code analysis

- Total tokens: 5000
- Cache hits: 0
- Cache misses: 5000
- Cost: 5.0 yuan

**Second Request**: Same prefix + different code analysis  

- Total tokens: 5200
- Cache hits: 4800 (prefix)
- Cache misses: 400 (new content)
- Cost: 0.48 + 0.40 = 0.88 yuan (82% savings)

## Monitoring Cache Performance

The system logs cache statistics for each API call:

```
Context cache stats: 4800 hit tokens, 400 miss tokens, cache efficiency: 92.3%
```

This information helps monitor:

- Cache hit rates across different improvement types
- Cost optimization effectiveness
- Potential areas for further optimization

## Technical Details

### Cache Storage

- DeepSeek uses 64-token storage units
- Content under 64 tokens won't be cached
- Cache automatically expires when unused (hours to days)

### Cache Matching

- Only exact prefix matches trigger cache hits
- Message order and content must be identical
- Temperature and other parameters don't affect caching

### Best Practices

- Maintain consistent system prompts
- Order messages predictably (system → context → history → new)
- Use architecture_agent() for repeated architectural patterns
- Monitor cache efficiency in logs
