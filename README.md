# Token-Saving-hack
here we present a way to save tokens in llm's 

The main idea about it is , since we have static text that the ai has to read over every single time 

(its system instructions and what not ) every time it tries to make a new post

we thought, what if we cached that ?!

after consulting good ol gemini we got the following set up 

you can come to your own solutions with your own ai now that you have the idea tho ! 

feel free to experiment with it !!

*Co-authored by [sighthough](https://youtu.be/UtPiUGwu-0Q) and [Googles Gemini](https://www.youtube.com/shorts/R3Qo4rBgrD8).*







AI made guide on an example of implementation : 

Here is a complete architectural specification for AI engineers or system architects to implement **Prefix KV-Caching with Dynamic System Instruction Pre-Warming**.

---

## Architectural Overview

Standard LLM inference performs redundant matrix multiplications during the **Prefill Phase** for static text on every API call. By implementing a **Persisted KV-Cache Architecture**, system instructions, safety guardrails, schema definitions, and base mode configs are pre-computed once and pinned in high-speed GPU memory (VRAM).

Subsequent inference requests append state changes without re-computing the base prefix tensors.

```
+-----------------------------------------------------------------------+
|                       STATIC SYSTEM PREFIX (1K-8K Tokens)              |
| [System Rules | Layout Schema | Tool Declarations | Mode Specs]        |
+-----------------------------------------------------------------------+
                                   |
                         [Initial Warmup Pass]
                                   v
+-----------------------------------------------------------------------+
|                      GPU VRAM: PERSISTED KV CACHE                     |
|           Key Tensors: K_sys | Value Tensors: V_sys                      |
+-----------------------------------------------------------------------+
                                   |
                +------------------+------------------+
                |                                     |
         [User Turn N]                         [User Turn N+1]
                |                                     |
   Compute: K_user, V_user               Compute: K_user_new, V_user_new
   Concat: [K_sys, K_user]               Concat: [K_sys, K_user_new]
   Attention: Softmax(Q K^T) V           Attention: Softmax(Q K^T) V

```

---

## Technical Specifications & Implementation Guide

### 1. Prefix Layout Standardization

To maximize cache hit rates, the input context must be strictly segmented into **Immutable (Static)** and **Mutable (Dynamic)** layers. Any modification to a single character in the static prefix invalidates the cached KV tensors downstream.

```typescript
// System Context Structure
interface ModelInputPayload {
  // --- IMMUTABLE PREFIX (Target for Caching) ---
  system_instructions: string;   // Persona, guardrails, formatting definitions
  tool_definitions: object[];    // Function schemas, layout component specs
  default_mode_specs: string;     // Explicit mode & capability descriptions
  
  // --- MUTABLE CONTEXT (Evaluated dynamically) ---
  conversation_history: Message[]; // Prior user/assistant turns (appended)
  current_user_prompt: string;    // Latest input turn
}

```

#### Cache Key Hash Formulation

Generate a deterministic SHA-256 fingerprint of the normalized immutable block:

$$\text{CacheKey} = \text{SHA256}(\text{Normalize}(\text{SystemInstructions} + \text{ToolDefinitions}))$$

---

### 2. Low-Level KV-Cache Management (Engine & Framework Level)

When hosting models directly via inference engines like **vLLM**, **TGI (Text Generation Inference)**, or **TensorRT-LLM**:

#### A. PagedAttention Configuration

Use PagedAttention (similar to OS virtual memory paging) to allocate memory for the KV-cache non-contiguously.

* **Block Size:** Set KV block size to 16 or 32 tokens.
* **Radix Tree Caching:** Enable automatic prefix sharing using a Radix Tree index over PagedAttention memory.

```python
# Example initialization using vLLM engine
from vllm import LLM, SamplingParams

llm = LLM(
    model="your-base-model-path",
    enable_prefix_caching=True,  # Enables RadixTree KV Cache sharing
    gpu_memory_utilization=0.90,  # Reserve high memory allocation for KV blocks
    max_num_batched_tokens=8192,
)

```

#### B. Cache Pinning & TTL Strategy

* **Explicit Pinning:** Maintain a warm instance of the system prefix KV-cache in VRAM during peak hours.
* **Sliding TTL (Time-To-Live):** Set an idle expiration timer (e.g., 600 seconds). If no request matches the `CacheKey`, flush the PagedAttention blocks back to system RAM or disk offload.

---

### 3. API-Level Implementation (Hosted Provider Layer)

When consuming hosted LLM APIs (e.g., Anthropic Claude, OpenAI, Google Gemini API):

#### A. Padding & Token Threshold Compliance

Most hosted APIs enforce a minimum token threshold before writing to cache storage.

* **Threshold Check:** Ensure the combined system instructions, tool definitions, and baseline examples exceed **1,024 tokens**.
* **Canonical Header Positioning:** Place all system prompts and system-level configuration blocks strictly at the very top of the prompt payload.

#### B. Explicit Caching Request Schema (Example implementation)

```json
{
  "model": "model-tier-name",
  "system": [
    {
      "type": "text",
      "text": "<MASSIVE_SYSTEM_INSTRUCTIONS_AND_TOOL_SCHEMAS>",
      "cache_control": { "type": "ephemeral" }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "Current user prompt..."
    }
  ]
}

```

---

## Performance & Optimization Metrics

To verify that the system is operating correctly, track the following key performance indicators (KPIs) in your telemetry dashboard:

| Metric | Target Goal | Monitoring Query / Formula |
| --- | --- | --- |
| **Cache Hit Ratio** | $\ge 85\%$ | $\frac{\text{Cache Hit Requests}}{\text{Total Inference Requests}}$ |
| **Time-To-First-Token (TTFT)** | $< 150\text{ ms}$ | Time from request receipt to generation of output token 1 |
| **Prefill Compute Reduction** | $\ge 80\%$ | $\frac{\text{Uncached Prefill FLOPs} - \text{Cached Prefill FLOPs}}{\text{Uncached Prefill FLOPs}}$ |
| **Memory Eviction Rate** | $< 5\%$ | Rate of premature VRAM block evictions due to memory pressure |

---

## Architectural Pitfalls to Avoid

1. **Dynamic Timestamp Injection in Static System Prompts:**
* *Anti-Pattern:* Placing dynamic timestamps like `Current time: 2026-11-07 18:53` inside the system prompt string.
* *Fix:* Move dynamic variables out of the system block into the mutable user message frame, preserving the static system hash.


2. **Unsorted Tool Schemas:**
* *Anti-Pattern:* Serializing JSON tool definitions in non-deterministic key orders.
* *Fix:* Pass JSON strings through a canonical serializer (`sort_keys=True`) before hashing and prefix loading.


3. **Inappropriate Chunk Granularity:**
* *Anti-Pattern:* Attempting to cache tiny prefixes ($< 256$ tokens). The memory management overhead outweighs the compute saved.
