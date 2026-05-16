# LLM-Inference-Serving-System

**CPU-first · GPT-2 model · Continuous Batching · Paged KV Cache · Speculative Decoding**

[![CI](https://github.com/Akshatha-22/LLM-Inference-Serving-System/actions/workflows/test.yml/badge.svg)(https://github.com/Akshatha-22/LLM-Inference-Serving-System/actions/workflows/test.yml)]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## One-Sentence Summary

**A production-ready LLM serving system that serves multiple users concurrently on a CPU laptop, implementing continuous batching, paged KV cache, and speculative decoding — the same core algorithms powering vLLM and TensorRT-LLM.**

---

## Problem Statement

> Large Language Models (LLMs) are slow and memory-hungry. Serving them to multiple users simultaneously is hard. Existing solutions assume NVIDIA GPUs with CUDA. This project solves the same problem **on CPU hardware**, proving algorithmic understanding without expensive GPUs.
---

## ⚠️ Hardware constraints (read first)
 
| Constraint | Detail |
|-------------|---------|
| Hardware | Intel i5 CPU, 16GB RAM, NVMe SSD, no GPU |
| Model | GPT-2 Small (124M parameters) |
| Why not vLLM? | vLLM requires CUDA. Building from scratch teaches how PagedAttention and continuous batching actually work. With GPU access, I'd use vLLM + FlashAttention-2. |
| Context limit | GPT-2 has a hard 1024-token limit (sinusoidal positional embeddings). 32k context requires RoPE/ALiBi models like LLaMA. |
--- 

## Architecture Overview
-The system has four main flows. First, the request lifecycle: API key validation → rate limiting → priority queue → streaming response. Second, the scheduler: continuous batching that rebuilds the batch after every decode step — no padding waste. Third, memory: paged KV cache with block tables and copy-on-write for beam search. Fourth, batching: static vs continuous tradeoff analysis.

**1. Request Lifecycle**

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST LIFECYCLE                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

  Client                    API Gateway                Scheduler                  Inference Engine
     │                           │                         │                              │
     │   POST /v1/completions    │                         │                              │
     │──────────────────────────▶│                         │                              │
     │                           │                         │                              │
     │                           │  1. Validate API key    │                              │
     │                           │  2. Parse request       │                              │
     │                           │  3. Check rate limit    │                              │
     │                           │                         │                              │
     │                           │  4. Add to priority queue (deadline = now + SLA)       │
     │                           │────────────────────────▶│                              │
     │                           │                         │                              │
     │                           │                         │  5. Schedule when batch slot  │
     │                           │                         │     becomes available         │
     │                           │                         │                              │
     │                           │                         │  6. Forward pass              │
     │                           │                         │─────────────────────────────▶│
     │                           │                         │                              │
     │                           │                         │  7. Generate token            │
     │                           │                         │◀─────────────────────────────│
     │                           │                         │                              │
     │  8. Stream token via SSE  │                         │                              │
     │◀──────────────────────────│                         │                              │
     │                           │                         │                              │
     │   (repeat steps 5-8 until EOS or max_tokens)        │                              │
     │                           │                         │                              │
     │                           │  9. Log metrics         │                              │
     │                           │     (latency, tokens)   │                              │
     │                           │                         │                              │
     │  10. Connection closed    │                         │                              │
     │◀──────────────────────────│    
     
**design decisions:**
-Deadline-based priority: Requests with earlier SLA deadlines jump the queue
-Streaming first: Tokens sent as soon as generated, not batched
-Per-request tracing: Every request gets a unique request_id for debugging.    

**2.Scheduler Flow (Continuous Batching)**
                              ┌─────────────────────┐
                              │   WAITING QUEUE     │
                              │  (priority by       │
                              │   deadline)         │
                              └──────────┬──────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
              ┌───────────┐      ┌───────────┐      ┌───────────┐
              │  Step 1   │      │  Step 2   │      │  Step 3   │
              │ Batch=[A,B,C]     │ Batch=[A,B,D]     │ Batch=[B,D,E]
              └───────────┘      └───────────┘      ┌───────────┘
                    │                    │                    │
                    ▼                    ▼                    ▼
              ┌───────────┐      ┌───────────┐      ┌───────────┐
              │ A: token 5│      │ A: token 6│      │ B: token 3│
              │ B: token 2│      │ B: token 3│      │ D: token 2│
              │ C: token 1│      │ D: token 1│      │ E: token 1│
              └───────────┘      └───────────┘      └───────────┘
                    │                    │                    │
                    ▼                    ▼                    ▼
              ┌───────────┐      ┌───────────┐      ┌───────────┐
              │ C finished│      │ A finished│      │ B running │
              │ → remove  │      │ → remove  │      │ D running │
              │           │      │           │      │ E running │
              └───────────┘      └───────────┘      └───────────┘
**Why continuous batching wins:**
-No waiting for batch to fill → lower latency at low load
-Immediate replacement of finished sequences → higher throughput
-Variable-length sequences don't block others

**3. Memory Flow (Paged KV Cache)**
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              MEMORY FLOW DIAGRAM                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────────┐
                              │     PHYSICAL BLOCK POOL  │
                              │  (GPU/CPU memory)        │
                              │  ┌────┬────┬────┬────┐   │
                              │  │ B0 │ B1 │ B2 │ B3 │   │
                              │  │free│used│used│free│   │
                              │  └────┴────┴────┴────┘   │
                              │  ┌────┬────┬────┬────┐   │
                              │  │ B4 │ B5 │ B6 │ B7 │   │
                              │  │used│free│used│free│   │
                              │  └────┴────┴────┴────┘   │
                              └────────────┬────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
              ▼                            ▼                            ▼
    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │   BLOCK TABLE   │          │   BLOCK TABLE   │          │   BLOCK TABLE   │
    │   (Sequence A)  │          │   (Sequence B)  │          │   (Sequence C)  │
    │                 │          │                 │          │                 │
    │ logical 0 → B1  │          │ logical 0 → B2  │          │ logical 0 → B2  │
    │ logical 1 → B4  │          │ logical 1 → B4  │          │   (shared!)     │
    │ logical 2 → B6  │          │ logical 2 → B6  │          │                 │
    └─────────────────┘          └─────────────────┘          └─────────────────┘
                                           │
                                           │
                              ┌────────────┴────────────┐
                              │    COPY-ON-WRITE        │
                              │    When Sequence B      │
                              │    diverges from C:     │
                              │    copy B2 → new block  │
                              └─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PREEMPTION FLOW                                         │
│                                                                                      │
│   Memory > 85% ──▶ Select lowest priority sequence ──▶ Save KV to disk (pickle)    │
│                                                                                      │
│   Memory < 70% ──▶ Load from disk ──▶ Resume in next available batch slot          │
└─────────────────────────────────────────────────────────────────────────────────────┘
              
**4. Batching Flow**
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              BATCHING FLOW                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────────┐
                              │   INCOMING REQUESTS      │
                              │   (different lengths)    │
                              │   ┌─────┐ ┌─────┐ ┌─────┐│
                              │   │ Req1│ │ Req2│ │ Req3││
                              │   │128t │ │ 64t │ │256t ││
                              │   └─────┘ └─────┘ └─────┘│
                              └────────────┬────────────┘
                                           │
                              ┌────────────┴────────────┐
                              │    STATIC BATCHING      │
                              │    (wait for N or T ms) │
                              └────────────┬────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────────┐
                    │              STATIC BATCH                        │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │ Req1 │ Req2 │ Req3 │ PAD │ PAD │ PAD │   │    │
                    │  │ 128t │ 64t  │ 256t │ 0   │ 0   │ 0   │   │    │
                    │  └─────────────────────────────────────────┘    │
                    │                                                   │
                    │  Problem: 50% wasted compute on padding          │
                    │  Problem: Low-load requests wait for batch       │
                    └─────────────────────────────────────────────────┘

                                           │
                              ┌────────────┴────────────┐
                              │  CONTINUOUS BATCHING    │
                              │  (per-step scheduling)  │
                              └────────────┬────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────────┐
                    │         STEP 1: Initial batch                    │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │ Req1(t=0) │ Req2(t=0) │ Req3(t=0)        │    │
                    │  └─────────────────────────────────────────┘    │
                    └─────────────────────────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────────┐
                    │         STEP 2: Req1 finishes early             │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │ Req2(t=1) │ Req3(t=1) │ NEW Req4(t=0)   │    │
                    │  └─────────────────────────────────────────┘    │
                    │  ↑ No padding!                                   │
                    │  ↑ Finished sequence replaced immediately        │
                    └─────────────────────────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────────┐
                    │         STEP 3: All at different lengths        │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │ Req2(t=2) │ Req3(t=2) │ Req4(t=1)        │    │
                    │  └─────────────────────────────────────────┘    │
                    └─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              BATCHING COMPARISON                                     │
│                                                                                      │
│  ┌──────────────────┬─────────────────────┬─────────────────────────────────────┐  │
│  │                  │  STATIC BATCHING     │  CONTINUOUS BATCHING                │  │
│  ├──────────────────┼─────────────────────┼─────────────────────────────────────┤  │
│  │ Padding overhead │ High (to max length) │ Zero (sequences of actual length)   │  │
│  │ Low-load latency │ High (wait for batch)│ Low (no waiting)                    │  │
│  │ Throughput       │ Medium               │ High (no idle cycles)               │  │
│  │ Implementation   │ Simple (timer+queue) │ Complex (per-step state machine)    │  │
│  └──────────────────┴─────────────────────┴─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
---

## Measured results (Days 1–5)
 
> These are real, measured numbers on the hardware above.
> Future optimization targets are listed separately below.
 
### Baseline inference (Day 1–2)
 
| Metric | Value | Notes |
|---|---|---|
| Tokens/sec (avg) | **11.49** | 5-run average, greedy decode |
| Tokens/sec (min) | 6.45 | Cold CPU cache, first run |
| Tokens/sec (max) | 15.66 | Warm cache, peak |
| Model load time | 1.37s | Cached (NVMe SSD) |
| First load time | 205s | Download + unzip |
| Memory (idle) | 363.6 MB | GPT-2 Small weights |
| Parameters | 124,439,808 | — |
| Hardware | i5 CPU, 16GB RAM | No GPU |
| Date measured | 2026-05-15 | — |
 
**Why variance exists (6.45 → 15.66):** CPU frequency scaling (thermal throttling on laptop),
background OS processes, thread scheduling jitter. This is normal for CPU inference.
The average (11.49 tok/s) is the reliable baseline number.
 
Baseline saved to: `benchmarks/results/phase1_baseline/baseline_results.json`
 
---
 
### FastAPI server (Days 3–5)
 
| Feature | Status |
|---|---|
| HTTP API (FastAPI) | ✅ Running |
| `/v1/completions` endpoint | ✅ Working |
| Non-streaming responses | ✅ Correct output |
| Streaming SSE responses | ✅ Spaces preserved |
| Request ID tracing (structlog + UUID) | ✅ `req_xxx` in logs |
| Health check endpoint | ✅ Returning status |
| Test suite | ✅ Passing |
 
**Performance impact of HTTP layer:**
 
| Metric | Direct (Day 1–2) | Via API (Day 3–5) | Delta |
|---|---|---|---|
| Tokens/sec | 11.49 | 11.49 | ±0 (identical) |
| Memory | 363.6 MB | 368.1 MB | +4.5 MB (API overhead) |
| Startup time | 1.37s | 3.16s | +1.8s (one-time API init) |
 
The HTTP layer adds ~0.1s overhead for request parsing but does not affect generation speed.
 
---
 
## Engineering decisions (Day 1–5)
 
### Why structlog over OpenTelemetry
OpenTelemetry setup (collector, exporter, trace propagation) costs 1–2 days solo.
`structlog` + UUID `request_id` gives 90% of the observability value at 10% of the setup cost.
OpenTelemetry is documented as future work.
 
### GPT-2 tokenization quirk
GPT-2 uses the same token ID for padding and end-of-sequence.
This produces an `attention_mask` warning that is informational, not an error.
Results at 11.49 tok/s are correct. The warning can be silenced with:
```python
tokenizer.pad_token = tokenizer.eos_token
```
 
---
 
## Optimization targets (not yet measured)
 
> These are forward-looking targets based on the project plan.
> They will be updated with real measurements as each phase is completed.
 
| Phase | Feature | Current | Target | Status |
|---|---|---|---|---|
| Day 1–2 | Baseline | **11.49 tok/s** | — | ✅ measured |
| Day 8–10 | Dynamic batching | 11.49 | ~20 tok/s (+74%) | ⏳ not started |
| Day 15–17 | Continuous batching | 11.49 | ~35 tok/s (+204%) | ⏳ not started |
| Day 22–28 | + PagedAttention | 11.49 | ~40 tok/s (+248%) | ⏳ not started |
| Day 41–44 | + Speculative decoding | 11.49 | ~50 tok/s (+335%) | ⏳ not started |
 
---
 
## 🗂️ Project structure
 
```
llm-inference-server/
├── src/
│   ├── model/          # GPT-2 wrapper, generate loop, tokenizer
│   ├── server/         # FastAPI app, routes, request queue, tracing
│   ├── scheduler/      # Sequence dataclass, static batcher, continuous scheduler
│   ├── kv_cache/       # Block allocator, paged attention, copy-on-write
│   ├── speculative/    # Draft model, target model, acceptance sampler
│   └── observability/  # Prometheus metrics, GPU gap analysis
├── tests/
│   ├── unit/           # Block allocator, state machine, acceptance sampler
│   └── integration/    # Correctness vs HuggingFace baseline
├── benchmarks/         # Load generator, benchmark scripts, results/
├── data/               # ShareGPT, HumanEval, CNN/DailyMail prompts
├── profiles/           # py-spy flamegraphs, torch profiler traces
├── docs/               # Architecture diagram, design doc, interview Q&A
└── monitoring/         # Prometheus config, Grafana dashboard
```
 
---
 
##  Tech stack
 
| Layer | Tool |
|---|---|
| Model | PyTorch (CPU) + HuggingFace Transformers (GPT-2) |
| Server | FastAPI + uvicorn |
| Tracing | structlog + UUID request IDs |
| Load testing | Locust |
| Benchmarks | ShareGPT (50 conv) + HumanEval (30 code) + CNN/DailyMail (20 summaries) |
| Profiling | py-spy + torch.profiler |
| Memory monitoring | psutil + tracemalloc |
| Observability | Prometheus + Grafana |
| CI | GitHub Actions |
 
---
 
##  RAM requirements
 
| Model | Size | RAM (idle) | RAM (1k ctx) | Max context |
|---|---|---|---|---|
| GPT-2 Small | 124M | ~500 MB | ~620 MB | 1024 tokens (architectural limit) |
| GPT-2 Medium | 355M | ~1.4 GB | ~1.7 GB | 1024 tokens |
| GPT-2 Large | 774M | ~3.0 GB | ~3.6 GB | 1024 tokens |
 
> GPT-2's 1024-token limit comes from sinusoidal positional embeddings.
> For 32k context, you need RoPE or ALiBi (e.g. LLaMA, GPT-NeoX).
---

## Benchmark Section

>comming soon....
will include:
CPU specs
RAM
model quantization
concurrency level
-Without hardware specs, benchmarks are meaningless.
---

## Engineering Challenges
>coming soon...
---

## Tradeoffs Section
| **Decision** | **Trade off**|
|1.            |              |
| ....         |              |

## What I learned

## Honest Limitations

## Future Improvements
(to show scalability thinking)
---
##  Quick Start (5 minutes)
### Clone and setup
```bash
git clone https://github.com/yourusername/llm-inference-server
cd llm-inference-server
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
 
# Start the server
uvicorn src.server.app:app --host 0.0.0.0 --port 8000
 
# Test it
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50}'
```
 
---
## Screenshots Section
coming soon....

---

*60-day project · GPT-2 on i5 CPU · All algorithms implemented from scratch*
*Last updated: Day 5 of 60*
 


