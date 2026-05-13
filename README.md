# LLM-Inference-Serving-System

**CPU-first · Continuous Batching · Paged KV Cache · Speculative Decoding**

[![CI](https://github.com/Akshatha-22/LLM-Inference-Serving-System/actions/workflows/test.yml/badge.svg)(https://github.com/Akshatha-22/LLM-Inference-Serving-System/actions/workflows/test.yml)]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## One-Sentence Summary

> **A production-ready LLM serving system that serves multiple users concurrently on a CPU laptop, implementing continuous batching, paged KV cache, and speculative decoding — the same core algorithms powering vLLM and TensorRT-LLM.**

---

## Problem Statement

Large Language Models (LLMs) are slow and memory-hungry. Serving them to multiple users simultaneously is hard. Existing solutions assume NVIDIA GPUs with CUDA. This project solves the same problem **on CPU hardware**, proving algorithmic understanding without expensive GPUs.

**What this system does:**
| Question | Answer |
|----------|--------|
| **What problem?** | Serve LLMs to multiple users concurrently on limited hardware |
| **What's unique?** | Implements vLLM-style algorithms from scratch on CPU |
| **What scale?** | 20-30 concurrent users, 35-65 tok/sec on i5 laptop |
| **Key concepts?** | Continuous batching, PagedAttention, preemption, speculative decoding |
| **Hero Metrics?** | p99 latency, throughput, fragmentation, accept rate |
| **Tradeoffs?** | Throughput vs latency, fairness vs efficiency, RAM vs context |

**What you can learn from this codebase:**
- How vLLM's PagedAttention works under the hood
- How continuous batching differs from static batching
- When speculative decoding helps (and when it hurts)
- How to build production-ready inference systems

## Key Results
coming soon....

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i5-8250U (4 cores) | Intel i7-1165G7 (8 cores) |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB SSD |
| OS | Ubuntu 20.04+ / macOS 12+ / WSL2 | Same |
| GPU | ❌ Not required | ❌ Not required |


> ⚠️ **Honest constraint:** This is a CPU implementation. It demonstrates algorithms, not scale. The same architecture on GPU would serve 1000+ users at 100+ tok/s.

## Why This Project Exists
(-make it intentional and mature)

## Systems Concepts Implemented
Examples- Component |	Description
Continuous Batching, KV Cache Paging, Preemption, Speculative Decoding, ...


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


## Benchmark Section
comming soon....
will include:
CPU specs
RAM
model quantization
concurrency level
-Without hardware specs, benchmarks are meaningless.

## Engineering Challenges

## Tradeoffs Section
| **Decision** | **Trade off**|
|1.            |              |
| ....         |              |

## What I learned

## Honest Limitations

## Future Improvements
(to show scalability thinking)

##  Quick Start (5 minutes)

### Clone and setup

```bash
git clone https://github.com/yourusername/llm-inference-serving.git
cd llm-inference-serving
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Repository Structure

## Screenshots Section




