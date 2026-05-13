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
- Takes HTTP requests from multiple users
- Batches them intelligently for maximum throughput
- Manages memory with paged KV cache (no wasted space)
- Accelerates generation with speculative decoding
- Survives memory pressure via preemption (swap to disk)

**What you can learn from this codebase:**
- How vLLM's PagedAttention works under the hood
- How continuous batching differs from static batching
- When speculative decoding helps (and when it hurts)
- How to build production-ready inference systems

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i5-8250U (4 cores) | Intel i7-1165G7 (8 cores) |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB SSD |
| OS | Ubuntu 20.04+ / macOS 12+ / WSL2 | Same |
| GPU | ❌ Not required | ❌ Not required |

**Realistic Performance on i5-1135G7 (16GB RAM):**

| Metric | Value |
|--------|-------|
| Max concurrent users | 20-25 (saturation point) |
| Throughput (tokens/sec) | 15-18 tok/s aggregate |
| P99 latency (10 users) | ~800ms |
| P99 latency (20 users) | ~1.8s |
| Model size supported | 1B-3B (quantized Q4) |

> ⚠️ **Honest constraint:** This is a CPU implementation. It demonstrates algorithms, not scale. The same architecture on GPU would serve 1000+ users at 100+ tok/s.

## Architecture Overview




