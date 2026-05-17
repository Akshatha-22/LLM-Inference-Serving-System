## Performance Baseline (Day 6-7)

**Hardware:** Intel i5 CPU, 16GB RAM, no GPU
**Model:** GPT-2 Small (124M)
**Test duration:** 30 seconds

| Users | Avg Latency | P99 Latency | Throughput | Status |
|-------|-------------|-------------|------------|--------|
| 1 | 3.3s | 3.4s | 0.3 req/s | ✅ Acceptable |
| 5 | 5.6s | 10.0s | 0.5 req/s | 🟡 Degraded |
| 10 | 8.1s | 13.0s | 0.7 req/s | 🔴 Saturated |

**Saturation point:** 5-7 concurrent users
**Next step:** Dynamic batching (Day 8-10) to increase capacity to 15-20 users
