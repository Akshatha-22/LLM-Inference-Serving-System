## Failure log
 
### Entry 1 — Day 1: False baseline (attention_mask bug)
 
| Field | Detail |
|---|---|
| Date | 2026-05-15 |
| What happened | Baseline reported 20.60 tok/s — unrealistically high for i5 CPU |
| How I found it | Expected range is 8–12 tok/s for this hardware; 20.60 far exceeded it |
| Root cause | GPT-2 pad token = eos token, causing incorrect attention_mask inference |
| Fix applied | Corrected mask handling; result dropped to honest 11.49 tok/s |
| Time to fix | ~2 hours |
| What I learned | Always sanity-check performance numbers against hardware specs. A number that looks too good is a bug, not a win. |
 
---
