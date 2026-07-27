# R3 comparison caveat — decide before reading the numbers

The harness's concurrency dimension issues C "decisions" that all use the SAME
stable prefix. So at C>1 every request in the run shares one fusion key:

  C=1, N=16  -> 16 requests, 1 key   -> gateway fuses to ONE n=16 call
               vs R2's one n=16 call.        <-- APPLES TO APPLES
  C=4, N=16  -> 64 requests, 1 key   -> gateway fuses to 2x n=32 (max_batch=32)
               vs R2's FOUR separate n=16 calls.  <-- NOT apples to apples

At C>1 the gateway gets to fuse ACROSS decisions because the harness gave them
identical prompts. A real agent's 4 concurrent decisions would carry different
prompts and would NOT merge. So a favourable R3-vs-R2 at C>1 partly measures an
artifact of the harness, not the product.

CONSEQUENCE (chosen before seeing data):
  - Report C=1 as the headline R3-vs-R2 number. It is the honest one.
  - Report C>1 rows with this caveat stated inline, never as the headline.
  - R3-vs-R1 is unaffected: both fan out the same way, only one is fused.
