# Background

## SVD of the score matrix $S = QK^\top / \sqrt{d}$

This is the best case, so I'll start here. Per head and layer you have:

- $Q \in \mathbb{R}^{L \times d}$
- $K \in \mathbb{R}^{L \times d}$
- $S = QK^\top \in \mathbb{R}^{L \times L}$

You want the top 1–2 singular values/vectors of $S$, but never build $S$.

## Matrix-free multiplies

Crucial observation: for any vector $v \in \mathbb{R}^L$,

$$Sv = Q(K^\top v), \quad S^\top u = K(Q^\top u)$$

So:

1. Compute $z = K^\top v \in \mathbb{R}^d$ — cost $O(Ld)$
2. Compute $w = Qz \in \mathbb{R}^L$ — cost $O(Ld)$

You've applied $S$ to $v$ without ever forming $S$. Same for $S^\top u$. This is exactly what you need for:

- Power iteration on singular vectors (via $S^\top S$ or $SS^\top$), or
- Lanczos / randomized SVD.

All of that stays linear in $Ld$ per iteration (with $d \ll L$), versus $O(L^2)$ if you formed the matrix.

## How to actually do it (sketch)

Say you want the top 2 singular values/vectors per head:

1. Extract $Q_h, K_h \in \mathbb{R}^{L \times d}$ for that head at the snapshot point.
   In vLLM, this means either:
   - You've instrumented `Attention.forward` and stashed Q/K somewhere, or
   - You recompute the layer once in a side pass with standard PyTorch to get Q, K.

2. Implement a tiny "matvec" helper on GPU:

   ```python
   def matvec_S(v, Q, K):
       # v: [L], Q,K: [L, d]
       z = K.T @ v          # [d]
       return Q @ z         # [L]

   def matvec_St(u, Q, K):
       z = Q.T @ u          # [d]
       return K @ z         # [L]
   ```

   (You'd make these batched and head-aware in real code.)

3. Plug these into any matrix-free SVD routine, e.g. block power method / Lanczos:
   - Start with random vectors $v_1, \ldots, v_k$.
   - Iteratively apply $S$ and $S^\top$ with re-orthonormalization.
   - After a small number of iterations, you have approximate top singular pairs.

You never build an $L \times L$ tensor; you only keep:

- A few vectors in $\mathbb{R}^L$ (for the current Krylov subspace),
- $Q$ and $K$ ($L \times d$ each, which you already have if you're doing attention).

## Tying this "into the streaming path"

Conceptually:

1. At a snapshot step (e.g. end of generation, or every $n$ tokens):
   - Construct $Q$ and $K$ for the current sequence length $L$ for the layers/heads you care about.
   - Run a few iterations of matrix-free SVD on each such pair $(Q, K)$.
   - Use those singular values/vectors in your safety computation.

2. Where do you get Q/K?
   - In vLLM you don't keep historical Qs by default — only KV are cached.
   - So for a snapshot, you'd either:
     - Re-run the model in a cheap side pass for that sequence (just for those layers) to recover all Q/K; or
     - Modify the attention layer to additionally cache Q per token (doubling KV-like memory for those layers).

But once you have Q/K, the SVD itself can be done entirely without $L \times L$ matrices and with only $O(Ld)$ overhead per matvec.


# Implementation Details
Plan: Custom vLLM Attention Backend for Matrix-Free SVD

 Context

 During vLLM decode, the attention layer only sees Q for the new token (shape [1, heads, d]). The full K lives in the paged KV cache but is never exposed as a contiguous
 tensor. The user's prior approach of instrumenting the torch.compile graph only captured [1x1] score matrices — useless for SVD.

 The goal is to create a custom attention backend that wraps the existing Triton backend, accumulates Q tokens over decode steps, extracts K from the paged cache, and runs
 matrix-free SVD at configurable intervals — all without modifying vLLM source.

 Approach

 Create two new files in the whitebox project. No vLLM source files are modified.

 File 1: glassbox/svd_backend.py

 A thin wrapper around vLLM's TritonAttentionBackend / TritonAttentionImpl.

 Registration — Use the @register_backend(AttentionBackendEnum.CUSTOM) decorator from vllm.v1.attention.backends.registry (line 205-257). This populates the override dict so
 CUSTOM.get_class() resolves to our backend. No vLLM enum changes needed.

 SVDTritonAttentionBackend — Subclass TritonAttentionBackend. Override only:
 - get_name() → "CUSTOM"
 - get_impl_cls() → SVDTritonAttentionImpl

 Everything else (metadata builder, KV cache shape, etc.) is inherited unchanged.

 SVDTritonAttentionImpl — Subclass TritonAttentionImpl. Override forward():

 1. Call super().forward(...) for normal attention (unchanged Triton kernel path).
 2. Skip SVD logic during prefill (max_query_len > 1) and profiling (attn_metadata is None).
 3. During decode: accumulate query[0:1] (first sequence in batch) into a per-layer Q buffer.
 4. Every SVD_INTERVAL steps: extract K from the paged cache and run matrix-free SVD.

 Q accumulation — A module-level dict[str, PerLayerSVDState] keyed by layer.layer_name (e.g. "model.layers.0.self_attn"). Each entry holds a list[Tensor] of shape-[1,
 num_heads, head_size] clones.

 K extraction from paged cache — _extract_k_from_cache() method:
 - kv_cache shape: [num_blocks, 2, block_size, num_kv_heads, head_size]
 - Unbind dim=1 to get key_cache: [num_blocks, block_size, num_kv_heads, head_size]
 - Use attn_metadata.block_table[seq_idx] to get physical block indices for the sequence
 - Use attn_metadata.seq_lens[seq_idx] for actual sequence length
 - Index key_cache[block_indices], reshape, and trim to [seq_len, num_kv_heads, head_size]

 SVD call — _run_svd() method:
 - torch.cat the Q buffer → [L_q, num_heads, head_size]
 - For each configured head, extract Qh [L_q, d] and Kh [L_k, d]
 - Truncate to min(L_q, L_k) (Q buffer misses prefill tokens; follow-up to fix)
 - Call matvec_S / matvec_ST from glassbox.svd as the operator
 - Call randomized_svd or svd_via_lanczos from glassbox.svd
 - Log singular values via Python logging

 Configuration via env vars:
 - WHITEBOX_SVD_INTERVAL — run SVD every N decode steps (default: 32)
 - WHITEBOX_SVD_RANK — number of singular values (default: 4)
 - WHITEBOX_SVD_METHOD — "randomized" or "lanczos" (default: "randomized")
 - WHITEBOX_SVD_HEADS — comma-separated head indices (default: "0")

 File 2: glassbox/svd_backend_runner.py

 Entry-point script that:
 1. Imports glassbox.svd_backend (triggers @register_backend)
 2. Creates vllm.LLM(model=..., attention_backend="CUSTOM", enforce_eager=True)
 3. Runs a simple generation to verify SVD output in logs

 

 Known limitations (prototype):
 - enforce_eager=True disables CUDA graphs (our SVD code has dynamic shapes).
 - First-sequence-only: We track Q for batch index 0. vLLM may reorder sequences across steps, so Q tokens could mix. Proper fix needs sequence-ID tracking.
 - FP8 cache: If KV cache is FP8, the extracted K needs conversion to float before SVD. We'll add a .float() cast.

 Verification

```bash
 $ cd /home/ubuntu/src/whitebox
 $ GLASSBOX_SVD_INTERVAL=16 GLASSBOX_SVD_RANK=2 python -m glassbox.svd_backend_runner
```

 Expected: normal generation output, plus log lines like:
 [SVD] model.layers.0.self_attn head=0 step=16 L=22 top-2 singular values: [166.75, 84.70]

 ## Sample Run

 Run details:

 - Model: `facebook/opt-125m` (12 decoder layers, 12 attention heads, d=64)
 - Prompt: "The future of artificial intelligence is" (7 tokens)
 - Generated: "clouding the world. A report from Canalys found that artificial intelligence will be more popular than ever in 2019. The report said the world's most advanced artificial intelligence is expected to be used for more than 90 per cent of all applications – from applications like consumer research to law enforcement."
 - Q is accumulated from both prefill and decode, so L = prompt_len + decode_steps - 1
 - SVD snapshots at steps 16, 32, 48, 64 (L = 22, 38, 54, 70)

 Cleaned up Output:

 ```
 $ GLASSBOX_SVD_INTERVAL=16 GLASSBOX_SVD_RANK=2 /opt/pytorch/bin/python -m glassbox.svd_backend_runner

 step=16  L=22  (7 prompt + 15 decode)
 [SVD] layer.0  head=0 [166.76, 84.71]    σ₁/σ₂ = 1.97
 [SVD] layer.1  head=0 [584.30, 41.42]    σ₁/σ₂ = 14.1
 [SVD] layer.2  head=0 [201.66, 169.63]   σ₁/σ₂ = 1.19
 [SVD] layer.3  head=0 [816.10, 66.66]    σ₁/σ₂ = 12.2
 [SVD] layer.4  head=0 [396.88, 91.63]    σ₁/σ₂ = 4.33
 [SVD] layer.5  head=0 [692.16, 134.62]   σ₁/σ₂ = 5.14
 [SVD] layer.6  head=0 [694.17, 139.38]   σ₁/σ₂ = 4.98
 [SVD] layer.7  head=0 [277.48, 227.12]   σ₁/σ₂ = 1.22
 [SVD] layer.8  head=0 [942.26, 259.02]   σ₁/σ₂ = 3.64
 [SVD] layer.9  head=0 [413.35, 94.84]    σ₁/σ₂ = 4.36
 [SVD] layer.10 head=0 [495.52, 146.97]   σ₁/σ₂ = 3.37
 [SVD] layer.11 head=0 [430.89, 170.45]   σ₁/σ₂ = 2.53

 step=32  L=38  (7 prompt + 31 decode)
 [SVD] layer.0  head=0 [255.46, 147.19]   σ₁/σ₂ = 1.74
 [SVD] layer.1  head=0 [1009.70, 77.87]   σ₁/σ₂ = 13.0
 [SVD] layer.2  head=0 [545.46, 314.40]   σ₁/σ₂ = 1.73
 [SVD] layer.3  head=0 [1562.25, 206.03]  σ₁/σ₂ = 7.58
 [SVD] layer.4  head=0 [782.33, 173.44]   σ₁/σ₂ = 4.51
 [SVD] layer.5  head=0 [1280.43, 247.90]  σ₁/σ₂ = 5.16
 [SVD] layer.6  head=0 [1416.93, 256.01]  σ₁/σ₂ = 5.53
 [SVD] layer.7  head=0 [457.38, 311.17]   σ₁/σ₂ = 1.47
 [SVD] layer.8  head=0 [1448.77, 370.08]  σ₁/σ₂ = 3.91
 [SVD] layer.9  head=0 [711.01, 146.42]   σ₁/σ₂ = 4.86
 [SVD] layer.10 head=0 [864.10, 210.71]   σ₁/σ₂ = 4.10
 [SVD] layer.11 head=0 [770.47, 230.35]   σ₁/σ₂ = 3.34

 step=48  L=54  (7 prompt + 47 decode)
 [SVD] layer.0  head=0 [417.50, 194.91]   σ₁/σ₂ = 2.14
 [SVD] layer.1  head=0 [1504.97, 106.41]  σ₁/σ₂ = 14.1
 [SVD] layer.2  head=0 [897.47, 457.01]   σ₁/σ₂ = 1.96
 [SVD] layer.3  head=0 [2358.72, 374.79]  σ₁/σ₂ = 6.29
 [SVD] layer.4  head=0 [1280.71, 280.75]  σ₁/σ₂ = 4.56
 [SVD] layer.5  head=0 [1872.36, 401.01]  σ₁/σ₂ = 4.67
 [SVD] layer.6  head=0 [2076.45, 383.92]  σ₁/σ₂ = 5.41
 [SVD] layer.7  head=0 [643.82, 405.51]   σ₁/σ₂ = 1.59
 [SVD] layer.8  head=0 [1848.62, 410.52]  σ₁/σ₂ = 4.50
 [SVD] layer.9  head=0 [1011.05, 193.67]  σ₁/σ₂ = 5.22
 [SVD] layer.10 head=0 [1283.76, 265.81]  σ₁/σ₂ = 4.83
 [SVD] layer.11 head=0 [1099.57, 320.44]  σ₁/σ₂ = 3.43

 step=64  L=70  (7 prompt + 63 decode)
 [SVD] layer.0  head=0 [605.55, 268.25]   σ₁/σ₂ = 2.26
 [SVD] layer.1  head=0 [1972.43, 139.81]  σ₁/σ₂ = 14.1
 [SVD] layer.2  head=0 [1208.11, 626.74]  σ₁/σ₂ = 1.93
 [SVD] layer.3  head=0 [3167.29, 619.72]  σ₁/σ₂ = 5.11
 [SVD] layer.4  head=0 [1755.43, 392.19]  σ₁/σ₂ = 4.48
 [SVD] layer.5  head=0 [2657.34, 548.44]  σ₁/σ₂ = 4.84
 [SVD] layer.6  head=0 [2967.04, 603.63]  σ₁/σ₂ = 4.92
 [SVD] layer.7  head=0 [941.70, 454.27]   σ₁/σ₂ = 2.07
 [SVD] layer.8  head=0 [2119.39, 609.10]  σ₁/σ₂ = 3.48
 [SVD] layer.9  head=0 [1365.53, 224.39]  σ₁/σ₂ = 6.09
 [SVD] layer.10 head=0 [1702.90, 348.14]  σ₁/σ₂ = 4.89
 [SVD] layer.11 head=0 [1506.31, 397.88]  σ₁/σ₂ = 3.78
 ```

Interpretation:

 These are the top-2 singular values of S = QK^T (the raw scores matrix, before softmax), for head 0 of each layer. The σ₁/σ₂ ratio reveals attention sharpness: high ratio means the scores matrix is nearly rank-1 (all queries project onto the same key direction); low ratio means multiple independent attention patterns are active.

 Ratio trajectory over time reveals layer behavior:

 | Layer | step 16 | step 32 | step 48 | step 64 | Behavior |
 |-------|---------|---------|---------|---------|----------|
 | 1     | 14.1    | 13.0    | 14.1    | 14.1    | Constant — content-independent (likely positional) |
 | 3     | 12.2    | 7.6     | 6.3     | 5.1     | Decaying — attention spreads as context diversifies |
 | 7     | 1.22    | 1.47    | 1.59    | 2.07    | Increasing — one direction consolidates over time |

 - **Layer 1** has a rock-stable ratio of ~14 regardless of sequence length. This is the signature of a fixed structural pattern — almost certainly positional attention (e.g. always attend to BOS). The generated content doesn't affect it.
 - **Layer 3** starts very sharp (12x) but steadily decays to 5x. As the generated text introduces diverse entities ("Canalys", "2019", "consumer research", "law enforcement"), the dominant Q-K direction can't capture everything — a second direction gains weight.
 - **Layer 7** starts nearly isotropic (1.2x) but slowly sharpens to 2x. As the generation settles into a repetitive theme (repeated mentions of "artificial intelligence", "report"), one attention direction begins to dominate where none did before.
 - **Middle layers (4-6)** have stable ratios around 4.5-5.5x — moderately sharp and content-insensitive.
 - **Raw σ₁ values scale linearly with L** (layer 3: 816 → 1562 → 2359 → 3167, roughly ∝ L). This is expected since more entries in QK^T contribute to the Frobenius norm. The ratio is the more informative quantity.
 