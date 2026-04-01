# Memory-Constrained Exact Attention

A from-scratch implementation of memory-efficient attention in PyTorch, using tiled computation and online softmax to avoid materialising the full n×n attention matrix. Produces mathematically identical results to standard attention while reducing memory from O(n²) to O(1) relative to sequence length.

---

## The Problem

Standard attention computes a score between every pair of tokens, producing an n×n matrix:

```
n = 1,000   →   1,000,000 values   (~4 MB)
n = 10,000  →   100,000,000 values (~400 MB per layer)
n = 100,000 →   10,000,000,000 values (impossible)
```

Memory scales quadratically with sequence length. For long contexts across many layers, this simply does not fit on real hardware.

---

## The Solution

Two key ideas:

**1. Online Softmax**
Softmax normally requires all scores to be visible before any can be normalised. Online softmax breaks this by maintaining a running maximum `m` and running sum `l`, updating them as new blocks of scores arrive. When a larger maximum is found, the old sum is rescaled by `exp(m_old - m_new)`. The result is mathematically exact — nothing is approximated.

**2. Tiled Computation**
The sequence is processed in small blocks. For each query block, the inner loop iterates over all key/value blocks, computing only a small `(block_size x block_size)` score matrix at a time and immediately discarding it after updating the running statistics. Peak memory becomes `O(block_size x d)` regardless of n.

---

## Implementation

```
Round2_Code_YourName.py
├── standard_attention()          # naive O(n²) baseline
├── tiled_attention_forward()     # chunked forward pass with online softmax
├── tiled_attention_backward()    # gradient checkpointing backward pass
└── run_verification()            # correctness check + memory comparison
```

### Forward Pass

The tiled forward pass maintains three running values per query token:
- `m` — running maximum score seen so far
- `l` — running normalisation sum (rescaled when m updates)
- `O` — running weighted output (rescaled when m updates)

After all key/value blocks are processed, the final output is `O / l`.

### Backward Pass

Standard backprop would store the full attention weight matrix A during the forward pass for use in the backward pass — O(n²) memory again. Instead:

- During the forward pass, A is discarded. Only `m` and `l` are saved (2 values per token, O(n) total).
- During the backward pass, A is recomputed block by block on the fly using the saved statistics.

The softmax gradient requires a correction term `D[i] = sum over all j of A[i,j] * dA[i,j]` that must cover all key positions. This requires two separate passes: the first accumulates `D` completely, the second uses it to compute gradients.

---

## Results

Verified on n=128, d=32, block_size=32:

```
[Forward Pass]
  Max difference from standard attention: ~1e-06
  Outputs match!

[Backward Pass]
  Max gradient difference: ~1e-05
  Gradients match!

[Memory Analysis]
  Standard attention matrix:  0.1250 MB  (n=128)
  Tiled attention block:      0.0078 MB  (block=32)
  Standard backward mem:      0.0625 MB
  Tiled backward stat mem:    0.0010 MB
  Reduction factor:           64x

[Scaling]
         n        Standard           Tiled
       128          0.13MB        0.0078MB
       512          2.00MB        0.0078MB
      1024          8.00MB        0.0078MB
      4096        128.00MB        0.0078MB
     10000        762.94MB        0.0078MB
```

The tiled block memory stays flat at 0.0078 MB regardless of sequence length.

---

## Memory vs Compute Tradeoff

|                    | Standard   | Tiled            |
|--------------------|------------|------------------|
| Forward memory     | O(n²)      | O(block_size²)   |
| Backward memory    | O(n²)      | O(n)             |
| Forward compute    | 1x         | 1x               |
| Backward compute   | 1x         | ~2x (recompute)  |

The backward pass does roughly twice the computation since the forward pass runs again during recomputation. In practice this is almost always worth it — GPU memory is the bottleneck, not compute.

---

## Usage

```bash
pip install torch
python Round2_Code_YourName.py
```

To test with different parameters:

```python
run_verification(n=512, d=64, block_size=32)
```

---

## Requirements

- Python 3.8+
- PyTorch 1.10+

