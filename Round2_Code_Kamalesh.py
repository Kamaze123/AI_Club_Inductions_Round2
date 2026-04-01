import torch
import math

def standard_attention(X):
    n,d = X.shape

    S = X @ X.T

    S = S/math.sqrt(d)
    A = torch.softmax(S, dim =-1)
    Y = A @ X

    return Y

def forward_pass(X, block_size = 32):
    n,d = X.shape
    scale = 1.0 / math.sqrt(d)

    Y = torch.zeros(n,d)
    m_stats = torch.full((n,), float('-inf'))  
    l_stats = torch.zeros(n)

    for q_start in range(0, n, block_size):
        q_end = min(q_start + block_size, n)
        Q_block = X[q_start:q_end]
        bq = q_end - q_start

        m = torch.full((bq,), float('-inf'))   
        l = torch.zeros(bq)                    
        O = torch.zeros(bq, d)

        for kv_start in range(0,n, block_size):
            kv_end = min(kv_start + block_size, n)
            K_block = X[kv_start: kv_end]
            V_block = X[kv_start:kv_end] 

            S = (Q_block @ K_block.T) * scale

            m_block = S.max(dim = -1).values
            m_new = torch.maximum(m, m_block)

            correction = torch.exp(m - m_new)
            exp_S = torch.exp(S - m_new.unsqueeze(1))

            l_new = correction * l + exp_S.sum(dim=-1)
            O = correction.unsqueeze(1) * O + exp_S @ V_block 

            m = m_new
            l = l_new
        
        Y[q_start:q_end] = O/l.unsqueeze(1)
        m_stats[q_start:q_end] = m
        l_stats[q_start:q_end] = l

    return Y,m_stats, l_stats

def backward_pass(X, dY, m_stats, l_stats, block_size=32):
    n, d = X.shape
    scale = 1.0 / math.sqrt(d)
    dX = torch.zeros_like(X)
 
    D = torch.zeros(n)
 
    for q_start in range(0, n, block_size):
        q_end = min(q_start + block_size, n)
        Q_block = X[q_start:q_end]
        dY_block = dY[q_start:q_end]
        m_block = m_stats[q_start:q_end]
        l_block = l_stats[q_start:q_end]
 
        for kv_start in range(0, n, block_size):
            kv_end = min(kv_start + block_size, n)
            K_block = X[kv_start:kv_end]
            V_block = X[kv_start:kv_end]
 
            S = (Q_block @ K_block.T) * scale
            A = torch.exp(S - m_block.unsqueeze(1)) / l_block.unsqueeze(1)
 
            dA = dY_block @ V_block.T
            D[q_start:q_end] += (A * dA).sum(dim=-1)
 
    for q_start in range(0, n, block_size):
        q_end = min(q_start + block_size, n)
        Q_block = X[q_start:q_end]
        dY_block = dY[q_start:q_end]
        m_block = m_stats[q_start:q_end]
        l_block = l_stats[q_start:q_end]
        D_block = D[q_start:q_end]
 
        for kv_start in range(0, n, block_size):
            kv_end = min(kv_start + block_size, n)
            K_block = X[kv_start:kv_end]
            V_block = X[kv_start:kv_end]
 
            S = (Q_block @ K_block.T) * scale
            A = torch.exp(S - m_block.unsqueeze(1)) / l_block.unsqueeze(1)
 
            dX[kv_start:kv_end] += A.T @ dY_block
 
            dA = dY_block @ V_block.T
            dS = A * (dA - D_block.unsqueeze(1))
            dS = dS * scale
 
            dX[q_start:q_end]   += dS @ K_block
            dX[kv_start:kv_end] += dS.T @ Q_block
 
    return dX


def bytes_to_mb(n_elements):
    return (n_elements * 4) / (1024 ** 2)

def run_verification(n=128, d=32, block_size=32):
    print("=" * 60)
    print(f"  Verification: n={n}, d={d}, block_size={block_size}")
    print("=" * 60)
 
    torch.manual_seed(42)
    X = torch.randn(n, d)
 
    Y_standard = standard_attention(X)
    Y_tiled, m_stats, l_stats = forward_pass(X, block_size)
 
    max_diff = (Y_standard - Y_tiled).abs().max().item()
    print(f"\n[Forward Pass]")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  {'✓ Outputs match!' if max_diff < 1e-5 else '✗ Outputs differ!'}")
 
    X_auto = X.clone().requires_grad_(True)
    Y_auto = standard_attention(X_auto)
    loss = Y_auto.sum()
    loss.backward()
    dX_standard = X_auto.grad.clone()
 
    dY = torch.ones_like(Y_tiled)
    dX_tiled = backward_pass(X, dY, m_stats, l_stats, block_size)
 
    grad_diff = (dX_standard - dX_tiled).abs().max().item()
    print(f"\n[Backward Pass]")
    print(f"  Max gradient difference: {grad_diff:.2e}")
    print(f"  {'✓ Gradients match!' if grad_diff < 1e-4 else '✗ Gradients differ!'}")
 
    print(f"\n[Memory Analysis]")
    print(f"  Standard attention matrix:  {bytes_to_mb(n * n * 2):.4f} MB  (n={n})")
    print(f"  Tiled attention block:      {bytes_to_mb(block_size * block_size * 2):.4f} MB  (block={block_size})")
    print(f"  Standard backward mem:      {bytes_to_mb(n * n):.4f} MB")
    print(f"  Tiled backward stat mem:    {bytes_to_mb(n * 2):.4f} MB")
    print(f"  Reduction factor:           {(n * n) / (n * 2):.1f}×")
 
    print(f"\n[Scaling]")
    print(f"  {'n':>8}  {'Standard':>14}  {'Tiled':>14}")
    for test_n in [128, 512, 1024, 4096, 10000]:
        std = bytes_to_mb(test_n * test_n)
        til = bytes_to_mb(block_size * block_size)
        print(f"  {test_n:>8}  {std:>12.2f}MB  {til:>12.4f}MB")
 
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_verification(n=128, d=32, block_size=32)

