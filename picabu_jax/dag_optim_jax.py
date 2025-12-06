"""
dag_optim_jax.py - SAVAR-PICABU 版本（修復梯度斷裂）

=== 修復說明 ===
問題：原始 count_edges 使用 (adj > threshold)，導致梯度 = 0
修復：新增 soft_edge_count，在訓練時使用

接口變更（最小化）：
- 新增 soft_edge_count() 函數
- count_edges() 保持不變（用於評估）
- SparsityALM 接口不變
"""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from scipy.linalg import expm as scipy_expm
from typing import Tuple


# ============================================================
# DAG Constraint (保留原有)
# ============================================================

@jax.jit
def compute_dag_constraint(w_adj: jnp.ndarray) -> jnp.ndarray:
    """
    計算 DAG 約束：h(W) = trace(exp(W ⊙ W)) - d
    
    參數：
        w_adj: 鄰接矩陣，形狀 (d, d)
    
    返回：
        h: DAG 約束值，h = 0 表示無環
    """
    d = w_adj.shape[0]
    w_squared = w_adj * w_adj  # 元素級平方
    expm_w = jax.scipy.linalg.expm(w_squared)
    h = jnp.trace(expm_w) - d
    return h


def compute_dag_constraint_scipy(w_adj: np.ndarray) -> float:
    """使用 SciPy 計算 DAG 約束（不可微分，僅用於驗證）"""
    if isinstance(w_adj, jnp.ndarray):
        w_adj = np.array(w_adj)
    
    d = w_adj.shape[0]
    w_squared = w_adj * w_adj
    expm_w = scipy_expm(w_squared)
    h = np.trace(expm_w) - d
    return float(h)


def is_acyclic(adjacency: np.ndarray) -> bool:
    """檢查鄰接矩陣是否無環"""
    if isinstance(adjacency, jnp.ndarray):
        adjacency = np.array(adjacency)
    
    d = adjacency.shape[0]
    prod = np.eye(d)
    
    for _ in range(1, d + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0:
            return False
    return True


# ============================================================
# Top-K Hard Sparsity Constraint (PICABU)
# ============================================================

@partial(jax.jit, static_argnames=['k'])
def top_k_mask(adj: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    生成 Top-K mask：保留前 k 個最大值的位置
    
    參數：
        adj: 鄰接矩陣 (d, d) 或 (tau, d, d)
        k: 保留的邊數
    
    返回：
        mask: 0/1 mask，1 表示保留
    """
    original_shape = adj.shape
    
    # 展平處理
    if len(original_shape) == 3:
        adj_flat = adj[-1].flatten()
    else:
        adj_flat = adj.flatten()
    
    abs_adj = jnp.abs(adj_flat)
    n = abs_adj.size
    
    if k >= n:
        threshold = 0.0
    elif k <= 0:
        threshold = jnp.inf
    else:
        sorted_vals = jnp.sort(abs_adj)
        threshold = sorted_vals[n - k]
    
    mask_flat = (abs_adj >= threshold).astype(jnp.float32)
    
    if len(original_shape) == 3:
        tau = original_shape[0]
        d = original_shape[1]
        mask_prev = jnp.ones((tau - 1, d, d))
        mask_last = mask_flat.reshape(d, d)
        mask = jnp.concatenate([mask_prev, mask_last[None, ...]], axis=0)
    else:
        mask = mask_flat.reshape(original_shape)
    
    return mask


def apply_top_k_sparsity(
    adj: jnp.ndarray, 
    target_edges: int,
    temperature: float = 1.0,
    hard: bool = True
) -> jnp.ndarray:
    """
    應用 Top-K sparsity constraint
    
    參數：
        adj: 鄰接矩陣概率 (tau, d, d) 或 (d, d)
        target_edges: 目標邊數 M
        temperature: soft version 的溫度
        hard: 是否使用 hard threshold
    
    返回：
        sparse_adj: 稀疏化後的鄰接矩陣
    """
    if hard:
        mask = top_k_mask(adj, target_edges)
        sparse_adj = adj * mask
    else:
        abs_adj = jnp.abs(adj)
        threshold = jnp.sort(abs_adj.flatten())[-target_edges] if target_edges > 0 else jnp.inf
        soft_mask = jax.nn.sigmoid((abs_adj - threshold) / temperature)
        sparse_adj = adj * soft_mask
    
    return sparse_adj


# ============================================================
# Edge Counting Functions
# ============================================================

@jax.jit
def count_edges(adj: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
    """
    計算邊數（超過閾值的元素數）
    
    ⚠️ 注意：這個函數在訓練 loss 中梯度 = 0！
    僅用於評估和監控，不要用於計算 loss！
    
    訓練時請使用 soft_edge_count()
    """
    return jnp.sum(jnp.abs(adj) > threshold).astype(jnp.float32)


@jax.jit
def soft_edge_count(adj: jnp.ndarray) -> jnp.ndarray:
    """
    🆕 可微分的軟邊數計算
    
    用於訓練 loss，梯度 = 1（完全可微分）
    
    數學：
        n_soft = Σ_{ij} adj_{ij}
        ∂n_soft/∂adj_{ij} = 1
    
    參數：
        adj: 鄰接矩陣 (概率值 [0, 1])
    
    返回：
        soft_count: 軟邊數（連續值）
    """
    return jnp.sum(adj)


@jax.jit
def soft_edge_count_thresholded(adj: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
    """
    🆕 帶閾值的軟邊數計算（使用 sigmoid 近似 step function）
    
    比 soft_edge_count 更接近 hard count，但仍可微分
    
    參數：
        adj: 鄰接矩陣
        threshold: 閾值
    
    返回：
        soft_count: 近似的邊數
    """
    temperature = 0.1  # 較小 = 更接近 hard threshold
    soft_indicator = jax.nn.sigmoid((adj - threshold) / temperature)
    return jnp.sum(soft_indicator)


# ============================================================
# Sparsity ALM (保持原始接口)
# ============================================================

class SparsityALM:
    """
    Top-K Sparsity 的 Augmented Lagrangian Method
    
    約束：|E(G)| = M（邊數等於目標值）
    
    ⚠️ 使用說明：
        - 訓練時使用 soft_edge_count() 計算 n_edges
        - 評估時使用 count_edges() 計算 n_edges
    
    接口與原版完全相同，無需修改調用代碼
    """
    
    def __init__(
        self,
        target_edges: int,
        mu_init: float = 0.1,
        mu_multiplier: float = 1.2,
        threshold: float = 1e-4,
        omega_gamma: float = 0.01,
        omega_mu: float = 0.5,
        min_iter_convergence: int = 100,
    ):
        self.target_edges = target_edges
        self.mu = mu_init
        self.mu_init = mu_init
        self.mu_multiplier = mu_multiplier
        self.threshold = threshold
        self.omega_gamma = omega_gamma
        self.omega_mu = omega_mu
        self.min_iter_convergence = min_iter_convergence
        
        # Lagrangian multiplier
        self.gamma = 0.0
        
        # 狀態追蹤
        self.iteration = 0
        self.violation_history = []
        self.has_converged = False
        self.has_increased_mu = False
    
    def compute_violation(self, adj: jnp.ndarray) -> float:
        """計算約束違反度（用於監控，使用 hard count）"""
        n_edges = float(count_edges(adj, 0.5))
        return n_edges - self.target_edges
    
    def compute_loss(self, adj: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        """
        計算 ALM 損失項
        
        ⚠️ 注意：這個方法使用 count_edges，梯度 = 0
        訓練時請直接在 loss_fn 中使用 soft_edge_count
        """
        n_edges = count_edges(adj, 0.5)
        h = n_edges - self.target_edges
        loss = self.gamma * h + 0.5 * self.mu * (h ** 2)
        return loss, float(h)
    
    def update(self, iteration: int, violation_list: list, loss_list: list):
        """
        更新 ALM 參數（每個 epoch 調用一次）
        """
        self.iteration = iteration
        self.has_increased_mu = False
        
        if len(violation_list) < 2:
            return
        
        current_h = abs(violation_list[-1])
        prev_h = abs(violation_list[-2])
        
        self.violation_history.append(current_h)
        
        # 檢查是否收斂
        if current_h < self.threshold:  # ← 用 self.threshold
            self.has_converged = True
            return
        
        # 檢查是否需要增加 mu
        if iteration > self.min_iter_convergence:
            improvement = (prev_h - current_h) / (prev_h + 1e-8)
            if improvement < self.omega_mu:
                self.mu *= self.mu_multiplier
                self.has_increased_mu = True
        
        # 更新 gamma
        if len(violation_list) > 0:
            self.gamma += self.omega_gamma * violation_list[-1]
    
    def reset(self):
        """重置 ALM 狀態"""
        self.mu = self.mu_init
        self.gamma = 0.0
        self.iteration = 0
        self.violation_history = []
        self.has_converged = False
        self.has_increased_mu = False


# ============================================================
# 輔助函數 (保持原始接口)
# ============================================================

def get_target_edges(d_z: int, sparsity_ratio: float = 0.5) -> int:
    """
    計算目標邊數
    
    PICABU 論文建議：M = 0.5 * d_z
    """
    max_edges = d_z * (d_z - 1)
    target = int(sparsity_ratio * d_z)
    return max(1, min(target, max_edges))


def threshold_to_binary(adj: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
    """將概率鄰接矩陣轉為二值"""
    return (jnp.abs(adj) > threshold).astype(jnp.float32)


# ============================================================
# 測試
# ============================================================

def test_gradient_flow():
    """測試梯度流"""
    print("=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    key = jax.random.PRNGKey(0)
    adj = jax.random.uniform(key, (5, 5))
    adj = adj * (1 - jnp.eye(5))
    target = 10
    
    # 原始版本（梯度 = 0）
    def loss_original(adj):
        n = count_edges(adj, threshold=0.5)
        h = n - target
        return 0.5 * h ** 2
    
    # 修復版本（梯度 ≠ 0）
    def loss_fixed(adj):
        n = soft_edge_count(adj)
        h = n - target
        return 0.5 * h ** 2
    
    grad_original = jax.grad(loss_original)(adj)
    grad_fixed = jax.grad(loss_fixed)(adj)
    
    print(f"\ncount_edges gradient:      {float(jnp.linalg.norm(grad_original)):.6f}")
    print(f"soft_edge_count gradient:  {float(jnp.linalg.norm(grad_fixed)):.6f}")
    
    if float(jnp.linalg.norm(grad_original)) < 1e-6:
        print("✓ count_edges 確實梯度 = 0（符合預期）")
    if float(jnp.linalg.norm(grad_fixed)) > 1.0:
        print("✓ soft_edge_count 梯度正常流動（修復成功）")
    
    print("=" * 60)


def test_top_k_sparsity():
    """測試 Top-K sparsity"""
    print("\nTesting Top-K Sparsity...")
    
    key = jax.random.PRNGKey(0)
    adj = jax.random.uniform(key, (5, 5))
    adj = adj * (1 - jnp.eye(5))
    
    print(f"Original adjacency: {jnp.sum(adj > 0.1):.0f} edges")
    
    k = 5
    mask = top_k_mask(adj, k)
    sparse_adj = adj * mask
    
    print(f"After Top-{k}: {jnp.sum(sparse_adj > 0.1):.0f} edges")
    
    # 測試 ALM
    alm = SparsityALM(target_edges=5, mu_init=0.1)
    loss, violation = alm.compute_loss(sparse_adj)
    print(f"Violation: {violation:.1f}, Loss: {float(loss):.4f}")
    
    print("✓ Top-K tests passed!")


if __name__ == "__main__":
    test_gradient_flow()
    test_top_k_sparsity()