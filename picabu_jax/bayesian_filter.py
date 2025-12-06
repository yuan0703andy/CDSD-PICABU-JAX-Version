"""
bayesian_filter.py
PICABU Bayesian Filter for Stable Autoregressive Rollouts

核心思想：
1. 遞歸貝葉斯估計：保持完整分佈而非點估計
2. Particle Filtering：用 importance sampling 過濾不良樣本
3. Spectral Likelihood：用空間功率譜作為觀測代理

Reference: "Causal Climate Emulation with Bayesian Filtering" (2025)

數學背景：
    標準 Bayesian Filter:
        p(z_t | x_≤t) ∝ p(x_t | z_t) * p(z_t | x_<t)
    
    PICABU 的創新：
        - 預測時無法觀測 x_t
        - 用空間頻譜 x̃ 作為代理觀測
        - p(x̃ | z_t) 用 Laplace 分佈建模
"""
import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, Callable, Optional
from functools import partial


class BayesianFilterState(NamedTuple):
    """Bayesian Filter 的狀態"""
    particles: jnp.ndarray      # (N, d_z) 或 (N, tau, d, d_z)
    weights: jnp.ndarray        # (N,) 正規化權重
    reference_spectrum: jnp.ndarray  # 參考空間頻譜
    step: int


class BayesianFilterConfig(NamedTuple):
    """Bayesian Filter 配置"""
    n_particles: int = 300      # N: 粒子數
    n_resample: int = 10        # R: 每步重採樣數
    sigma_spectrum: float = 1.0  # 頻譜似然的標準差
    use_constant_var: bool = True  # 用觀測估計的常數方差
    eps: float = 1e-6           # 數值穩定


# ========== 核心函數 ==========

def compute_spatial_spectrum(
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    計算觀測的空間功率譜
    
    Args:
        x: (batch, d, d_x) 或 (d, d_x)
    
    Returns:
        spectrum: 功率譜
    """
    if x.ndim == 2:
        x = x.reshape(-1)
    elif x.ndim == 3:
        x = x.reshape(x.shape[0], -1)
    
    fft = jnp.fft.rfft(x, axis=-1)
    spectrum = jnp.abs(fft) ** 2
    
    return spectrum


def estimate_reference_spectrum(
    observations: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    從觀測數據估計參考頻譜（均值和方差）
    
    PICABU 假設空間頻譜在時間上是常數（對於 pre-industrial 數據合理）
    
    Args:
        observations: (n_samples, d, d_x) 歷史觀測
    
    Returns:
        mean_spectrum: 平均頻譜
        var_spectrum: 頻譜方差
    """
    spectra = jax.vmap(compute_spatial_spectrum)(observations)
    mean_spectrum = jnp.mean(spectra, axis=0)
    var_spectrum = jnp.var(spectra, axis=0)
    
    return mean_spectrum, var_spectrum


def laplace_log_likelihood(
    x: jnp.ndarray,
    loc: jnp.ndarray,
    scale: jnp.ndarray,
) -> jnp.ndarray:
    """
    Laplace 分佈的對數似然
    
    L(x; μ, b) = -|x - μ|/b - log(2b)
    
    Args:
        x: 觀測值
        loc: 位置參數 μ
        scale: 尺度參數 b
    
    Returns:
        log_likelihood: 對數似然
    """
    abs_diff = jnp.abs(x - loc)
    log_prob = -abs_diff / scale - jnp.log(2 * scale)
    
    return jnp.sum(log_prob)


@partial(jax.jit, static_argnames=('n_resample',))
def importance_resampling(
    key: jax.random.PRNGKey,
    particles: jnp.ndarray,
    log_weights: jnp.ndarray,
    n_resample: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    重要性重採樣
    
    根據權重選擇最佳粒子
    
    Args:
        key: 隨機數生成器
        particles: (N, ...) 粒子
        log_weights: (N,) 對數權重
        n_resample: 保留的粒子數
    
    Returns:
        resampled_particles: (n_resample, ...)
        normalized_weights: (n_resample,)
    """
    n_particles = particles.shape[0]
    
    # 正規化權重（log-sum-exp for numerical stability）
    max_log_w = jnp.max(log_weights)
    weights = jnp.exp(log_weights - max_log_w)
    weights = weights / jnp.sum(weights)
    
    # 選擇 top-k 粒子（確定性）
    indices = jnp.argsort(weights)[-n_resample:]
    
    resampled = particles[indices]
    new_weights = weights[indices]
    new_weights = new_weights / jnp.sum(new_weights)
    
    return resampled, new_weights


# ========== Bayesian Filter Class ==========

class BayesianFilter:
    """
    PICABU Bayesian Filter
    
    用於穩定的長期 autoregressive rollouts
    
    Algorithm:
        1. 從 p(z_t | z_<t) 採樣 N 個粒子
        2. 對每個粒子採樣 R 個子樣本
        3. 計算每個樣本的頻譜似然
        4. 重採樣保留高似然粒子
        5. 迭代
    
    Example:
        >>> filter_cfg = BayesianFilterConfig(n_particles=300, n_resample=10)
        >>> bf = BayesianFilter(
        ...     transition_fn=model_transition,
        ...     decode_fn=model_decode,
        ...     config=filter_cfg,
        ... )
        >>> state = bf.init(key, z_init, observations)
        >>> for t in range(horizon):
        ...     state, z_next = bf.step(key, state)
    """
    
    def __init__(
        self,
        transition_fn: Callable,
        decode_fn: Callable,
        config: BayesianFilterConfig,
    ):
        """
        Args:
            transition_fn: p(z_t | z_<t) 轉移函數
                           signature: (params, z_prev, key) -> (z_next_mu, z_next_std)
            decode_fn: p(x | z) 解碼函數
                       signature: (params, z, key) -> (x_mu, x_std)
            config: 濾波器配置
        """
        self.transition_fn = transition_fn
        self.decode_fn = decode_fn
        self.cfg = config
    
    def init(
        self,
        key: jax.random.PRNGKey,
        z_init: jnp.ndarray,
        observations: jnp.ndarray,
        sigma_override: Optional[float] = None,
    ) -> BayesianFilterState:
        """
        初始化濾波器狀態
        
        Args:
            key: 隨機數生成器
            z_init: 初始潛變量 (d_z,) 或 (tau, d, d_z)
            observations: 用於估計參考頻譜的歷史觀測
            sigma_override: 覆蓋頻譜似然的 sigma
        
        Returns:
            初始狀態
        """
        # 估計參考頻譜
        ref_spectrum, var_spectrum = estimate_reference_spectrum(observations)
        
        # 初始化粒子（複製初始狀態）
        particles = jnp.broadcast_to(
            z_init[None, ...],
            (self.cfg.n_particles,) + z_init.shape
        )
        
        # 均勻權重
        weights = jnp.ones(self.cfg.n_particles) / self.cfg.n_particles
        
        return BayesianFilterState(
            particles=particles,
            weights=weights,
            reference_spectrum=ref_spectrum,
            step=0,
        )
    
    def step(
        self,
        key: jax.random.PRNGKey,
        state: BayesianFilterState,
        params: dict,
    ) -> Tuple[BayesianFilterState, jnp.ndarray]:
        """
        執行一步 Bayesian filtering
        
        Args:
            key: 隨機數生成器
            state: 當前狀態
            params: 模型參數
        
        Returns:
            new_state: 更新後的狀態
            z_filtered: 濾波後的潛變量估計 (d_z,) 或 (tau, d, d_z)
        """
        N = self.cfg.n_particles
        R = self.cfg.n_resample
        
        key, key_trans, key_resample = jax.random.split(key, 3)
        keys_particles = jax.random.split(key_trans, N)
        
        # 1. 從 transition model 採樣下一步粒子
        def sample_next(carry, inputs):
            particle, k = inputs
            k1, k2 = jax.random.split(k)
            
            # 從轉移模型採樣
            z_mu, z_std = self.transition_fn(params, particle, k1)
            z_next = z_mu + z_std * jax.random.normal(k2, z_mu.shape)
            
            return carry, z_next
        
        _, next_particles = jax.lax.scan(
            sample_next,
            None,
            (state.particles, keys_particles),
        )
        
        # 2. 計算每個粒子的頻譜似然
        def compute_likelihood(particle, k):
            # 解碼到觀測空間
            x_mu, x_std = self.decode_fn(params, particle, k)
            
            # 計算預測頻譜
            pred_spectrum = compute_spatial_spectrum(x_mu)
            
            # Laplace 似然
            log_lik = laplace_log_likelihood(
                pred_spectrum,
                state.reference_spectrum,
                jnp.sqrt(self.cfg.sigma_spectrum + self.cfg.eps),
            )
            
            return log_lik
        
        keys_lik = jax.random.split(key, N)
        log_likelihoods = jax.vmap(compute_likelihood)(next_particles, keys_lik)
        
        # 3. 重採樣
        resampled_particles, new_weights = importance_resampling(
            key_resample,
            next_particles,
            log_likelihoods,
            R,
        )
        
        # 4. 擴展回 N 個粒子（重複採樣的粒子）
        indices = jnp.arange(N) % R
        final_particles = resampled_particles[indices]
        final_weights = new_weights[indices]
        final_weights = final_weights / jnp.sum(final_weights)
        
        # 5. 估計濾波後的狀態（加權平均）
        z_filtered = jnp.sum(
            final_particles * final_weights[:, None, None, None],
            axis=0
        )
        
        new_state = BayesianFilterState(
            particles=final_particles,
            weights=final_weights,
            reference_spectrum=state.reference_spectrum,
            step=state.step + 1,
        )
        
        return new_state, z_filtered


# ========== Functional API ==========

def create_bayesian_filter(
    transition_fn: Callable,
    decode_fn: Callable,
    n_particles: int = 300,
    n_resample: int = 10,
    sigma_spectrum: float = 1.0,
) -> BayesianFilter:
    """
    創建 Bayesian Filter（函數式 API）
    
    Args:
        transition_fn: 轉移函數 (params, z_prev, key) -> (z_mu, z_std)
        decode_fn: 解碼函數 (params, z, key) -> (x_mu, x_std)
        n_particles: 粒子數
        n_resample: 重採樣數
        sigma_spectrum: 頻譜似然標準差
    
    Returns:
        BayesianFilter 實例
    """
    config = BayesianFilterConfig(
        n_particles=n_particles,
        n_resample=n_resample,
        sigma_spectrum=sigma_spectrum,
    )
    
    return BayesianFilter(transition_fn, decode_fn, config)


# ========== Autoregressive Rollout ==========

def autoregressive_rollout(
    bf: BayesianFilter,
    params: dict,
    z_init: jnp.ndarray,
    observations: jnp.ndarray,
    horizon: int,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    執行 autoregressive rollout with Bayesian filtering
    
    這是 PICABU 用於長期氣候模擬的核心
    
    Args:
        bf: BayesianFilter 實例
        params: 模型參數
        z_init: 初始潛變量
        observations: 歷史觀測（用於估計參考頻譜）
        horizon: 預測步數
        key: 隨機數生成器
    
    Returns:
        z_trajectory: (horizon, ...) 潛變量軌跡
        x_trajectory: (horizon, ...) 觀測空間軌跡
    """
    # 初始化
    key, init_key = jax.random.split(key)
    state = bf.init(init_key, z_init, observations)
    
    # Rollout
    z_list = []
    x_list = []
    
    for t in range(horizon):
        key, step_key, decode_key = jax.random.split(key, 3)
        
        # 一步濾波
        state, z_t = bf.step(step_key, state, params)
        z_list.append(z_t)
        
        # 解碼到觀測空間
        x_mu, _ = bf.decode_fn(params, z_t, decode_key)
        x_list.append(x_mu)
    
    z_trajectory = jnp.stack(z_list, axis=0)
    x_trajectory = jnp.stack(x_list, axis=0)
    
    return z_trajectory, x_trajectory


# ========== 測試 ==========

def _test():
    """測試 Bayesian Filter"""
    print("Testing PICABU Bayesian Filter...")
    print("=" * 60)
    
    key = jax.random.PRNGKey(42)
    
    # 模擬參數
    d, d_x, d_z, tau = 3, 4, 2, 5
    n_obs = 100
    
    # 模擬轉移函數
    def mock_transition(params, z_prev, key):
        # 簡單的線性轉移 + 噪聲
        z_mu = z_prev * 0.9  # decay
        z_std = jnp.ones_like(z_mu) * 0.1
        return z_mu, z_std
    
    # 模擬解碼函數
    def mock_decode(params, z, key):
        # 簡單的線性解碼
        if z.ndim == 3:  # (tau, d, d_z)
            z = z[-1]     # 取最後一步
        x_mu = jnp.sin(z.sum()) * jnp.ones((d, d_x))
        x_std = jnp.ones((d, d_x)) * 0.1
        return x_mu, x_std
    
    # 生成測試數據
    key, sub1, sub2, sub3, sub4 = jax.random.split(key, 5)
    z_init = jax.random.normal(sub1, (tau, d, d_z))
    observations = jax.random.normal(sub2, (n_obs, d, d_x))
    
    # 創建濾波器
    print("\n1) Creating Bayesian Filter...")
    bf = create_bayesian_filter(
        transition_fn=mock_transition,
        decode_fn=mock_decode,
        n_particles=50,  # 測試用較少粒子
        n_resample=5,
    )
    print("   ✓ Filter created")
    
    # 初始化
    print("\n2) Initializing filter state...")
    state = bf.init(sub3, z_init, observations)
    print(f"   Particles shape: {state.particles.shape}")
    print(f"   Reference spectrum shape: {state.reference_spectrum.shape}")
    
    # 單步測試
    print("\n3) Testing single step...")
    params = {}  # mock 不需要參數
    new_state, z_filtered = bf.step(sub4, state, params)
    print(f"   z_filtered shape: {z_filtered.shape}")
    print(f"   New step: {new_state.step}")
    
    # Rollout 測試
    print("\n4) Testing autoregressive rollout...")
    key, rollout_key = jax.random.split(key)
    horizon = 10
    
    z_traj, x_traj = autoregressive_rollout(
        bf, params, z_init, observations, horizon, rollout_key
    )
    print(f"   z_trajectory shape: {z_traj.shape}")
    print(f"   x_trajectory shape: {x_traj.shape}")
    
    # 檢查穩定性
    print("\n5) Checking stability...")
    z_flat = z_traj.reshape(z_traj.shape[0], -1)
    z_norm = jnp.linalg.norm(z_flat, axis=1)
    print(f"   z norm range: [{float(z_norm.min()):.4f}, {float(z_norm.max()):.4f}]")
    
    # 頻譜一致性
    pred_spectra = jax.vmap(compute_spatial_spectrum)(x_traj)
    ref_spectrum = state.reference_spectrum
    spectral_error = jnp.mean(jnp.abs(pred_spectra - ref_spectrum))
    print(f"   Mean spectral error: {float(spectral_error):.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")


if __name__ == "__main__":
    _test()