"""
picabu_losses_jax.py - SAVAR-PICABU 版本

SAVAR 適用的 PICABU 輔助損失函數：
1. CRPS (Continuous Ranked Probability Score) - Gaussian closed-form
2. Temporal Spectral Loss - 1D FFT

=== 注意 ===
移除 Spatial Spectral Loss（SAVAR 沒有 2D 空間網格）

參考：
- PICABU 論文 page 4-5
- CDSD 論文 page 4 (Gaussian decoder)
"""
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any


# ============================================================
# CRPS Loss (Gaussian Closed-Form)
# ============================================================

@jax.jit
def crps_gaussian(
    y_true: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    """
    Continuous Ranked Probability Score for Gaussian distribution
    
    CRPS 衡量預測分布與真實值的距離，是 proper scoring rule。
    
    閉式解（Gaussian）：
        CRPS(N(μ,σ²), y) = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
        where z = (y - μ) / σ
    
    參數：
        y_true: 真實值 (batch, ...)
        mu: 預測均值 (batch, ...)
        sigma: 預測標準差 (batch, ...) 或 (...,) 會自動 broadcast
    
    返回：
        crps: CRPS 值（越小越好）
    
    參考：
        Gneiting & Raftery (2007) "Strictly Proper Scoring Rules"
    """
    # 確保 sigma 是正數
    sigma = jnp.maximum(sigma, 1e-6)
    
    # 標準化
    z = (y_true - mu) / sigma
    
    # 標準正態 PDF 和 CDF
    phi = jax.scipy.stats.norm.pdf(z)  # φ(z)
    Phi = jax.scipy.stats.norm.cdf(z)  # Φ(z)
    
    # CRPS 閉式解
    sqrt_pi = jnp.sqrt(jnp.pi)
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt_pi)
    
    return crps


@jax.jit
def crps_loss(
    y_true: jnp.ndarray,
    y_pred_mu: jnp.ndarray,
    logvar_decoder: jnp.ndarray,
) -> jnp.ndarray:
    """
    計算 CRPS 損失（用於訓練）
    
    參數：
        y_true: 真實觀測值 (batch, d, d_x)
        y_pred_mu: 預測均值 (batch, d, d_x)
        logvar_decoder: decoder 的 log-variance (d_x,)
    
    返回：
        loss: 平均 CRPS（scalar）
    """
    # 從 log-variance 計算 std
    sigma = jnp.exp(0.5 * logvar_decoder)
    
    # 計算 CRPS
    crps = crps_gaussian(y_true, y_pred_mu, sigma)
    
    # 平均
    return jnp.mean(crps)


# ============================================================
# Temporal Spectral Loss (1D FFT)
# ============================================================

@jax.jit
def compute_temporal_spectrum(x: jnp.ndarray) -> jnp.ndarray:
    """
    計算時間序列的功率譜密度 (PSD)
    
    使用 1D FFT 沿時間維度計算功率譜。
    
    參數：
        x: 時間序列 (batch, time, features) 或 (time, features)
    
    返回：
        psd: 功率譜密度 (batch, freq, features) 或 (freq, features)
    
    注意：
        - 只返回正頻率部分（Nyquist）
        - 使用 |FFT|² 計算功率
    """
    # 沿時間維度做 FFT
    if x.ndim == 2:
        # (time, features)
        fft_result = jnp.fft.rfft(x, axis=0)
        psd = jnp.abs(fft_result) ** 2
    elif x.ndim == 3:
        # (batch, time, features)
        fft_result = jnp.fft.rfft(x, axis=1)
        psd = jnp.abs(fft_result) ** 2
    else:
        raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")
    
    # 正規化
    psd = psd / (x.shape[-2] if x.ndim == 3 else x.shape[0])
    
    return psd


@jax.jit  
def temporal_spectral_loss(
    x_true: jnp.ndarray,
    x_pred: jnp.ndarray,
) -> jnp.ndarray:
    """
    計算時間頻譜損失
    
    L_temporal = mean(|PSD(x_true) - PSD(x_pred)|)
    
    參數：
        x_true: 真實時間序列 (batch, time, features) 或 (time, features)
        x_pred: 預測時間序列（同形狀）
    
    返回：
        loss: L1 頻譜距離
    
    注意：
        PICABU 論文使用 L1 距離（對高頻更敏感）
    """
    psd_true = compute_temporal_spectrum(x_true)
    psd_pred = compute_temporal_spectrum(x_pred)
    
    # L1 距離
    loss = jnp.mean(jnp.abs(psd_true - psd_pred))
    
    return loss


@jax.jit
def log_spectral_distance(
    x_true: jnp.ndarray,
    x_pred: jnp.ndarray,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """
    計算 Log-Spectral Distance (LSD)
    
    LSD = sqrt(mean((log(PSD_true) - log(PSD_pred))²))
    
    這是語音處理中常用的頻譜距離度量。
    
    參數：
        x_true: 真實時間序列
        x_pred: 預測時間序列
        eps: 數值穩定性
    
    返回：
        lsd: Log-spectral distance
    """
    psd_true = compute_temporal_spectrum(x_true)
    psd_pred = compute_temporal_spectrum(x_pred)
    
    # Log spectral distance
    log_diff = jnp.log(psd_true + eps) - jnp.log(psd_pred + eps)
    lsd = jnp.sqrt(jnp.mean(log_diff ** 2))
    
    return lsd


# ============================================================
# 組合損失函數
# ============================================================

@jax.jit
def picabu_auxiliary_loss(
    y_true: jnp.ndarray,
    y_pred_mu: jnp.ndarray,
    logvar_decoder: jnp.ndarray,
    x_history_true: jnp.ndarray = None,
    x_history_pred: jnp.ndarray = None,
    coeff_crps: float = 1.0,
    coeff_temporal: float = 2000.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    計算 PICABU 輔助損失（SAVAR 版本）
    
    L_aux = coeff_crps * CRPS + coeff_temporal * L_temporal
    
    參數：
        y_true: 當前時間步的真實觀測 (batch, d, d_x)
        y_pred_mu: 當前時間步的預測均值 (batch, d, d_x)
        logvar_decoder: decoder log-variance (d_x,)
        x_history_true: 歷史真實序列 (batch, tau, d, d_x)（可選）
        x_history_pred: 歷史預測序列（可選）
        coeff_crps: CRPS 係數（默認 1.0）
        coeff_temporal: 時間頻譜係數（默認 2000.0）
    
    返回：
        total_loss: 總輔助損失
        metrics: 各項損失的字典
    """
    # 1. CRPS loss
    loss_crps = crps_loss(y_true, y_pred_mu, logvar_decoder)
    
    # 2. Temporal spectral loss（如果有歷史數據）
    if x_history_true is not None and x_history_pred is not None:
        # 重塑為 (batch, time, features)
        batch, tau, d, d_x = x_history_true.shape
        x_true_flat = x_history_true.reshape(batch, tau, d * d_x)
        x_pred_flat = x_history_pred.reshape(batch, tau, d * d_x)
        
        loss_temporal = temporal_spectral_loss(x_true_flat, x_pred_flat)
    else:
        loss_temporal = jnp.array(0.0)
    
    # 總損失
    total_loss = coeff_crps * loss_crps + coeff_temporal * loss_temporal
    
    metrics = {
        'crps': loss_crps,
        'temporal_spectral': loss_temporal,
        'total_aux': total_loss,
    }
    
    return total_loss, metrics


# ============================================================
# 用於 Bayesian Filter 的 Spectral Score
# ============================================================

@jax.jit
def temporal_spectral_score(
    x_true: jnp.ndarray,
    x_samples: jnp.ndarray,
    scale: float = 1.0,
) -> jnp.ndarray:
    """
    計算時間頻譜似然分數（用於 Bayesian Filter）
    
    使用 Laplace 分布計算頻譜匹配程度：
        p(x | z) ∝ exp(-|PSD(x) - PSD(x_true)| / scale)
    
    參數：
        x_true: 真實觀測序列 (batch, time, features)
        x_samples: 從模型採樣的序列 (n_samples, batch, time, features)
        scale: Laplace 分布的 scale 參數
    
    返回：
        log_scores: 對數似然分數 (n_samples, batch)
    """
    # 計算真實序列的 PSD
    psd_true = compute_temporal_spectrum(x_true)  # (batch, freq, features)
    
    def compute_score(x_sample):
        """計算單個樣本的分數"""
        psd_sample = compute_temporal_spectrum(x_sample)
        # L1 距離
        l1_dist = jnp.sum(jnp.abs(psd_sample - psd_true), axis=(-1, -2))  # (batch,)
        # Laplace log-likelihood
        log_score = -l1_dist / scale
        return log_score
    
    # 對所有樣本計算
    log_scores = jax.vmap(compute_score)(x_samples)  # (n_samples, batch)
    
    return log_scores


# ============================================================
# 測試
# ============================================================

def test_picabu_losses():
    """測試 PICABU 損失函數"""
    print("Testing SAVAR-PICABU Losses...")
    
    key = jax.random.PRNGKey(42)
    
    # 測試數據
    batch, d, d_x = 16, 1, 10
    tau = 5
    
    key, k1, k2 = jax.random.split(key, 3)
    y_true = jax.random.normal(k1, (batch, d, d_x))
    y_pred_mu = jax.random.normal(k2, (batch, d, d_x))
    logvar_decoder = jnp.zeros((d_x,))  # σ = 1
    
    # 1. 測試 CRPS
    print("\n1. CRPS Loss:")
    crps = crps_loss(y_true, y_pred_mu, logvar_decoder)
    print(f"   CRPS = {float(crps):.4f}")
    
    # 驗證：當預測完美時，CRPS 應該很小
    crps_perfect = crps_loss(y_true, y_true, logvar_decoder)
    print(f"   CRPS (perfect) = {float(crps_perfect):.4f}")
    assert crps_perfect < crps, "CRPS should be smaller for perfect prediction"
    print("   ✓ CRPS test passed")
    
    # 2. 測試 Temporal Spectral Loss
    print("\n2. Temporal Spectral Loss:")
    
    key, k1, k2 = jax.random.split(key, 3)
    x_history_true = jax.random.normal(k1, (batch, tau, d, d_x))
    x_history_pred = jax.random.normal(k2, (batch, tau, d, d_x))
    
    spec_loss = temporal_spectral_loss(
        x_history_true.reshape(batch, tau, -1),
        x_history_pred.reshape(batch, tau, -1)
    )
    print(f"   Temporal Spectral Loss = {float(spec_loss):.4f}")
    
    # 驗證：相同輸入應該損失為 0
    spec_loss_zero = temporal_spectral_loss(
        x_history_true.reshape(batch, tau, -1),
        x_history_true.reshape(batch, tau, -1)
    )
    print(f"   Temporal Spectral Loss (same) = {float(spec_loss_zero):.6f}")
    assert spec_loss_zero < 1e-5, "Spectral loss should be ~0 for identical inputs"
    print("   ✓ Temporal spectral test passed")
    
    # 3. 測試組合損失
    print("\n3. Combined PICABU Loss:")
    total_loss, metrics = picabu_auxiliary_loss(
        y_true, y_pred_mu, logvar_decoder,
        x_history_true, x_history_pred,
        coeff_crps=1.0, coeff_temporal=2000.0
    )
    print(f"   CRPS: {float(metrics['crps']):.4f}")
    print(f"   Temporal: {float(metrics['temporal_spectral']):.4f}")
    print(f"   Total: {float(total_loss):.4f}")
    print("   ✓ Combined loss test passed")
    
    # 4. 測試 JIT 編譯
    print("\n4. JIT Compilation:")
    import time
    
    # 第一次調用（編譯）
    t0 = time.time()
    _ = crps_loss(y_true, y_pred_mu, logvar_decoder)
    t1 = time.time()
    
    # 第二次調用（已編譯）
    _ = crps_loss(y_true, y_pred_mu, logvar_decoder)
    t2 = time.time()
    
    print(f"   First call (compile): {(t1-t0)*1000:.2f}ms")
    print(f"   Second call (cached): {(t2-t1)*1000:.2f}ms")
    print("   ✓ JIT compilation successful")
    
    # 5. 測試梯度
    print("\n5. Gradient Check:")
    
    def loss_fn(mu):
        return crps_loss(y_true, mu, logvar_decoder)
    
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(y_pred_mu)
    print(f"   Gradient shape: {grad.shape}")
    print(f"   Gradient norm: {float(jnp.linalg.norm(grad)):.4f}")
    print("   ✓ Gradient computation successful")
    
    print("\n" + "="*50)
    print("✓ All SAVAR-PICABU loss tests passed!")
    print("="*50)


if __name__ == "__main__":
    test_picabu_losses()