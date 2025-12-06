# CDSD-JAX

**CDSD 的 JAX/Flax 重構版本**

## 📂 目錄結構

```
cdsd-jax/
├── model/
│   ├── __init__.py
│   ├── tsdcd_latent.py              (660 行) - PyTorch 原版
│   └── tsdcd_latent-jax.py          (待創建) - JAX/Flax 版本
│
├── dataset/
│   ├── savar_N25_med-easy_seed0/    - 主要訓練數據
│   ├── savar_N4_easy_seed0/         - 小規模測試數據
│   └── savar_benchmark/             - 完整 benchmark
│       ├── N4_easy/   (5 seeds)
│       ├── N4_med-easy/
│       ├── N4_med-hard/
│       ├── N4_hard/
│       ├── N25_easy/
│       ├── N25_med-easy/
│       ├── N25_med-hard/
│       ├── N25_hard/
│       ├── N100_easy/
│       ├── N100_med-easy/
│       ├── N100_med-hard/
│       └── N100_hard/
│
├── __init__.py
├── train_latent.py                  (433 行) - PyTorch 原版
├── train_latent-jax.py              (待創建) - JAX 版本
├── data_loader.py                   (152 行) - PyTorch 原版
├── data_loader-jax.py               (待創建) - JAX 版本
├── metrics.py                       (218 行) - PyTorch 原版
├── metrics-jax.py                   (待創建) - JAX 版本
├── dag_optim.py                     (62 行) - PyTorch 原版
├── dag_optim-jax.py                 (待創建) - JAX 版本
├── prox.py                          (60 行) - PyTorch 原版
├── prox-jax.py                      (待創建) - Optax 版本
├── utils.py                         (100 行) - PyTorch 原版
├── utils-jax.py                     (待創建) - JAX 版本
├── plot_savar.py                    (300 行) - 可視化
└── README.md                        (本文件)
```

## 🎯 核心文件說明

### 1. 模型架構 (660 行)
**文件**: `model/tsdcd_latent.py`

包含的類：
- `LatentTSDCD` - 主模型
- `Autoencoder` - VAE 架構
- `Encoder` - 編碼器 (X → Z)
- `Decoder` - 解碼器 (Z → X)
- `Transition` - 時間序列轉移模型
- `MLP` - 多層感知器
- `Mask` - Gumbel-Softmax mask

### 2. 訓練循環 (433 行)
**文件**: `train_latent.py`

包含的類/函數：
- `TrainingLatent` - 訓練器類
- `train_with_QPM()` - 主訓練循環（ALM/QPM 約束優化）
- `train_step()` - 單步訓練
- `valid_step()` - 驗證
- `get_ortho_violation()` - 正交性約束
- `threshold()` - 圖阈值化

### 3. 數據加載 (152 行)
**文件**: `data_loader.py`

包含的類：
- `DataLoader` - 數據加載器
  - 支持 numpy 格式
  - 訓練/驗證集分割
  - 批次採樣

### 4. 評估指標 (218 行)
**文件**: `metrics.py`

包含的函數：
- `mcc_latent()` - 潛變量恢復準確度
- `shd()` - 結構 Hamming 距離
- `precision_recall()` - 精確率/召回率
- `edge_errors()` - TP/FP/TN/FN
- `w_mae()` - 混合矩陣誤差

### 5. DAG 約束 (62 行)
**文件**: `dag_optim.py`

包含的函數：
- `compute_dag_constraint()` - 計算 h(A) = tr(exp(A)) - d

### 6. 自定義優化器 (60 行)
**文件**: `prox.py`

包含的函數：
- `monkey_patch_RMSprop()` - 返回 effective learning rates 的 RMSprop

### 7. ALM 算法 (100 行)
**文件**: `utils.py`

包含的類：
- `ALM` - 增強拉格朗日方法

### 8. 可視化 (300 行)
**文件**: `plot_savar.py`

包含的類：
- `Plotter` - 繪圖器（簡化版，適用於 SAVAR）

## 📊 數據集說明

### SAVAR 數據格式
每個數據集包含：
- `data_x.npy` - 觀測數據 (T, N, d_x)
- `data_z.npy` - 潛變量數據 (T, N, d_z)
- `graph.npy` - 因果圖 (tau+1, d_z, d_z)
- `graph_w.npy` - 混合矩陣 W (tau+1, d_x, d_z)
- `data_params.json` - 數據參數
- `best_metrics.json` - 最佳性能基準

### Benchmark 結構
- **N4**: 4 個潛變量
- **N25**: 25 個潛變量
- **N100**: 100 個潛變量
- **難度**: easy, med-easy, med-hard, hard
- **Seeds**: 每個配置 5 個隨機種子 (seed_0 到 seed_4)

## 🔄 JAX 重構計劃

### 階段 1：底層工具 (3 天)
1. `dag_optim-jax.py` (0.5 天)
2. `data_loader-jax.py` (0.5 天)
3. `utils-jax.py` (ALM) (1 天)
4. `prox-jax.py` (Optax) (0.5 天)
5. `metrics-jax.py` (1 天)

### 階段 2：核心模型 (4-5 天)
6. `model/tsdcd_latent-jax.py` (4-5 天)

### 階段 3：訓練循環 (3-4 天)
7. `train_latent-jax.py` (3-4 天)

### 階段 4：測試 & 可視化 (2-3 天)
8. 端到端測試
9. 數值對比（PyTorch vs JAX）
10. 可視化（可選）

**總計**: 10-15 工作日

## 🚀 使用方式

### PyTorch 版本（當前）
```python
from cdsd_jax.model.tsdcd_latent import LatentTSDCD
from cdsd_jax.train_latent import TrainingLatent
from cdsd_jax.data_loader import DataLoader

# 標準訓練流程
data_loader = DataLoader(...)
model = LatentTSDCD(...)
trainer = TrainingLatent(model, data_loader, hp, best_metrics)
trainer.train_with_QPM()
```

### JAX 版本（重構後）
```python
from cdsd_jax.model.tsdcd_latent_jax import LatentTSDCD
from cdsd_jax.train_latent_jax import train_with_qpm
from cdsd_jax.data_loader_jax import load_savar_data

# JAX 訓練流程
data = load_savar_data(...)
model = LatentTSDCD(...)
params = model.init(rng, ...)
params, history = train_with_qpm(model, params, data, hp)
```

## 📝 命名規範

- **PyTorch 原版**: `filename.py`
- **JAX 重構版**: `filename-jax.py`

這樣可以：
- ✅ 保留原始代碼作為參考
- ✅ 清楚區分版本
- ✅ 方便對比測試

## 🔗 與主項目的關係

- **cdsd-jax/**: 獨立的重構項目，包含所有需要的代碼和數據
- **cdsd/**: 原始 PyTorch 代碼（保留作為參考）
- **train_savar_complete_v2.ipynb**: 可以使用兩個版本：
  - 引用 `cdsd-jax/` 中的 PyTorch 文件（與原版功能相同）
  - 後續引用 `-jax.py` 文件進行 JAX 訓練

## 📦 依賴

### PyTorch 版本
```bash
torch>=2.0.0
numpy
matplotlib
scipy
```

### JAX 版本（待實現）
```bash
jax
jaxlib
flax
optax
numpy
matplotlib
scipy
```

## 🎯 下一步

1. 從最簡單的模塊開始：`dag_optim-jax.py`
2. 逐步重構其他模塊
3. 保持與 PyTorch 版本的數值一致性
4. 完整的測試和驗證

---

**創建日期**: 2025-12-02
**基於**: CDSD PyTorch 實現
**目標**: JAX/Flax 重構，提升性能和可維護性
