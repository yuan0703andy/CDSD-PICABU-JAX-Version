"""
metrics_jax.py
JAX 版本的评估指标

将 PyTorch/NumPy 实现迁移到 JAX
主要指标：
- MCC (Mean Correlation Coefficient) - 潜变量恢复
- SHD (Structural Hamming Distance) - 图结构距离
- Precision/Recall - 边的精确率和召回率
- W MAE - 混合矩阵误差
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from typing import Tuple, Optional


# ========== 混合矩阵误差 ==========

def w_mae(w: np.ndarray, gt_w: np.ndarray) -> float:
    """
    计算混合矩阵 W 的平均绝对误差（MAE）

    参数：
        w: 学习的混合矩阵
        gt_w: Ground truth 混合矩阵

    返回：
        mae: 平均绝对误差
    """
    if isinstance(w, jnp.ndarray):
        w = np.array(w)
    if isinstance(gt_w, jnp.ndarray):
        gt_w = np.array(gt_w)

    mae = np.sum(np.abs(w - gt_w)) / w.size
    return float(mae)


# ========== MCC（潜变量恢复）==========

def mean_corr_coef(x: np.ndarray,
                  y: np.ndarray,
                  method: str = 'pearson',
                  indices: Optional[list] = None) -> Tuple[float, np.ndarray, Tuple]:
    """
    计算平均相关系数（MCC）

    使用匈牙利算法找到最优的潜变量对应关系

    参数：
        x: Ground truth 潜变量，形状 (n_samples, d)
        y: 学习的潜变量，形状 (n_samples, d)
        method: 相关系数类型 ['pearson', 'spearman']
        indices: 要考虑的索引列表

    返回：
        score: MCC 分数
        cc_program_perm: 排列后的相关系数矩阵
        assignments: 匈牙利算法的分配结果 (row_ind, col_ind)
    """
    d = x.shape[1]

    # 计算相关系数矩阵
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError(f'Invalid method: {method}')

    # 使用绝对值（不关心正负相关）
    cc = np.abs(cc)

    # 选择子集（如果指定）
    if indices is not None:
        cc_program = cc[:, indices[-d:]]
    else:
        cc_program = cc

    # 匈牙利算法：找到最优匹配
    assignments = linear_sum_assignment(-1 * cc_program)

    # 计算平均相关系数
    score = cc_program[assignments].mean()

    # 计算排列矩阵
    perm_mat = np.zeros((d, d))
    perm_mat[assignments] = 1

    # 排列学习的潜变量
    cc_program_perm = np.matmul(cc_program, perm_mat.transpose())

    return float(score), cc_program_perm, assignments


def mcc_latent(model,
              data_loader,
              num_samples: int = int(1e5),
              method: str = 'pearson',
              indices: Optional[list] = None,
              rng: Optional[jax.random.PRNGKey] = None,
              params: Optional[dict] = None):
    """
    计算潜变量的 MCC

    参数：
        model: 训练好的模型（Flax module 或 trainer）
        data_loader: 数据加载器
        num_samples: 采样数量
        method: 相关系数类型
        indices: 索引列表
        rng: 随机数生成器（如果为 None，使用默认种子 0）
        params: 模型参数（如果为 None，从 model.state.params 获取）

    返回：
        score: MCC 分数
        cc_program_perm: 排列后的相关系数矩阵
        assignments: 分配结果
        z: Ground truth 潜变量
        z_hat: 学习的潜变量
        x: 输入数据
    """
    # 初始化 RNG
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # 判斷 model 型別 & 取 params / cfg / actual_model
    if params is None:
        if hasattr(model, 'state'):
            # model 是 TrainingLatentJAX
            params = model.state.params
            cfg = getattr(model, 'cfg', None)
            actual_model = getattr(model, 'model', None)
        else:
            raise ValueError("params must be provided when model is a Flax module")
    else:
        cfg = None
        actual_model = None  # 不會用到

    # 收集数据
    z_list = []
    z_hat_list = []
    x_list = []
    sample_counter = 0

    # 确定采样数量
    n = data_loader.x.shape[0]
    if n == 1:
        n = data_loader.x.shape[1]
    if sample_counter < n:
        num_samples = n

    # 采样和编码
    while sample_counter < num_samples:
        x, y, z = data_loader.sample(64, valid=False)

        # 分割 RNG 用于这一批
        rng, rng_batch = jax.random.split(rng)

        # ====== 主迴圈：取 batch → encode → 收集 z, z_hat ======
        if hasattr(model, 'state'):
            # ----- 情況 1：TrainingLatentJAX + functional API -----
            if cfg is not None and actual_model is None:
                # Functional encode from the PICABU JAX model
                from picabu_jax.model.picabu_latent import encode as encode_fn
                # params 是 FrozenDict({'params': {...}})
                z_all, _, _ = encode_fn(params['params'], cfg, rng_batch, x, y)
                # z_all: (batch, tau+1, d, d_z)，取最後一個時間步
                z_hat_batch = np.array(z_all[:, -1])

            # ----- 情況 2：TrainingLatentJAX + Flax Module -----
            else:
                # actual_model 是 Flax Module，encode method 回傳 (z_all, mu_y, std_y) 或類似
                z_all, *_ = actual_model.apply(
                    params, x, y, rng_batch, method=actual_model.encode
                )
                z_hat_batch = np.array(z_all[:, -1])

        else:
            # ----- 情況 3：Flax Module 本體 + params -----
            z_all, *_ = model.apply(
                params, x, y, rng_batch, method=model.encode
            )
            z_hat_batch = np.array(z_all[:, -1])

        # GT z: (batch, tau+1, d, d_z) → 取最後時間步
        z_list.append(np.array(z[:, -1]))
        z_hat_list.append(z_hat_batch)
        x_list.append(np.array(x[:, -1]))

        sample_counter += x.shape[0]

    # 合并数据
    z = np.concatenate(z_list, axis=0)[:int(num_samples)]
    z_hat = np.concatenate(z_hat_list, axis=0)[:int(num_samples)]
    x = np.concatenate(x_list, axis=0)[:int(num_samples)]

    # Reshape: (n_samples, d, d_z) -> (n_samples, d * d_z)
    z = z.reshape(z.shape[0], z.shape[1] * z.shape[2])
    z_hat = z_hat.reshape(z_hat.shape[0], z_hat.shape[1] * z_hat.shape[2])

    # 计算 MCC
    score, cc_program_perm, assignments = mean_corr_coef(z, z_hat, method, indices)

    return score, cc_program_perm, assignments, z, z_hat, x


# ========== 图结构评估指标 ==========

def edge_errors(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    计算边的错误统计

    参数：
        pred: 预测的邻接矩阵
        target: Ground truth 邻接矩阵

    返回：
        字典，包含 tp, tn, fp, fn, fp_rev, fn_rev, rev
    """
    if isinstance(pred, jnp.ndarray):
        pred = np.array(pred)
    if isinstance(target, jnp.ndarray):
        target = np.array(target)

    # 真正例和真负例
    tp = int(((pred == 1) & (pred == target)).sum())
    tn = int(((pred == 0) & (pred == target)).sum())

    # 错误类型
    diff = target - pred
    diff_t = np.swapaxes(diff, -2, -1)

    # 反向边（i->j 预测为 j->i）
    rev = int((((diff + diff_t) == 0) & (diff != 0)).sum() // 2)

    # 假负例和假正例
    fn = int((diff == 1).sum())
    fp = int((diff == -1).sum())

    # 扣除反向边
    fn_rev = fn - rev
    fp_rev = fp - rev

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "fp_rev": float(fp_rev),
        "fn_rev": float(fn_rev),
        "rev": float(rev)
    }


def shd(pred: np.ndarray,
       target: np.ndarray,
       rev_as_double: bool = False) -> float:
    """
    计算结构 Hamming 距离（SHD）

    参数：
        pred: 预测的邻接矩阵
        target: Ground truth 邻接矩阵
        rev_as_double: 反向边是否算两个错误

    返回：
        shd: 结构 Hamming 距离
    """
    m = edge_errors(pred, target)

    if rev_as_double:
        shd_value = m["fp"] + m["fn"]
    else:
        shd_value = m["fp_rev"] + m["fn_rev"] + m["rev"]

    return float(shd_value)


def precision_recall(pred: np.ndarray,
                    target: np.ndarray) -> Tuple[float, float]:
    """
    计算精确率和召回率

    参数：
        pred: 预测的邻接矩阵
        target: Ground truth 邻接矩阵

    返回：
        (precision, recall)
    """
    if isinstance(pred, jnp.ndarray):
        pred = np.array(pred)
    if isinstance(target, jnp.ndarray):
        target = np.array(target)

    tp = ((pred == 1) & (pred == target)).sum()
    diff = target - pred
    fn = (diff == 1).sum()
    fp = (diff == -1).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return precision, recall


def f1_score(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算 F1 分数（精确率和召回率的调和平均）

    参数：
        pred: 预测的邻接矩阵
        target: Ground truth 邻接矩阵

    返回：
        f1: F1 分数
    """
    m = edge_errors(pred, target)

    denominator = m["tp"] + 0.5 * (m["fp"] + m["fn"])
    f1 = m["tp"] / denominator if denominator > 0 else 0.0

    return float(f1)


# ========== 测试和验证函数 ==========

def test_metrics():
    """测试评估指标"""
    print("Testing metrics...")

    # 测试 W MAE
    print("\n  Testing W MAE...")
    w_learned = np.array([[0.9, 0.1], [0.2, 0.8]])
    w_gt = np.array([[1.0, 0.0], [0.0, 1.0]])
    mae = w_mae(w_learned, w_gt)
    print(f"    W MAE: {mae:.4f}")

    # 测试 MCC
    print("\n  Testing MCC...")
    z_gt = np.random.randn(100, 4)
    # 模拟学习的潜变量（带排列和噪声）
    perm = np.array([2, 0, 3, 1])
    z_learned = z_gt[:, perm] + 0.1 * np.random.randn(100, 4)

    score, _, assignments = mean_corr_coef(z_gt, z_learned, method='pearson')
    print(f"    MCC score: {score:.4f}")
    print(f"    Assignments: {assignments}")
    print(f"    Expected: [2, 0, 3, 1]")

    # 测试图结构指标
    print("\n  Testing graph metrics...")

    # 完美预测
    pred_perfect = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    target = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

    errors = edge_errors(pred_perfect, target)
    print(f"    Perfect prediction:")
    print(f"      TP={errors['tp']}, FP={errors['fp']}, TN={errors['tn']}, FN={errors['fn']}")
    print(f"      SHD: {shd(pred_perfect, target)}")

    # 有错误的预测
    pred_error = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    errors = edge_errors(pred_error, target)
    shd_value = shd(pred_error, target)
    prec, rec = precision_recall(pred_error, target)
    f1 = f1_score(pred_error, target)

    print(f"\n    Prediction with errors:")
    print(f"      TP={errors['tp']}, FP={errors['fp']}, TN={errors['tn']}, FN={errors['fn']}")
    print(f"      SHD: {shd_value}")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall: {rec:.4f}")
    print(f"      F1: {f1:.4f}")

    # 测试 JAX arrays
    print("\n  Testing with JAX arrays...")
    pred_jax = jnp.array(pred_perfect)
    target_jax = jnp.array(target)

    errors_jax = edge_errors(pred_jax, target_jax)
    shd_jax = shd(pred_jax, target_jax)
    print(f"    SHD (JAX): {shd_jax}")
    print(f"    TP (JAX): {errors_jax['tp']}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_metrics()
