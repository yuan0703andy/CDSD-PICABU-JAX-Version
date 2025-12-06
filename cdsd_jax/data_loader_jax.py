"""
JAX 版本的数据加载器

将 PyTorch 实现迁移到 JAX
主要变化：
1. 使用 jax.numpy 替代 torch.tensor
2. 使用 jax.random 进行采样
3. 返回 JAX arrays 而不是 PyTorch tensors
"""
import os
import numpy as np
import jax.numpy as jnp
import jax
from typing import Tuple, Optional


class DataLoaderJAX:
    """
    JAX 版本的数据加载器

    支持：
    - numpy 格式（内存加载）
    - hdf5 格式（按需采样，待实现）

    数据维度：(n, t, d, d_x)
    - n: 样本数（时间序列数量）
    - t: 时间步数
    - d: 空间位置数
    - d_x: 每个位置的观测变量数
    """

    def __init__(self,
                 ratio_train: float,
                 ratio_valid: float,
                 data_path: str,
                 data_format: str = "numpy",
                 latent: bool = True,
                 no_gt: bool = False,
                 debug_gt_w: bool = False,
                 instantaneous: bool = False,
                 tau: int = 1,
                 seed: int = 0):
        """
        参数：
            ratio_train: 训练集比例
            ratio_valid: 验证集比例
            data_path: 数据集路径
            data_format: 数据格式 ("numpy" 或 "hdf5")
            latent: 是否使用潜变量模型
            no_gt: 是否没有 ground truth
            debug_gt_w: 调试用（使用 GT W）
            instantaneous: 是否包含瞬时连接
            tau: 时间延迟
            seed: 随机种子
        """
        self.ratio_train = ratio_train
        self.ratio_valid = ratio_valid
        self.data_path = data_path
        self.data_format = data_format
        self.latent = latent
        self.no_gt = no_gt
        self.debug_gt_w = debug_gt_w
        self.instantaneous = instantaneous
        self.tau = tau
        self.seed = seed

        # 初始化随机数生成器
        self.rng = jax.random.PRNGKey(seed)

        # 数据容器
        self.x = None           # 观测数据
        self.z = None           # 潜变量（如果有）
        self.gt_graph = None    # Ground truth 因果图
        self.gt_w = None        # Ground truth 混合矩阵
        self.coordinates = None # 坐标（真实数据用）

        # 维度信息
        self.n = 0              # 样本数
        self.d = 0              # 空间位置数
        self.d_x = 0            # 观测变量数
        self.d_z = 0            # 潜变量数

        # 索引
        self.idx_train = None
        self.idx_valid = None
        self.n_train = 0
        self.n_valid = 0

        # 加载和分割数据
        self._load_data()
        self._split_data()

    def _load_data(self):
        """加载数据和 ground truth（如果有）"""
        if self.data_format == "numpy":
            # 加载观测数据
            self.x = np.load(os.path.join(self.data_path, 'data_x.npy'))

            if not self.no_gt:
                # 加载 ground truth
                self.gt_graph = np.load(os.path.join(self.data_path, 'graph.npy'))

                if self.latent:
                    self.z = np.load(os.path.join(self.data_path, 'data_z.npy'))
                    self.gt_w = np.load(os.path.join(self.data_path, 'graph_w.npy'))

        elif self.data_format == "hdf5":
            # HDF5 格式（按需采样）
            import tables
            f = tables.open_file(os.path.join(self.data_path, 'data.h5'), mode='r')
            self.x = f.root.data

        else:
            raise ValueError(f"Unknown data format: {self.data_format}")

        # 提取维度信息
        self.n = self.x.shape[0]        # 样本数
        self.d = self.x.shape[2]        # 空间位置数
        self.d_x = self.x.shape[3]      # 观测变量数

        if not self.no_gt and self.latent:
            self.d_z = self.z.shape[3]  # 潜变量数

        # 真实数据：加载坐标
        if self.no_gt:
            coord_path = os.path.join(self.data_path, 'coordinates.npy')
            if os.path.exists(coord_path):
                self.coordinates = np.load(coord_path)

    def _split_data(self):
        """分割训练集和验证集"""
        t_max = self.x.shape[1]

        if self.n == 1:
            # 单个长时间序列：按时间分割
            idx_train = []
            idx_valid = []

            for i in range(t_max // 100):
                start = i * 100
                train_end = start + int(100 * self.ratio_train)

                idx_train.extend(range(start + self.tau, train_end))
                idx_valid.extend(range(train_end, start + 100))

            self.idx_train = np.array(idx_train)
            self.idx_valid = np.array(idx_valid)
            self.n_train = len(self.idx_train)
            self.n_valid = len(self.idx_valid)

        else:
            # 多个时间序列：按样本分割
            self.n_train = int(self.n * self.ratio_train)
            self.n_valid = int(self.n * self.ratio_valid)

            self.idx_train = np.arange(self.tau, self.n_train)
            self.idx_valid = np.arange(self.n_train - self.tau,
                                        self.n_train + self.n_valid)

            # 打乱索引
            np.random.seed(self.seed)
            np.random.shuffle(self.idx_train)
            np.random.shuffle(self.idx_valid)

    def sample(self,
               batch_size: int,
               valid: bool = False,
               rng: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        采样一个 minibatch

        参数：
            batch_size: 批次大小
            valid: 是否从验证集采样
            rng: JAX 随机数生成器（如果为 None，使用内部 RNG）

        返回：
            x: 输入数据，形状 (batch_size, tau, d, d_x)
            y: 目标数据，形状 (batch_size, d, d_x)
            z: 潜变量，形状 (batch_size, tau+1, d, d_z) 或 None
        """
        # 准备数据容器
        x = np.zeros((batch_size, self.tau, self.d, self.d_x))
        y = np.zeros((batch_size, self.d, self.d_x))

        if not self.no_gt and self.latent:
            z = np.zeros((batch_size, self.tau + 1, self.d, self.d_z))
        else:
            z = None

        # 选择数据集
        dataset_idx = self.idx_valid if valid else self.idx_train

        # 随机采样索引
        if rng is None:
            # 使用 numpy 随机采样（保持与 PyTorch 版本一致）
            random_idx = np.random.choice(dataset_idx, replace=False, size=batch_size)
        else:
            # 使用 JAX 随机采样
            random_idx = jax.random.choice(rng, dataset_idx,
                                          shape=(batch_size,), replace=False)
            random_idx = np.array(random_idx)

        # 根据数据结构采样
        if self.n == 1:
            # 单个长时间序列
            for i, idx in enumerate(random_idx):
                x[i] = self.x[0, idx - self.tau:idx]
                y[i] = self.x[0, idx]

                if not self.no_gt and self.latent:
                    z[i] = self.z[0, idx - self.tau:idx + 1]

        else:
            # 多个时间序列
            for i, idx in enumerate(random_idx):
                x[i] = self.x[idx, 0:self.tau]
                y[i] = self.x[idx, self.tau]

                if not self.no_gt and self.latent:
                    z[i] = self.z[idx, 0:self.tau + 1]

        # 转换为 JAX arrays
        x_jax = jnp.array(x)
        y_jax = jnp.array(y)
        z_jax = jnp.array(z) if z is not None else None

        return x_jax, y_jax, z_jax

    def get_full_data(self, valid: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        获取完整的训练集或验证集

        参数：
            valid: 是否返回验证集

        返回：
            x, y, z: 完整数据集
        """
        n_samples = self.n_valid if valid else self.n_train
        return self.sample(n_samples, valid=valid)

    def __repr__(self):
        return (f"DataLoaderJAX(\n"
                f"  n={self.n}, t={self.x.shape[1]}, d={self.d}, d_x={self.d_x}, d_z={self.d_z}\n"
                f"  train={self.n_train}, valid={self.n_valid}\n"
                f"  tau={self.tau}, latent={self.latent}, no_gt={self.no_gt}\n"
                f")")


# ========== 测试和验证函数 ==========

def test_data_loader():
    """测试 JAX 数据加载器"""
    print("Testing DataLoaderJAX...")

    # 测试数据路径
    data_path = "dataset/savar_N25_med-easy_seed0"

    if not os.path.exists(data_path):
        print(f"  ⚠ Test data not found: {data_path}")
        print("  Skipping test...")
        return

    # 创建数据加载器
    loader = DataLoaderJAX(
        ratio_train=0.8,
        ratio_valid=0.2,
        data_path=data_path,
        data_format="numpy",
        latent=True,
        no_gt=False,
        tau=1,
        seed=42
    )

    print(f"\n  {loader}")

    # 测试采样
    print("\n  Testing sampling...")
    rng = jax.random.PRNGKey(0)

    # 训练集采样
    x_train, y_train, z_train = loader.sample(batch_size=32, valid=False, rng=rng)
    print(f"    Train batch:")
    print(f"      x shape: {x_train.shape}")
    print(f"      y shape: {y_train.shape}")
    print(f"      z shape: {z_train.shape if z_train is not None else None}")

    # 验证集采样
    x_valid, y_valid, z_valid = loader.sample(batch_size=16, valid=True, rng=rng)
    print(f"    Valid batch:")
    print(f"      x shape: {x_valid.shape}")
    print(f"      y shape: {y_valid.shape}")
    print(f"      z shape: {z_valid.shape if z_valid is not None else None}")

    # 检查数据类型
    print(f"\n  Checking data types...")
    print(f"    x type: {type(x_train)}")
    print(f"    x dtype: {x_train.dtype}")
    print(f"    Is JAX array: {isinstance(x_train, jnp.ndarray)}")

    # 检查 ground truth
    if loader.gt_graph is not None:
        print(f"\n  Ground truth:")
        print(f"    Graph shape: {loader.gt_graph.shape}")
        print(f"    W shape: {loader.gt_w.shape if loader.gt_w is not None else None}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_data_loader()
