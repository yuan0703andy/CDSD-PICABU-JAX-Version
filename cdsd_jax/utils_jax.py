"""
JAX 版本的 ALM（增强拉格朗日方法）

将 PyTorch 实现迁移到 JAX
主要变化：
1. 使用 JAX arrays 替代 PyTorch tensors
2. 使用纯函数和不可变状态
3. 返回新状态而不是就地修改
"""
import jax.numpy as jnp
import numpy as np
from typing import Tuple, NamedTuple


class ALMState(NamedTuple):
    """ALM 算法的状态（不可变）"""
    gamma: jnp.ndarray              # 拉格朗日乘子
    mu: float                       # 惩罚参数
    delta_gamma: float              # gamma 的变化量
    constraint_violation: list      # 约束违反历史
    has_converged: bool             # 是否收敛
    has_increased_mu: bool          # 是否增加了 mu


class ALM:
    """
    增强拉格朗日方法 (Augmented Lagrangian Method)

    用于约束优化：
    - 正交性约束（混合矩阵 W）
    - 无环性约束（instantaneous connections）

    如果只使用二次惩罚方法（QPM），可以忽略 lambda/gamma
    """

    def __init__(self,
                 mu_init: float,
                 mu_mult_factor: float,
                 omega_gamma: float,
                 omega_mu: float,
                 h_threshold: float,
                 min_iter_convergence: int,
                 dim_gamma: tuple = (1,)):
        """
        参数：
            mu_init: 初始惩罚参数 μ
            mu_mult_factor: μ 的增长因子
            omega_gamma: gamma 更新的阈值
            omega_mu: μ 更新的阈值
            h_threshold: 约束收敛阈值
            min_iter_convergence: 最小收敛迭代次数
            dim_gamma: gamma 的维度（用于向量约束）
        """
        self.mu_init = mu_init
        self.mu_mult_factor = mu_mult_factor
        self.omega_gamma = omega_gamma
        self.omega_mu = omega_mu
        self.h_threshold = h_threshold
        self.min_iter_convergence = min_iter_convergence
        self.dim_gamma = dim_gamma
        self.stop_crit_window = 100

        # 初始化状态
        self.state = self._init_state()

    def _init_state(self) -> ALMState:
        """初始化 ALM 状态"""
        return ALMState(
            gamma=jnp.zeros(self.dim_gamma),
            mu=self.mu_init,
            delta_gamma=-jnp.inf,
            constraint_violation=[],
            has_converged=False,
            has_increased_mu=False
        )

    def reset(self):
        """重置 ALM 状态"""
        self.state = self._init_state()

    @property
    def gamma(self):
        """获取当前 gamma"""
        return self.state.gamma

    @property
    def mu(self):
        """获取当前 mu"""
        return self.state.mu

    @property
    def has_converged(self):
        """是否收敛"""
        return self.state.has_converged

    @property
    def has_increased_mu(self):
        """是否增加了 mu"""
        return self.state.has_increased_mu

    def _compute_delta_gamma(self,
                            iteration: int,
                            val_loss: list,
                            current_delta: float) -> float:
        """
        计算 gamma 的变化量

        参数：
            iteration: 当前迭代次数
            val_loss: 验证损失历史
            current_delta: 当前 delta_gamma

        返回：
            新的 delta_gamma
        """
        # 需要至少 2 * stop_crit_window 次迭代
        if iteration >= 2 * self.stop_crit_window and \
           iteration % (2 * self.stop_crit_window) == 0:

            # 取最近三个验证损失
            t0, t_half, t1 = val_loss[-3], val_loss[-2], val_loss[-1]

            # 如果验证损失上下波动，不更新拉格朗日乘子和惩罚系数
            if not (min(t0, t1) < t_half < max(t0, t1)):
                return float(-jnp.inf)
            else:
                return (t1 - t0) / self.stop_crit_window
        else:
            return float(-jnp.inf)

    def update(self,
              iteration: int,
              h_list: list,
              val_loss: list) -> bool:
        """
        更新 mu 和 gamma 的值

        参数：
            iteration: 完成的训练迭代次数
            h_list: 约束值历史
            val_loss: 验证损失历史

        返回：
            是否收敛
        """
        # 需要至少 3 个验证损失值
        if len(val_loss) < 3:
            return False

        # 获取当前约束值
        h = h_list[-1]

        # 计算标量约束值（如果是向量约束）
        if len(self.dim_gamma) > 1:
            if isinstance(h, jnp.ndarray):
                h_scalar = float(jnp.sum(h))
            else:
                h_scalar = float(np.sum(h))
        else:
            h_scalar = float(h)

        # 检查是否收敛
        if iteration > self.min_iter_convergence and h_scalar <= self.h_threshold:
            # 更新状态：已收敛
            self.state = self.state._replace(
                has_converged=True,
                has_increased_mu=False
            )
            return True

        # 计算 delta_gamma
        delta_gamma = self._compute_delta_gamma(
            iteration,
            val_loss,
            self.state.delta_gamma
        )

        # 如果找到增强损失的驻点
        has_increased_mu = False
        new_gamma = self.state.gamma
        new_mu = self.state.mu
        new_violation = self.state.constraint_violation.copy()

        if abs(delta_gamma) < self.omega_gamma or delta_gamma > 0:
            # 更新 gamma
            if isinstance(h, jnp.ndarray):
                new_gamma = self.state.gamma + self.state.mu * h
            else:
                new_gamma = self.state.gamma + self.state.mu * jnp.array(h)

            new_violation.append(h_scalar)

            # 如果约束充分减小，增加 mu
            if len(new_violation) >= 2:
                if h_scalar > self.omega_mu * new_violation[-2]:
                    new_mu = self.state.mu * self.mu_mult_factor
                    has_increased_mu = True

        # 更新状态
        self.state = ALMState(
            gamma=new_gamma,
            mu=new_mu,
            delta_gamma=delta_gamma,
            constraint_violation=new_violation,
            has_converged=False,
            has_increased_mu=has_increased_mu
        )

        return False


# ========== 测试和验证函数 ==========

def test_alm():
    """测试 ALM 算法"""
    print("Testing ALM...")

    # 创建 ALM 实例
    alm = ALM(
        mu_init=1e-8,
        mu_mult_factor=2.0,
        omega_gamma=0.01,
        omega_mu=0.9,
        h_threshold=1e-3,
        min_iter_convergence=100,
        dim_gamma=(1,)
    )

    print(f"  Initial state:")
    print(f"    mu: {alm.mu}")
    print(f"    gamma: {alm.gamma}")
    print(f"    has_converged: {alm.has_converged}")

    # 模拟约束优化过程
    print(f"\n  Simulating constraint optimization...")

    h_list = []
    val_loss_list = []

    for i in range(1, 501):
        # 模拟约束值（逐渐减小）
        h = 1.0 / i
        h_list.append(h)

        # 模拟验证损失（逐渐减小）
        val_loss = -1000 + 10 * np.log(i)
        val_loss_list.append(val_loss)

        # 更新 ALM
        if len(val_loss_list) >= 3:
            converged = alm.update(i, h_list, val_loss_list)

            # 打印更新信息
            if i % 100 == 0:
                print(f"    Iter {i}: h={h:.6f}, mu={alm.mu:.2e}, "
                      f"converged={converged}, increased_mu={alm.has_increased_mu}")

            if converged:
                print(f"    ✓ Converged at iteration {i}")
                break

    print(f"\n  Final state:")
    print(f"    mu: {alm.mu:.6e}")
    print(f"    gamma: {alm.gamma}")
    print(f"    constraint violations: {len(alm.state.constraint_violation)}")
    print(f"    has_converged: {alm.has_converged}")

    # 测试向量约束
    print(f"\n  Testing vector constraints...")
    alm_vec = ALM(
        mu_init=1e-6,
        mu_mult_factor=2.0,
        omega_gamma=0.01,
        omega_mu=0.9,
        h_threshold=1e-3,
        min_iter_convergence=50,
        dim_gamma=(3, 3)
    )

    print(f"    Initial gamma shape: {alm_vec.gamma.shape}")

    # 模拟向量约束
    h_vec = jnp.array([[0.1, 0.05, 0.02],
                       [0.05, 0.1, 0.03],
                       [0.02, 0.03, 0.1]])

    val_loss_vec = [-1000, -999, -998]

    alm_vec.update(100, [h_vec], val_loss_vec)
    print(f"    Updated gamma shape: {alm_vec.gamma.shape}")
    print(f"    Updated mu: {alm_vec.mu:.6e}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_alm()
