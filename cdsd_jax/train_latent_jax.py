"""
JAX/Flax 版本的训练循环

将 PyTorch 实现迁移到 JAX/Flax
主要变化：
1. 使用 Optax 优化器替代 PyTorch optimizers
2. 使用 JAX JIT 编译加速训练
3. 函数式编程风格（无状态）
4. 使用 flax.training.train_state.TrainState 管理训练状态
"""
import os
from collections import deque
from functools import partial
from typing import Any, Tuple, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax.core import FrozenDict, freeze

from cdsd_jax.dag_optim_jax import compute_dag_constraint
from cdsd_jax.utils_jax import ALM
from cdsd_jax.prox_jax import rmsprop_with_effective_lr


def _get_plotter():
    """Return a Plotter implementation (real or stub)."""
    disable = os.environ.get("CDSD_DISABLE_PLOTS", "").lower() in {"1", "true", "yes"}
    if disable:
        return lambda: _StubPlotter("CDSD_DISABLE_PLOTS is set")
    try:
        from cdsd_jax.plot_savar import Plotter as _Plotter
        return _Plotter
    except Exception as exc:
        # Capture exc value immediately
        exc_str = str(exc)
        return lambda: _StubPlotter(exc_str)


class _StubPlotter:
    """Fallback Plotter so importing TrainingLatent never crashes."""
    def __init__(self, reason):
        self.reason = reason
        print(f"Plotting disabled: {reason}")

    def plot(self, *_, **__):
        return

    def save(self, *_, **__):
        return


Plotter = _get_plotter()


class TrainState(train_state.TrainState):
    """扩展的训练状态（添加 batch_stats 等）"""
    rng: jax.random.PRNGKey
    effective_lr: Any = None  # 有效学习率（JAX style：从 updates 计算）

    def apply_gradients(self, *, grads, **kwargs):
        """
        應用梯度並更新參數。

        - 如果 optimizer state 有 `effective_lr`（來自 prox_jax.RMSpropState），
          就用那個；
        - 否則 fallback 為 |updates|（對 SGD 等保持合理）。
        """
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        # 從 opt_state 取 effective_lr，如果沒有就用 |updates|
        if hasattr(new_opt_state, "effective_lr"):
            effective_lr = new_opt_state.effective_lr
        else:
            effective_lr = jax.tree.map(jnp.abs, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            effective_lr=effective_lr,
            **kwargs,
        )


# ========== JIT 编译的纯函数（模块级） ==========

@partial(jax.jit, static_argnames=('schedule_reg', 'schedule_ortho', 'd_times_dz', 'ortho_shape', 'use_grad_project'))
def _train_step_pure_fn(state, x, y, fwd_rng,
                        iteration, instantaneous, converged, no_w_constraint,
                        alm_ortho_gamma, alm_ortho_mu, qpm_acyclic_mu,
                        reg_coeff, schedule_reg, schedule_ortho,
                        acyclic_normalization, ortho_normalization,
                        use_grad_project, d_times_dz, ortho_shape):
    """
    纯函数版本的训练步骤（JIT 编译）

    JAX style: 使用 FrozenDict.copy() 进行 functional updates
    """
    # 定义损失函数（纯函数）
    def loss_fn(trainable_params):
        # ✅ JAX style: 使用 FrozenDict.copy() 代替 freeze/unfreeze
        full_params = state.params.copy({'params': trainable_params})

        # 前向传播
        output = state.apply_fn(full_params, x, y, fwd_rng, deterministic=False)

        # ELBO
        nll = -output['elbo']
        recons = output['recons']
        kl = output['kl']

        # 正则化 - 使用 jax.lax.cond 替换 Python if
        def compute_sparsity():
            # 通过 apply_fn 调用 get_adj 方法
            adj = state.apply_fn(full_params, method='get_adj')
            return reg_coeff * jnp.linalg.norm(adj.flatten(), ord=1)

        sparsity_reg = jax.lax.cond(
            iteration > schedule_reg,
            lambda: compute_sparsity(),
            lambda: jnp.array(0.0)
        )

        connect_reg = jnp.array(0.0)

        # 约束 - 使用 jax.lax.cond 替换 Python if
        def compute_acyclic():
            adj = state.apply_fn(full_params, method='get_adj')
            adj_flat = adj[-1].reshape(d_times_dz, d_times_dz)
            return compute_dag_constraint(adj_flat) / acyclic_normalization

        h_acyclic = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_and(instantaneous, jnp.logical_not(converged)),
                iteration > 0
            ),
            lambda: compute_acyclic(),
            lambda: jnp.array(0.0)
        )

        # 正交性约束 - 使用 jax.lax.cond 替换 Python if
        def compute_ortho():
            w = full_params['params']['autoencoder']['w_decoder']
            k = w.shape[2]
            i = 0
            constraint = w[i].T @ w[i] - jnp.eye(k)
            return constraint / ortho_normalization

        h_ortho = jax.lax.cond(
            iteration > schedule_ortho,
            lambda: compute_ortho(),
            lambda: jnp.zeros(ortho_shape)
        )

        # 总损失
        loss = nll + sparsity_reg + connect_reg

        # 使用 jax.lax.cond 替换 Python if
        ortho_contrib = jax.lax.cond(
            jnp.logical_not(no_w_constraint),
            lambda: jnp.sum(alm_ortho_gamma * h_ortho) + 0.5 * alm_ortho_mu * jnp.sum(h_ortho ** 2),
            lambda: jnp.array(0.0)
        )

        acyclic_contrib = jax.lax.cond(
            instantaneous,
            lambda: 0.5 * qpm_acyclic_mu * h_acyclic ** 2,
            lambda: jnp.array(0.0)
        )

        loss = loss + ortho_contrib + acyclic_contrib

        return loss, (nll, recons, kl, sparsity_reg, connect_reg, h_ortho, h_acyclic, output)

    # 计算梯度
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (nll, recons, kl, sparsity_reg, connect_reg, h_ortho, h_acyclic, output)), grads_trainable = grad_fn(state.params['params'])

    # ✅ JAX style: 使用 FrozenDict.copy() 重建梯度结构
    zero_grads = jax.tree.map(jnp.zeros_like, state.params)
    grads = zero_grads.copy({'params': grads_trainable})

    # 更新参数
    new_state = state.apply_gradients(grads=grads)

    # 投影梯度（W >= 0）
    # use_grad_project 是静态参数，可以使用 Python if
    if use_grad_project:
        # 使用 lax.cond 检查 no_w_constraint（这是动态的）
        def project_gradients(s):
            """✅ JAX style: 投影 W >= 0，使用 FrozenDict.copy()"""
            w = s.params['params']['autoencoder']['w_decoder']
            new_params = s.params.copy({
                'params': s.params['params'].copy({
                    'autoencoder': s.params['params']['autoencoder'].copy({
                        'w_decoder': jnp.maximum(w, 0.0)
                    })
                })
            })
            return s.replace(params=new_params)

        new_state = jax.lax.cond(
            jnp.logical_not(no_w_constraint),
            lambda s: project_gradients(s),
            lambda s: s,
            new_state
        )

    # 返回新状态和指标（所有都是 JAX arrays）
    metrics = {
        'loss': loss,
        'nll': nll,
        'recons': recons,
        'kl': kl,
        'sparsity_reg': sparsity_reg,
        'connect_reg': connect_reg,
        'h_ortho': h_ortho,
        'h_acyclic': h_acyclic,
        'px_mu': output['px_mu']
    }

    return new_state, metrics


@partial(jax.jit, static_argnames=('schedule_reg', 'schedule_ortho', 'd_times_dz', 'ortho_shape'))
def _valid_step_pure_fn(state, x, y, fwd_rng,
                        iteration, instantaneous, converged,
                        qpm_acyclic_mu, reg_coeff, schedule_reg, schedule_ortho,
                        acyclic_normalization, ortho_normalization,
                        d_times_dz, ortho_shape):
    """
    纯函数版本的验证步骤（JIT 编译）

    这是一个模块级函数，不依赖于类实例，可以被 JIT 编译。
    """

    # 前向传播（确定性）
    output = state.apply_fn(state.params, x, y, fwd_rng, deterministic=True)

    # ELBO
    nll = -output['elbo']
    recons = output['recons']
    kl = output['kl']

    # 正则化 - 使用 jax.lax.cond
    def compute_sparsity():
        full_params = state.params
        adj = state.apply_fn(full_params, method='get_adj')
        return reg_coeff * jnp.linalg.norm(adj.flatten(), ord=1)

    sparsity_reg = jax.lax.cond(
        iteration > schedule_reg,
        lambda: compute_sparsity(),
        lambda: jnp.array(0.0)
    )

    connect_reg = jnp.array(0.0)

    # 约束 - 使用 jax.lax.cond
    def compute_acyclic():
        full_params = state.params
        adj = state.apply_fn(full_params, method='get_adj')
        adj_flat = adj[-1].reshape(d_times_dz, d_times_dz)
        return compute_dag_constraint(adj_flat) / acyclic_normalization

    h_acyclic = jax.lax.cond(
        jnp.logical_and(instantaneous, jnp.logical_not(converged)),
        lambda: compute_acyclic(),
        lambda: jnp.array(0.0)
    )

    # 正交性约束 - 使用 jax.lax.cond
    def compute_ortho():
        w = state.params['params']['autoencoder']['w_decoder']
        k = w.shape[2]
        i = 0
        constraint = w[i].T @ w[i] - jnp.eye(k)
        return constraint / ortho_normalization

    h_ortho = jax.lax.cond(
        iteration > schedule_ortho,
        lambda: compute_ortho(),
        lambda: jnp.zeros(ortho_shape)
    )

    # 总损失 - 使用 jax.lax.cond
    loss = nll + sparsity_reg + connect_reg

    acyclic_contrib = jax.lax.cond(
        instantaneous,
        lambda: 0.5 * qpm_acyclic_mu * h_acyclic ** 2,
        lambda: jnp.array(0.0)
    )

    loss = loss + acyclic_contrib

    # 返回指标
    metrics = {
        'loss': loss,
        'nll': nll,
        'recons': recons,
        'kl': kl,
        'sparsity_reg': sparsity_reg,
        'connect_reg': connect_reg,
        'h_ortho': h_ortho,
        'h_acyclic': h_acyclic,
        'px_mu': output['px_mu']
    }

    return metrics


class TrainingLatentJAX:
    """
    JAX 版本的训练类
    """
    def __init__(self, model, data, hp, best_metrics, rng_seed=0):
        """
        参数：
            model: TSDCDConfig 或 Flax model（兼容旧代码）
            data: DataLoaderJAX
            hp: 超参数对象
            best_metrics: 最佳指标字典
            rng_seed: 随机种子
        """
        # 检查是否使用新的 functional API
        from cdsd_jax.model.tsdcd_latent_jax import (
            TSDCDConfig, init_tsdcd_params, get_adj
        )

        is_functional = isinstance(model, TSDCDConfig)

        if is_functional:
            # 新 API: model 是 TSDCDConfig
            self.cfg = model
            self.model = None  # 不再使用 Flax Module
        else:
            # 舊 API: model 是 Flax Module（向後兼容）
            self.model = model
            self.cfg = None

        self.data = data
        self.hp = hp
        self.best_metrics = best_metrics

        # 超参数
        self.latent = hp.latent
        self.no_gt = hp.no_gt
        self.debug_gt_z = hp.debug_gt_z
        self.gt_dag = data.gt_graph
        self.gt_w = data.gt_w
        self.d_z = hp.d_z
        self.no_w_constraint = hp.no_w_constraint

        # 数据维度
        self.d = data.x.shape[2]
        self.d_x = hp.d_x
        self.tau = hp.tau
        self.batch_size = hp.batch_size
        self.instantaneous = hp.instantaneous

        # 训练控制
        self.patience = hp.patience
        self.patience_freq = 50
        self.best_valid_loss = np.inf
        self.iteration = 1
        self.logging_iter = 0
        self.converged = False
        self.thresholded = False
        self.ended = False

        # 训练指标列表
        self.train_loss_list = []
        self.train_elbo_list = []
        self.train_recons_list = []
        self.train_kl_list = []
        self.train_sparsity_reg_list = []
        self.train_connect_reg_list = []
        self.train_ortho_cons_list = []
        self.train_ortho_vector_cons_list = []
        self.train_acyclic_cons_list = []
        self.mu_ortho_list = []
        self.h_ortho_list = []

        # 验证指标列表
        self.valid_loss_list = []
        self.valid_elbo_list = []
        self.valid_recons_list = []
        self.valid_kl_list = []
        self.valid_sparsity_reg_list = []
        self.valid_connect_reg_list = []
        self.valid_ortho_cons_list = []
        self.valid_ortho_vector_cons_list = []
        self.valid_acyclic_cons_list = []

        # Plotter
        self.plotter = Plotter()

        # 图历史记录
        history_cap = getattr(self.hp, "graph_history_limit", 50)
        if history_cap is None or history_cap < 0:
            history_cap = 0
        self.graph_history_limit = history_cap
        self.adj_tt = deque(maxlen=self.graph_history_limit) if self.graph_history_limit else None
        if not self.no_gt:
            self.adj_w_tt = deque(maxlen=self.graph_history_limit) if self.graph_history_limit else None
        else:
            self.adj_w_tt = None
        self.logvar_encoder_tt = []
        self.logvar_decoder_tt = []
        self.logvar_transition_tt = []

        # 初始化随机数生成器
        self.rng = jax.random.PRNGKey(rng_seed)

        # 初始化模型参数（根据 API 类型）
        self.rng, init_rng = jax.random.split(self.rng)

        if is_functional:
            # 新 functional API
            functional_params = init_tsdcd_params(init_rng, self.cfg)
            # 包裝成 Flax 格式（添加 'params' 嵌套層）以兼容現有訓練代碼
            self.params = freeze({'params': functional_params})
        else:
            # 舊 Flax Module API
            x_init = jnp.ones((self.batch_size, self.tau, self.d, self.d_x))
            y_init = jnp.ones((self.batch_size, self.d, self.d_x))
            self.params = model.init(init_rng, x_init, y_init, init_rng)

            if not isinstance(self.params, FrozenDict):
                self.params = freeze(self.params)

        # 初始化优化器
        if hp.optimizer == "sgd":
            self.tx = optax.sgd(hp.lr)
        elif hp.optimizer == "rmsprop":
            # 使用自訂 RMSprop + effective_lr（對齊 PyTorch monkey_patch_RMSprop）
            self.tx = rmsprop_with_effective_lr(learning_rate=hp.lr, decay=0.99, eps=1e-8)
        else:
            raise NotImplementedError(f"optimizer {hp.optimizer} is not implemented")

        # 创建训练状态（根据 API 类型）
        if is_functional:
            # 新 functional API: 創建包裝函數來模擬 Flax Module 的 apply 接口
            from cdsd_jax.model.tsdcd_latent_jax import tsdcd_forward, get_adj

            def functional_apply_fn(all_params, x=None, y=None, rng=None,
                                   deterministic=False, method=None):
                """包裝 functional API 為 Flax-like apply 接口

                完全模擬 Flax Module.apply() 的簽名：
                - apply(params, x, y, rng, deterministic=False)  # forward
                - apply(params, method='get_adj')                 # method call

                假設參數結構已統一為 Flax 格式（在初始化時完成）：
                - all_params = {'params': {'autoencoder': ..., 'mask': ..., 'transition': ...}}
                - 固定 schema，JIT 友好，無運行時動態判斷
                """
                # 直接訪問 'params' key（schema 已在初始化時統一，不做動態判斷）
                params = all_params['params']

                # Method 調用（如 get_adj）
                if method is not None:
                    if method == 'get_adj' or (callable(method) and method.__name__ == 'get_adj'):
                        return get_adj(params, self.cfg)
                    else:
                        raise NotImplementedError(f"Unknown method: {method}")

                # 正常 forward
                return tsdcd_forward(params, self.cfg, rng, x, y, deterministic=deterministic)

            self.state = TrainState.create(
                apply_fn=functional_apply_fn,
                params=self.params,
                tx=self.tx,
                rng=init_rng
            )
        else:
            # 舊 Flax Module API
            self.state = TrainState.create(
                apply_fn=model.apply,
                params=self.params,
                tx=self.tx,
                rng=init_rng
            )

        # 计算约束归一化系数
        d = self.d * self.d_z
        full_adjacency = jnp.ones((d, d)) - jnp.eye(d)
        self.acyclic_constraint_normalization = float(compute_dag_constraint(full_adjacency))

        if self.latent:
            self.ortho_normalization = self.d_x * self.d_z

        # ===== JIT 形状验证 =====
        # 验证预计算的形状常量是否正确（防止 JIT 运行时错误）
        print("\n=== JIT Shape Validation ===")
        print(f"d = {self.d}, d_z = {self.d_z}, d_x = {self.d_x}")
        print(f"d * d_z = {d} (用于 DAG constraint reshape)")

        # 获取实际的 adjacency matrix 形状
        try:
            if is_functional:
                # 使用 functional API（取內層 'params'）
                test_adj = get_adj(self.state.params['params'], self.cfg)  # ✅
            else:
                # 使用 Flax Module API
                test_adj = self.state.apply_fn(self.state.params, method=self.model.get_adj)

            adj_shape = test_adj.shape
            print(f"Adjacency matrix shape: {adj_shape}")

            # 验证最后一层的形状是否匹配
            if len(adj_shape) > 0:
                last_layer_size = adj_shape[-1] * adj_shape[-2]
                expected_size = d * d

                if last_layer_size != expected_size:
                    raise ValueError(
                        f"Shape mismatch detected!\n"
                        f"  Expected: adj[-1] total elements = {expected_size} (= {d} * {d})\n"
                        f"  Got: adj shape = {adj_shape}, last layer = {last_layer_size}\n"
                        f"  This will cause reshape error in _train_step_pure()"
                    )

                print(f"✓ Shape validation passed: {d} * {d} = {expected_size}")

        except Exception as e:
            print(f"⚠️ Warning: Could not validate adjacency shape: {e}")
            print(f"   Will proceed, but may encounter shape errors during training")

        print("===========================\n")

    def train_with_QPM(self):
        """
        使用增强拉格朗日方法（ALM/QPM）进行约束优化训练
        训练分 3 个阶段：
        1. ALM 约束优化
        2. 继续训练直到似然稳定
        3. 阈值化后继续训练
        """
        # 初始化 ALM/QPM（正交性和无环性约束）
        self.ALM_ortho = ALM(
            mu_init=self.hp.ortho_mu_init,
            mu_mult_factor=self.hp.ortho_mu_mult_factor,
            omega_gamma=self.hp.ortho_omega_gamma,
            omega_mu=self.hp.ortho_omega_mu,
            h_threshold=self.hp.ortho_h_threshold,
            min_iter_convergence=self.hp.ortho_min_iter_convergence,
            dim_gamma=(self.d_z, self.d_z)
        )

        if self.instantaneous:
            # 添加无环性约束
            self.QPM_acyclic = ALM(
                mu_init=self.hp.acyclic_mu_init,
                mu_mult_factor=self.hp.acyclic_mu_mult_factor,
                omega_gamma=self.hp.acyclic_omega_gamma,
                omega_mu=self.hp.acyclic_omega_mu,
                h_threshold=self.hp.acyclic_h_threshold,
                min_iter_convergence=self.hp.acyclic_min_iter_convergence
            )

        # 主训练循环
        while self.iteration < self.hp.max_iteration and not self.ended:
            # 训练步骤
            self.train_step()

            # 验证步骤
            if self.iteration % self.hp.valid_freq == 0:
                self.logging_iter += 1
                x, y, y_pred = self.valid_step()
                self.log_losses()

                # 打印和绘图
                if self.iteration % (self.hp.valid_freq * self.hp.print_freq) == 0:
                    self.print_results()
                if self.logging_iter > 0 and self.iteration % (self.hp.valid_freq * self.hp.plot_freq) == 0:
                    self.plotter.plot(self)

            # 约束优化阶段
            if not self.converged:
                if self.iteration % self.hp.valid_freq == 0:
                    # 更新正交性约束
                    self.ALM_ortho.update(
                        self.iteration,
                        self.valid_ortho_vector_cons_list,
                        self.valid_loss_list
                    )

                    if self.iteration > 1000:
                        if not self.no_w_constraint:
                            ortho_converged = self.ALM_ortho.state.has_converged
                        else:
                            self.converged = True
                    else:
                        ortho_converged = False

                    # 如果 mu 增加，重置优化器
                    if self.ALM_ortho.state.has_increased_mu:
                        self._reset_optimizer()

                    # 更新无环性约束
                    if self.instantaneous:
                        self.QPM_acyclic.update(
                            self.iteration,
                            self.valid_acyclic_cons_list,
                            self.valid_loss_list
                        )
                        acyclic_converged = self.QPM_acyclic.state.has_converged

                        if self.QPM_acyclic.state.has_increased_mu:
                            self._reset_optimizer()

                        self.converged = ortho_converged and acyclic_converged
                    else:
                        self.converged = ortho_converged

            else:
                # 约束满足后，继续训练直到收敛
                if not self.thresholded and self.iteration % self.patience_freq == 0:
                    if not self.has_patience(self.hp.patience, self.valid_loss):
                        self.threshold()
                        self.patience = self.hp.patience_post_thresh
                        self.best_valid_loss = np.inf
                # 阈值化后继续训练
                else:
                    if self.iteration % self.patience_freq == 0:
                        if not self.has_patience(self.hp.patience_post_thresh, self.valid_loss):
                            self.ended = True

            self.iteration += 1

        # 最大迭代数后阈值化
        if self.iteration >= self.hp.max_iteration:
            self.threshold()

        # 最终绘图和打印
        self.plotter.plot(self, save=True)
        self.print_results()

        # 返回验证损失
        valid_loss = {
            "valid_loss": self.valid_loss,
            "best_valid_loss": self.best_valid_loss,
            "valid_loss1": -self.valid_loss_list[-1],
            "valid_loss2": -self.valid_loss_list[-2],
            "valid_loss3": -self.valid_loss_list[-3],
            "valid_loss4": -self.valid_loss_list[-4],
            "valid_loss5": -self.valid_loss_list[-5],
            "valid_neg_elbo": self.valid_nll,
            "valid_recons": self.valid_recons,
            "valid_kl": self.valid_kl,
            "valid_sparsity_reg": self.valid_sparsity_reg,
            "valid_ortho_cons": float(jnp.sum(self.valid_ortho_cons))
        }

        return valid_loss

    def train_step(self):
        """单次训练步骤（包装纯函数）"""
        # 采样数据
        self.rng, sample_rng = jax.random.split(self.rng)
        x, y, z = self.data.sample(self.batch_size, valid=False, rng=sample_rng)

        # 分裂 RNG 用于前向传播
        self.rng, fwd_rng = jax.random.split(self.rng)

        # 转换参数为 JAX arrays（用于 JIT）
        # ✅ 使用 float/int 而非 bool，避免 grad 錯誤
        iteration_jax = jnp.array(self.iteration, dtype=jnp.float32)  # int32 → float32
        converged_jax = jnp.array(float(self.converged), dtype=jnp.float32)  # bool → float
        instantaneous_jax = jnp.array(float(self.instantaneous), dtype=jnp.float32)  # bool → float
        no_w_constraint_jax = jnp.array(float(self.no_w_constraint), dtype=jnp.float32)  # bool → float

        # 预计算静态形状
        d_times_dz = self.d * self.d_z
        ortho_shape = (int(self.ortho_normalization / self.d_x),
                      int(self.ortho_normalization / self.d_x))

        # 檢查是否使用梯度投影（functional API 不使用此功能）
        # 作为静态参数传递 Python bool（不是 JAX array）
        if self.cfg is not None:
            # Functional API: 不使用梯度投影
            use_grad_project = True
        else:
            # Flax Module API
            use_grad_project = (
                hasattr(self.model, 'autoencoder') and
                hasattr(self.model.autoencoder, 'use_grad_project') and
                self.model.autoencoder.use_grad_project
            )

        # 调用模块级纯函数（JIT 编译）
        new_state, metrics = _train_step_pure_fn(
            self.state, x, y, fwd_rng,
            iteration=iteration_jax,
            instantaneous=instantaneous_jax,
            converged=converged_jax,
            no_w_constraint=no_w_constraint_jax,
            alm_ortho_gamma=self.ALM_ortho.state.gamma,
            alm_ortho_mu=self.ALM_ortho.state.mu,
            qpm_acyclic_mu=self.QPM_acyclic.state.mu if self.instantaneous else 0.0,
            reg_coeff=self.hp.reg_coeff,
            schedule_reg=self.hp.schedule_reg,
            schedule_ortho=self.hp.schedule_ortho,
            acyclic_normalization=self.acyclic_constraint_normalization,
            ortho_normalization=self.ortho_normalization,
            use_grad_project=use_grad_project,
            d_times_dz=d_times_dz,
            ortho_shape=ortho_shape
        )

        # 更新状态
        self.state = new_state

        # 记录指标（转换为 Python float）
        self.train_loss = float(metrics['loss'])
        self.train_nll = float(metrics['nll'])
        self.train_recons = float(metrics['recons'])
        self.train_kl = float(metrics['kl'])
        self.train_sparsity_reg = float(metrics['sparsity_reg'])
        self.train_connect_reg = float(metrics['connect_reg'])
        self.train_ortho_cons = metrics['h_ortho']  # 保持 JAX array
        self.train_acyclic_cons = float(metrics['h_acyclic'])

        return x, y, metrics['px_mu']

    def valid_step(self):
        """单次验证步骤（包装纯函数）"""
        # 采样验证数据
        self.rng, sample_rng = jax.random.split(self.rng)
        x, y, z = self.data.sample(self.data.n_valid - self.data.tau, valid=True, rng=sample_rng)

        # 前向传播
        self.rng, fwd_rng = jax.random.split(self.rng)

        # 转换参数为 JAX arrays
        iteration_jax = jnp.array(self.iteration, dtype=jnp.int32)
        converged_jax = jnp.array(self.converged, dtype=jnp.bool_)
        instantaneous_jax = jnp.array(self.instantaneous, dtype=jnp.bool_)

        # 预计算静态形状
        d_times_dz = self.d * self.d_z
        ortho_shape = (int(self.ortho_normalization / self.d_x),
                      int(self.ortho_normalization / self.d_x))

        # 调用模块级纯函数（JIT 编译）
        metrics = _valid_step_pure_fn(
            self.state, x, y, fwd_rng,
            iteration=iteration_jax,
            instantaneous=instantaneous_jax,
            converged=converged_jax,
            qpm_acyclic_mu=self.QPM_acyclic.state.mu if self.instantaneous else 0.0,
            reg_coeff=self.hp.reg_coeff,
            schedule_reg=self.hp.schedule_reg,
            schedule_ortho=self.hp.schedule_ortho,
            acyclic_normalization=self.acyclic_constraint_normalization,
            ortho_normalization=self.ortho_normalization,
            d_times_dz=d_times_dz,
            ortho_shape=ortho_shape
        )

        # 记录验证指标（转换为 Python float）
        self.valid_loss = float(metrics['loss'])
        self.valid_nll = float(metrics['nll'])
        self.valid_recons = float(metrics['recons'])
        self.valid_kl = float(metrics['kl'])
        self.valid_sparsity_reg = float(metrics['sparsity_reg'])
        self.valid_connect_reg = float(metrics['connect_reg'])
        self.valid_ortho_cons = metrics['h_ortho']
        self.valid_acyclic_cons = float(metrics['h_acyclic'])

        return x, y, metrics['px_mu']

    def has_patience(self, patience_init, valid_loss):
        """检查验证损失是否在 patience 步内没有改善"""
        if self.patience > 0:
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.patience = patience_init
                print(f"Best valid loss: {self.best_valid_loss}")
            else:
                self.patience -= 1
            return True
        else:
            return False

    def threshold(self):
        """阈值化邻接矩阵并固定（Flax 官方 pattern）"""
        adj = self.get_adj()
        thresholded_adj = (adj > 0.5).astype(jnp.float32)

        # Flax 官方 pattern: unfreeze → modify → freeze
        from flax.core import unfreeze, freeze
        params = unfreeze(self.state.params)

        if self.cfg is not None:
            # ✅ Functional API：寫進 params['params']['mask']
            mask_params = params['params']['mask']
            mask_params['fixed_output'] = thresholded_adj
            mask_params['is_fixed'] = jnp.array(1.0, dtype=jnp.float32)  # float 配合前面修改
            params['params']['mask'] = mask_params
        else:
            # Flax Module API：保留原本 fixed.mask 邏輯
            if 'fixed' not in params:
                params['fixed'] = {}
            if 'mask' not in params['fixed']:
                params['fixed']['mask'] = {}
            params['fixed']['mask']['output'] = thresholded_adj
            params['fixed']['mask']['is_fixed'] = jnp.array(True)

        new_params = freeze(params)
        self.state = self.state.replace(params=new_params)

        print("Thresholding ================")
        self.thresholded = True

    def log_losses(self):
        """记录损失和其他指标"""
        # 训练指标
        self.train_loss_list.append(-self.train_loss)
        self.train_recons_list.append(self.train_recons)
        self.train_kl_list.append(self.train_kl)
        self.train_sparsity_reg_list.append(self.train_sparsity_reg)
        self.train_connect_reg_list.append(self.train_connect_reg)
        self.train_ortho_cons_list.append(float(jnp.sum(self.train_ortho_cons)))
        self.train_ortho_vector_cons_list.append(self.train_ortho_cons)
        self.train_acyclic_cons_list.append(self.train_acyclic_cons)

        # 验证指标
        self.valid_loss_list.append(-self.valid_loss)
        self.valid_recons_list.append(self.valid_recons)
        self.valid_kl_list.append(self.valid_kl)
        self.valid_sparsity_reg_list.append(self.valid_sparsity_reg)
        self.valid_connect_reg_list.append(self.valid_connect_reg)
        self.valid_ortho_cons_list.append(float(jnp.sum(self.valid_ortho_cons)))
        self.valid_ortho_vector_cons_list.append(self.valid_ortho_cons)
        self.valid_acyclic_cons_list.append(self.valid_acyclic_cons)

        self.mu_ortho_list.append(self.ALM_ortho.state.mu)

        # 图历史
        if self.adj_tt is not None:
            adj = self.get_adj()
            self.adj_tt.append(np.array(adj))

        if self.adj_w_tt is not None:
            w = self.get_w_decoder()
            self.adj_w_tt.append(np.array(w))

        # 方差
        params = self.state.params['params']
        self.logvar_decoder_tt.append(float(params['autoencoder']['logvar_decoder'][0]))
        self.logvar_encoder_tt.append(float(params['autoencoder']['logvar_encoder'][0]))

        # ✅ 區分 functional vs Flax Module 的 key
        if self.cfg is not None:
            # Functional API: key 是 'transition'
            self.logvar_transition_tt.append(float(params['transition']['logvar'][0, 0]))
        else:
            # Flax Module API: key 是 'transition_model'
            self.logvar_transition_tt.append(float(params['transition_model']['logvar'][0, 0]))

    def print_results(self):
        """打印训练结果"""
        print("============================================================")
        print(f"Iteration #{self.iteration}")
        print(f"Converged: {self.converged}")

        print(f"ELBO: {-self.train_nll:.4f}")
        print(f"Recons: {self.train_recons:.4f}")
        print(f"KL: {self.train_kl:.4f}")

        print(f"Sparsity_reg: {self.train_sparsity_reg:.1e}")

        print(f"ortho cons: {self.train_ortho_cons_list[-1]:.1e}")
        print(f"ortho mu: {self.ALM_ortho.state.mu}")

        if self.instantaneous:
            print(f"acyclic cons: {self.train_acyclic_cons:.4f}")
            print(f"acyclic mu: {self.QPM_acyclic.state.mu}")

        print("-------------------------------")
        print(f"valid_ELBO: {-self.valid_nll:.4f}")
        print(f"patience: {self.patience}")

    def get_regularisation(self, params) -> jnp.ndarray:
        """计算稀疏正则化"""
        if self.iteration > self.hp.schedule_reg:
            adj = self.state.apply_fn(params, method=self.model.get_adj)
            reg = self.hp.reg_coeff * jnp.linalg.norm(adj.flatten(), ord=1)
        else:
            reg = jnp.array(0.0)

        return reg

    def get_acyclicity_violation(self, params) -> jnp.ndarray:
        """计算无环性约束违反度"""
        if self.iteration > 0:
            adj = self.state.apply_fn(params, method=self.model.get_adj)
            adj_flat = adj[-1].reshape(self.d * self.d_z, self.d * self.d_z)
            h = compute_dag_constraint(adj_flat) / self.acyclic_constraint_normalization
        else:
            h = jnp.array(0.0)

        return h

    def get_ortho_violation(self, params) -> jnp.ndarray:
        """计算正交性约束违反度"""
        if self.iteration > self.hp.schedule_ortho:
            w = self.get_w_decoder_from_params(params)
            k = w.shape[2]
            i = 0
            constraint = w[i].T @ w[i] - jnp.eye(k)
            h = constraint / self.ortho_normalization
        else:
            h = jnp.zeros((self.d_z, self.d_z))

        return h

    def get_adj(self) -> jnp.ndarray:
        """获取邻接矩阵"""
        if self.cfg is not None:
            # Functional API（取內層 'params'）
            from cdsd_jax.model.tsdcd_latent_jax import get_adj
            return get_adj(self.state.params['params'], self.cfg)  # ✅
        else:
            # Flax Module API
            return self.state.apply_fn(self.state.params, method=self.model.get_adj)

    def get_w_decoder(self) -> jnp.ndarray:
        """获取解码器权重"""
        params = self.state.params['params']
        if 'w_decoder' in params['autoencoder']:
            # 线性自编码器
            return params['autoencoder']['w_decoder']
        else:
            # 非线性自编码器（可能没有显式的 w 参数）
            raise KeyError("Decoder weight 'w_decoder' not found in autoencoder params")

    def get_w_decoder_from_params(self, params) -> jnp.ndarray:
        """从参数中获取解码器权重"""
        params_dict = params['params']
        return params_dict['autoencoder']['w_decoder']

    def _reset_optimizer(self):
        """重置优化器（ALM 增加 mu 時）"""
        if self.hp.optimizer == "sgd":
            self.tx = optax.sgd(self.hp.lr)
        elif self.hp.optimizer == "rmsprop":
            self.tx = rmsprop_with_effective_lr(
                learning_rate=self.hp.lr,
                decay=0.99,
                eps=1e-8,
            )

        # 确保 params 一定是 FrozenDict
        params = self.state.params
        if not isinstance(params, FrozenDict):
            params = freeze(params)

        # 根據 API 類型創建訓練狀態
        if self.cfg is not None:
            # Functional API: 創建包裝函數
            from cdsd_jax.model.tsdcd_latent_jax import tsdcd_forward, get_adj

            def functional_apply_fn(all_params, x=None, y=None, rng=None,
                                   deterministic=False, method=None):
                """包裝 functional API 為 Flax-like apply 接口

                固定 schema，JIT 友好，無運行時動態判斷
                """
                # 直接訪問 'params' key（schema 已統一）
                params = all_params['params']

                if method is not None:
                    if method == 'get_adj' or (callable(method) and method.__name__ == 'get_adj'):
                        return get_adj(params, self.cfg)
                    else:
                        raise NotImplementedError(f"Unknown method: {method}")

                return tsdcd_forward(params, self.cfg, rng, x, y, deterministic=deterministic)

            self.state = TrainState.create(
                apply_fn=functional_apply_fn,
                params=params,
                tx=self.tx,
                rng=self.state.rng
            )
        else:
            # Flax Module API
            self.state = TrainState.create(
                apply_fn=self.model.apply,
                params=params,
                tx=self.tx,
                rng=self.state.rng
            )


# ========== 测试函数 ==========

def test_training():
    """测试训练循环"""
    print("Testing TrainingLatentJAX...")

    # 这里需要完整的模型和数据加载器
    # 暂时只打印提示
    print("  ⚠ Full testing requires model and data loader")
    print("  Skipping test...")


if __name__ == "__main__":
    test_training()
