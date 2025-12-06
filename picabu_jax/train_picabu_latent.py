"""
train_picabu_latent.py - SAVAR-PICABU 版本（完全對齊論文）

=== 關鍵修正 ===
1. target_edges 用真實 M（gt_graph.sum()）
2. sparsity 對整個 adj 做（SAVAR 沒有 instantaneous layer）
3. 訓練時只用 ALM soft penalty，不做 Top-K Prox
4. Top-K 只在 threshold 時用（後處理）

=== SAVAR 數據結構理解 ===
當 instantaneous=False, tau=1 時：
- adj.shape = (1, d_z, d_z)  # 只有 temporal，沒有 instantaneous
- gt_graph.shape = (1, d_z, d_z)  # 同樣
- adj[-1] = 整個圖（不是 instantaneous layer！）
"""
import os
from collections import deque
from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax.core import FrozenDict, freeze, unfreeze

from picabu_jax.dag_optim_jax import (
    compute_dag_constraint,
    SparsityALM,
    count_edges,
)
from picabu_jax.spectral_losses import crps_loss


# ============================================================
# Plotter
# ============================================================

def _get_plotter():
    disable = os.environ.get("CDSD_DISABLE_PLOTS", "").lower() in {"1", "true", "yes"}
    if disable:
        return lambda: _StubPlotter("disabled")
    try:
        from picabu_jax.plot_savar import Plotter as _Plotter
        return _Plotter
    except Exception as e:
        return lambda: _StubPlotter(str(e))


class _StubPlotter:
    def __init__(self, reason):
        print(f"Plotting disabled: {reason}")
    def plot(self, *_, **__): pass
    def save(self, *_, **__): pass


Plotter = _get_plotter()


# ============================================================
# ALM
# ============================================================

class ALM:
    def __init__(self, mu_init=1e4, mu_mult_factor=1.2, omega_gamma=0.01,
                 omega_mu=0.9, h_threshold=1e-4, min_iter_convergence=100,
                 dim_gamma=None, mu_max=1e6):  # 🔧 FIX: 添加 mu_max
        self.mu_init = mu_init
        self.mu = mu_init
        self.mu_mult_factor = mu_mult_factor
        self.mu_max = mu_max  # 🔧 FIX: μ 上限
        self.omega_gamma = omega_gamma
        self.omega_mu = omega_mu
        self.h_threshold = h_threshold
        self.min_iter_convergence = min_iter_convergence
        self.gamma = jnp.zeros(dim_gamma) if dim_gamma else 0.0
        self.has_converged = False
        self.has_increased_mu = False

    class State:
        def __init__(self, p):
            self.mu = p.mu
            self.gamma = p.gamma
            self.has_converged = p.has_converged
            self.has_increased_mu = p.has_increased_mu

    @property
    def state(self):
        return self.State(self)

    def update(self, iteration, h_list, loss_list):
        self.has_increased_mu = False
        if len(h_list) < 2:
            return
        curr = abs(h_list[-1]) if isinstance(h_list[-1], (int, float)) else float(jnp.sum(jnp.abs(h_list[-1])))
        prev = abs(h_list[-2]) if isinstance(h_list[-2], (int, float)) else float(jnp.sum(jnp.abs(h_list[-2])))
        if curr < self.h_threshold:
            self.has_converged = True
            return
        if iteration > self.min_iter_convergence:
            if (prev - curr) / (prev + 1e-8) < self.omega_mu:
                self.mu *= self.mu_mult_factor
                # 🔧 FIX: μ 上限檢查
                if self.mu > self.mu_max:
                    self.mu = self.mu_max
                self.has_increased_mu = True
        if isinstance(self.gamma, jnp.ndarray):
            self.gamma = self.gamma + self.omega_gamma * jnp.array(h_list[-1])
        else:
            self.gamma = self.gamma + self.omega_gamma * h_list[-1]


# ============================================================
# TrainState
# ============================================================

class TrainState(train_state.TrainState):
    rng: jax.random.PRNGKey
    effective_lr: Any = None

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        effective_lr = getattr(new_opt_state, "effective_lr", jax.tree.map(jnp.abs, updates))
        return self.replace(step=self.step + 1, params=new_params,
                           opt_state=new_opt_state, effective_lr=effective_lr, **kwargs)


# ============================================================
# JIT Train Step
# ============================================================

@partial(jax.jit, static_argnames=(
    'schedule_reg', 'schedule_ortho', 'd_times_dz', 'ortho_shape',
    'use_grad_project', 'use_auxiliary_losses', 'target_edges'
))
def _train_step_pure_fn(
    state, x, y, fwd_rng,
    iteration, instantaneous, converged, no_w_constraint,
    alm_ortho_gamma, alm_ortho_mu, qpm_acyclic_mu,
    sparsity_gamma, sparsity_mu,
    reg_coeff, schedule_reg, schedule_ortho,
    acyclic_normalization, ortho_normalization,
    use_grad_project, d_times_dz, ortho_shape,
    use_auxiliary_losses=True,
    coeff_crps=1.0,
    target_edges=10,
):
    """訓練步驟：純 ALM soft penalty，無硬投影"""

    def loss_fn(trainable_params):
        full_params = state.params.copy({'params': trainable_params})
        output = state.apply_fn(full_params, x, y, fwd_rng, deterministic=False)

        nll = -output['elbo']
        recons = output['recons']
        kl = output['kl']

        # PICABU auxiliary
        if use_auxiliary_losses:
            def compute_picabu():
                px_mu = output['px_mu']
                logvar = output.get('logvar_decoder', jnp.zeros(px_mu.shape[-1]))
                return coeff_crps * crps_loss(y, px_mu, logvar), crps_loss(y, px_mu, logvar)
            def no_picabu():
                return jnp.array(0.0), jnp.array(0.0)
            picabu_loss, crps_val = jax.lax.cond(iteration > schedule_reg, compute_picabu, no_picabu)
        else:
            picabu_loss, crps_val = jnp.array(0.0), jnp.array(0.0)

        # Sparsity: 對整個 adj 做（SAVAR 沒有 instantaneous）
        def compute_sparsity():
            adj = state.apply_fn(full_params, method='get_adj')
            n_edges = jnp.sum(adj)  # 整個 adj 的 soft sum
            h = n_edges - jnp.float32(target_edges)
            return sparsity_gamma * h + 0.5 * sparsity_mu * (h ** 2), h
        sparsity_loss, h_sparsity = jax.lax.cond(
            iteration > schedule_reg, compute_sparsity,
            lambda: (jnp.array(0.0), jnp.array(0.0))
        )

        # Ortho
        def compute_ortho():
            w = full_params['params']['autoencoder']['w_decoder']
            return (w[0].T @ w[0] - jnp.eye(w.shape[2])) / ortho_normalization
        h_ortho = jax.lax.cond(iteration > schedule_ortho, compute_ortho, lambda: jnp.zeros(ortho_shape))
        ortho_contrib = jax.lax.cond(
            jnp.logical_not(no_w_constraint),
            lambda: jnp.sum(alm_ortho_gamma * h_ortho) + 0.5 * alm_ortho_mu * jnp.sum(h_ortho ** 2),
            lambda: jnp.array(0.0)
        )

        # Acyclic（只對 instantaneous 有效）
        def compute_acyclic():
            adj = state.apply_fn(full_params, method='get_adj')
            return compute_dag_constraint(adj[-1].reshape(d_times_dz, d_times_dz)) / acyclic_normalization
        h_acyclic = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(instantaneous, jnp.logical_not(converged)), iteration > 0),
            compute_acyclic, lambda: jnp.array(0.0)
        )
        acyclic_contrib = jax.lax.cond(instantaneous, lambda: 0.5 * qpm_acyclic_mu * h_acyclic ** 2, lambda: jnp.array(0.0))

        loss = nll + picabu_loss + sparsity_loss + ortho_contrib + acyclic_contrib
        return loss, (nll, recons, kl, h_ortho, h_acyclic, h_sparsity, crps_val, output)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads_trainable = grad_fn(state.params['params'])
    nll, recons, kl, h_ortho, h_acyclic, h_sparsity, crps_val, output = aux

    zero_grads = jax.tree.map(jnp.zeros_like, state.params)
    grads = zero_grads.copy({'params': grads_trainable})
    new_state = state.apply_gradients(grads=grads)

    # W >= 0
    if use_grad_project:
        def project(s):
            w = s.params['params']['autoencoder']['w_decoder']
            new_p = s.params.copy({'params': s.params['params'].copy({
                'autoencoder': s.params['params']['autoencoder'].copy({'w_decoder': jnp.maximum(w, 0.0)})
            })})
            return s.replace(params=new_p)
        new_state = jax.lax.cond(jnp.logical_not(no_w_constraint), project, lambda s: s, new_state)

    return new_state, {
        'loss': loss, 'nll': nll, 'recons': recons, 'kl': kl,
        'h_ortho': h_ortho, 'h_acyclic': h_acyclic, 'h_sparsity': h_sparsity,
        'crps': crps_val, 'px_mu': output['px_mu'],
    }


# ============================================================
# JIT Valid Step
# ============================================================

@partial(jax.jit, static_argnames=(
    'schedule_reg', 'schedule_ortho', 'd_times_dz', 'ortho_shape',
    'use_auxiliary_losses', 'target_edges'
))
def _valid_step_pure_fn(
    state, x, y, fwd_rng,
    iteration, instantaneous, converged,
    qpm_acyclic_mu,
    reg_coeff, schedule_reg, schedule_ortho,
    acyclic_normalization, ortho_normalization,
    d_times_dz, ortho_shape,
    use_auxiliary_losses=True,
    coeff_crps=1.0,
    target_edges=10,
):
    output = state.apply_fn(state.params, x, y, fwd_rng, deterministic=True)
    nll = -output['elbo']
    recons = output['recons']
    kl = output['kl']

    if use_auxiliary_losses:
        px_mu = output['px_mu']
        logvar = output.get('logvar_decoder', jnp.zeros(px_mu.shape[-1]))
        crps_val = crps_loss(y, px_mu, logvar)
    else:
        crps_val = jnp.array(0.0)

    # Sparsity: hard count，對整個 adj
    def compute_sparsity():
        adj = state.apply_fn(state.params, method='get_adj')
        n = count_edges(adj, threshold=0.5)
        return jnp.float32(n) - jnp.float32(target_edges)
    h_sparsity = jax.lax.cond(iteration > schedule_reg, compute_sparsity, lambda: jnp.array(0.0))

    def compute_ortho():
        w = state.params['params']['autoencoder']['w_decoder']
        return (w[0].T @ w[0] - jnp.eye(w.shape[2])) / ortho_normalization
    h_ortho = jax.lax.cond(iteration > schedule_ortho, compute_ortho, lambda: jnp.zeros(ortho_shape))

    def compute_acyclic():
        adj = state.apply_fn(state.params, method='get_adj')
        return compute_dag_constraint(adj[-1].reshape(d_times_dz, d_times_dz)) / acyclic_normalization
    h_acyclic = jax.lax.cond(
        jnp.logical_and(instantaneous, jnp.logical_not(converged)),
        compute_acyclic, lambda: jnp.array(0.0)
    )

    return {
        'loss': nll, 'nll': nll, 'recons': recons, 'kl': kl,
        'h_ortho': h_ortho, 'h_acyclic': h_acyclic, 'h_sparsity': h_sparsity,
        'crps': crps_val, 'px_mu': output['px_mu'],
    }


# ============================================================
# Training Class
# ============================================================

class TrainingLatentJAX:
    """SAVAR-PICABU 訓練類"""

    def __init__(self, model, data, hp, best_metrics, rng_seed=0):
        from picabu_jax.model.picabu_latent import TSDCDConfig, init_tsdcd_params, get_adj

        is_functional = isinstance(model, TSDCDConfig)
        self.cfg = model if is_functional else None
        self.model = None if is_functional else model
        self.data = data
        self.hp = hp
        self.best_metrics = best_metrics

        # 基本參數
        self.latent = hp.latent
        self.no_gt = hp.no_gt
        self.gt_dag = data.gt_graph
        self.gt_w = data.gt_w
        self.d_z = hp.d_z
        self.no_w_constraint = hp.no_w_constraint
        self.d = data.x.shape[2]
        self.d_x = hp.d_x
        self.tau = hp.tau
        self.batch_size = hp.batch_size
        self.instantaneous = hp.instantaneous

        # PICABU
        self.use_auxiliary_losses = getattr(hp, 'use_auxiliary_losses', True)
        self.coeff_crps = getattr(hp, 'coeff_crps', 1.0)

        # 🔧 關鍵：target_edges 用真實 M
        if hasattr(hp, 'target_edges') and hp.target_edges is not None:
            self.target_edges = hp.target_edges
        elif self.gt_dag is not None:
            self.target_edges = int(np.array(self.gt_dag).sum())
        else:
            ratio = getattr(hp, 'sparsity_ratio', 2.0)
            self.target_edges = int(ratio * self.d_z)

        print(f"\n=== SAVAR-PICABU Configuration ===")
        print(f"  instantaneous: {self.instantaneous}")
        print(f"  tau: {self.tau}")
        print(f"  target_edges: {self.target_edges} (from gt_graph.sum())")
        print(f"  Training: ALM soft penalty only")
        print(f"===================================\n")

        # 訓練控制
        self.patience = hp.patience
        self.patience_freq = 50
        self.best_valid_loss = np.inf
        self.iteration = 1
        self.logging_iter = 0
        self.converged = False
        self.thresholded = False
        self.ended = False

        # 指標列表
        self.train_loss_list, self.train_recons_list, self.train_kl_list = [], [], []
        self.train_ortho_cons_list, self.train_ortho_vector_cons_list = [], []
        self.train_acyclic_cons_list, self.train_crps_list, self.train_sparsity_list = [], [], []
        self.valid_loss_list, self.valid_recons_list, self.valid_kl_list = [], [], []
        self.valid_ortho_cons_list, self.valid_ortho_vector_cons_list = [], []
        self.valid_acyclic_cons_list, self.valid_crps_list, self.valid_sparsity_list = [], [], []
        self.mu_ortho_list = []
        self.logvar_encoder_tt, self.logvar_decoder_tt, self.logvar_transition_tt = [], [], []

        # Graph history
        history_cap = getattr(hp, "graph_history_limit", 50) or 0
        self.graph_history_limit = history_cap
        self.adj_tt = deque(maxlen=history_cap) if history_cap else None
        self.adj_w_tt = deque(maxlen=history_cap) if history_cap and not self.no_gt else None

        self.plotter = Plotter()

        # 初始化
        self.rng = jax.random.PRNGKey(rng_seed)
        self.rng, init_rng = jax.random.split(self.rng)

        if is_functional:
            self.params = freeze({'params': init_tsdcd_params(init_rng, self.cfg)})
        else:
            x_init = jnp.ones((self.batch_size, self.tau, self.d, self.d_x))
            y_init = jnp.ones((self.batch_size, self.d, self.d_x))
            self.params = freeze(model.init(init_rng, x_init, y_init, init_rng))

        # Optimizer
        if hp.optimizer == "sgd":
            self.tx = optax.sgd(hp.lr)
        elif hp.optimizer == "rmsprop":
            self.tx = optax.rmsprop(hp.lr, decay=0.99, eps=1e-8)
        else:
            self.tx = optax.adam(hp.lr)

        # Apply function
        if is_functional:
            def apply_fn(all_params, x=None, y=None, rng=None, deterministic=False, method=None):
                params = all_params['params']
                if method == 'get_adj' or (callable(method) and method.__name__ == 'get_adj'):
                    return get_adj(params, self.cfg)
                from picabu_jax.model.picabu_latent import tsdcd_forward
                return tsdcd_forward(params, self.cfg, rng, x, y, deterministic=deterministic)
            self.state = TrainState.create(apply_fn=apply_fn, params=self.params, tx=self.tx, rng=init_rng)
        else:
            self.state = TrainState.create(apply_fn=model.apply, params=self.params, tx=self.tx, rng=init_rng)

        # 正規化
        d = self.d * self.d_z
        self.acyclic_constraint_normalization = float(compute_dag_constraint(jnp.ones((d, d)) - jnp.eye(d)))
        self.ortho_normalization = self.d_x * self.d_z if self.latent else 1

        # 驗證
        print(f"=== Validation ===")
        print(f"  adj shape: {self.get_adj().shape}")
        print(f"  target_edges: {self.target_edges}")
        print(f"==================\n")

    def train_with_QPM(self):
        """訓練主循環"""
        self.ALM_ortho = ALM(
            mu_init=self.hp.ortho_mu_init, mu_mult_factor=self.hp.ortho_mu_mult_factor,
            omega_gamma=self.hp.ortho_omega_gamma, omega_mu=self.hp.ortho_omega_mu,
            h_threshold=self.hp.ortho_h_threshold, min_iter_convergence=self.hp.ortho_min_iter_convergence,
            dim_gamma=(self.d_z, self.d_z),
            mu_max=getattr(self.hp, 'ortho_mu_max', 1e6),  # 🔧 FIX: μ 上限
        )
        self.ALM_sparsity = SparsityALM(
            target_edges=self.target_edges,
            mu_init=getattr(self.hp, 'sparsity_mu_init', 0.1),
            mu_multiplier=getattr(self.hp, 'sparsity_mu_mult', 1.2),
            threshold=1,
            mu_max=getattr(self.hp, 'sparsity_mu_max', 1e4),  # 🔧 FIX: μ 上限
        )
        if self.instantaneous:
            self.QPM_acyclic = ALM(
                mu_init=self.hp.acyclic_mu_init, mu_mult_factor=self.hp.acyclic_mu_mult_factor,
                omega_gamma=self.hp.acyclic_omega_gamma, omega_mu=self.hp.acyclic_omega_mu,
                h_threshold=self.hp.acyclic_h_threshold, min_iter_convergence=self.hp.acyclic_min_iter_convergence,
                mu_max=getattr(self.hp, 'acyclic_mu_max', 1e6),  # 🔧 FIX: μ 上限
            )

        while self.iteration < self.hp.max_iteration and not self.ended:
            self.train_step()

            if self.iteration % self.hp.valid_freq == 0:
                self.logging_iter += 1
                self.valid_step()
                self.log_losses()

                if self.iteration % (self.hp.valid_freq * self.hp.print_freq) == 0:
                    self.print_results()
                    self.plotter.plot(self)

                if not self.converged:
                    self.ALM_ortho.update(self.iteration, self.valid_ortho_cons_list, self.valid_loss_list)
                    ortho_conv = self.hp.no_w_constraint or self.ALM_ortho.state.has_converged
                    if self.ALM_ortho.state.has_increased_mu:
                        self._reset_optimizer()

                    self.ALM_sparsity.update(self.iteration, self.valid_sparsity_list, self.valid_loss_list)
                    if self.ALM_sparsity.has_increased_mu:
                        self._reset_optimizer()

                    if self.instantaneous:
                        self.QPM_acyclic.update(self.iteration, self.valid_acyclic_cons_list, self.valid_loss_list)
                        if self.QPM_acyclic.state.has_increased_mu:
                            self._reset_optimizer()
                        self.converged = ortho_conv and self.QPM_acyclic.state.has_converged
                    else:
                        self.converged = ortho_conv
                else:
                    if not self.thresholded and self.iteration % self.patience_freq == 0:
                        if not self.has_patience(self.hp.patience, self.valid_loss):
                            self.threshold()
                            self.patience = self.hp.patience_post_thresh
                            self.best_valid_loss = np.inf
                    elif self.thresholded and self.iteration % self.patience_freq == 0:
                        if not self.has_patience(self.hp.patience_post_thresh, self.valid_loss):
                            self.ended = True

            self.iteration += 1

        if not self.thresholded:
            self.threshold()

        self.plotter.plot(self, save=True)
        self.print_results()
        return {"valid_loss": self.valid_loss, "best_valid_loss": self.best_valid_loss}

    def train_step(self):
        self.rng, sample_rng, fwd_rng = jax.random.split(self.rng, 3)
        x, y, _ = self.data.sample(self.batch_size, valid=False, rng=sample_rng)

        d_times_dz = self.d * self.d_z
        ortho_shape = (self.d_z, self.d_z)

        new_state, metrics = _train_step_pure_fn(
            self.state, x, y, fwd_rng,
            jnp.float32(self.iteration), jnp.float32(self.instantaneous),
            jnp.float32(self.converged), jnp.float32(self.no_w_constraint),
            self.ALM_ortho.state.gamma, self.ALM_ortho.state.mu,
            self.QPM_acyclic.state.mu if self.instantaneous else 0.0,
            self.ALM_sparsity.gamma, self.ALM_sparsity.mu,
            self.hp.reg_coeff, self.hp.schedule_reg, self.hp.schedule_ortho,
            self.acyclic_constraint_normalization, self.ortho_normalization,
            self.cfg is not None, d_times_dz, ortho_shape,
            self.use_auxiliary_losses, self.coeff_crps, self.target_edges,
        )
        self.state = new_state

        self.train_loss = float(metrics['loss'])
        self.train_nll = float(metrics['nll'])
        self.train_recons = float(metrics['recons'])
        self.train_kl = float(metrics['kl'])
        self.train_ortho_cons = metrics['h_ortho']
        self.train_acyclic_cons = float(metrics['h_acyclic'])
        self.train_crps = float(metrics['crps'])
        self.train_sparsity = float(metrics['h_sparsity'])

        return x, y, metrics['px_mu']  # 🔧 FIX: 添加返回值

    def valid_step(self):
        self.rng, sample_rng, fwd_rng = jax.random.split(self.rng, 3)
        x, y, _ = self.data.sample(self.data.n_valid - self.data.tau, valid=True, rng=sample_rng)

        d_times_dz = self.d * self.d_z
        ortho_shape = (self.d_z, self.d_z)

        metrics = _valid_step_pure_fn(
            self.state, x, y, fwd_rng,
            jnp.int32(self.iteration), jnp.bool_(self.instantaneous), jnp.bool_(self.converged),
            self.QPM_acyclic.state.mu if self.instantaneous else 0.0,
            self.hp.reg_coeff, self.hp.schedule_reg, self.hp.schedule_ortho,
            self.acyclic_constraint_normalization, self.ortho_normalization,
            d_times_dz, ortho_shape,
            self.use_auxiliary_losses, self.coeff_crps, self.target_edges,
        )

        self.valid_loss = float(metrics['loss'])
        self.valid_nll = float(metrics['nll'])
        self.valid_recons = float(metrics['recons'])
        self.valid_kl = float(metrics['kl'])
        self.valid_ortho_cons = metrics['h_ortho']
        self.valid_acyclic_cons = float(metrics['h_acyclic'])
        self.valid_crps = float(metrics['crps'])
        self.valid_sparsity = float(metrics['h_sparsity'])

        return x, y, metrics['px_mu']  # 🔧 FIX: 添加返回值

    def log_losses(self):
        self.train_loss_list.append(-self.train_loss)
        self.train_recons_list.append(self.train_recons)
        self.train_kl_list.append(self.train_kl)
        self.train_ortho_cons_list.append(float(jnp.sum(self.train_ortho_cons)))
        self.train_ortho_vector_cons_list.append(self.train_ortho_cons)
        self.train_acyclic_cons_list.append(self.train_acyclic_cons)
        self.train_crps_list.append(self.train_crps)
        self.train_sparsity_list.append(self.train_sparsity)

        self.valid_loss_list.append(-self.valid_loss)
        self.valid_recons_list.append(self.valid_recons)
        self.valid_kl_list.append(self.valid_kl)
        self.valid_ortho_cons_list.append(float(jnp.sum(self.valid_ortho_cons)))
        self.valid_ortho_vector_cons_list.append(self.valid_ortho_cons)
        self.valid_acyclic_cons_list.append(self.valid_acyclic_cons)
        self.valid_crps_list.append(self.valid_crps)
        self.valid_sparsity_list.append(self.valid_sparsity)

        self.mu_ortho_list.append(self.ALM_ortho.state.mu)

        if self.adj_tt is not None:
            self.adj_tt.append(np.array(self.get_adj()))
        if self.adj_w_tt is not None:
            self.adj_w_tt.append(np.array(self.get_w_decoder()))

        params = self.state.params['params']
        self.logvar_decoder_tt.append(float(params['autoencoder']['logvar_decoder'][0]))
        self.logvar_encoder_tt.append(float(params['autoencoder']['logvar_encoder'][0]))
        trans_key = 'transition' if self.cfg else 'transition_model'
        self.logvar_transition_tt.append(float(params[trans_key]['logvar'][0, 0]))

    def print_results(self):
        print("=" * 60)
        print(f"Iteration #{self.iteration} | Converged: {self.converged} | Thresholded: {self.thresholded}")
        print(f"ELBO: {-self.train_nll:.4f} | Recons: {self.train_recons:.4f} | KL: {self.train_kl:.4f}")
        print(f"Sparsity: {self.train_sparsity:.1f} (target={self.target_edges}) | μ={self.ALM_sparsity.mu:.2f}")
        print(f"Ortho: {self.train_ortho_cons_list[-1]:.1e} | Acyclic: {self.train_acyclic_cons:.4f}")
        print(f"Valid ELBO: {-self.valid_nll:.4f} | Patience: {self.patience}")
        print("=" * 60)

    def has_patience(self, patience_init, valid_loss):
        if self.patience > 0:
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.patience = patience_init
            else:
                self.patience -= 1
            return True
        return False

    def threshold(self, method='topk'):
        """閾值化（後處理）"""
        adj = self.get_adj()

        if method == 'topk':
            flat = np.array(adj.flatten())
            k = min(self.target_edges, flat.size)
            thr = float(np.sort(flat)[::-1][k - 1]) if k > 0 else 1.0
            print(f"Top-{k} threshold: {thr:.4f}")
        else:
            thr = 0.5

        adj_bin = (adj >= thr).astype(jnp.float32)
        n_edges = int(jnp.sum(adj_bin))
        print(f"Thresholding: {n_edges} edges (target={self.target_edges})")

        params = unfreeze(self.state.params)
        if self.cfg:
            params['params']['mask']['fixed_output'] = adj_bin
            params['params']['mask']['is_fixed'] = jnp.array(1.0)
        self.state = self.state.replace(params=freeze(params))
        self.thresholded = True
        return adj_bin

    def get_adj(self):
        if self.cfg:
            from picabu_jax.model.picabu_latent import get_adj
            return get_adj(self.state.params['params'], self.cfg)
        return self.state.apply_fn(self.state.params, method=self.model.get_adj)

    def get_w_decoder(self):
        return self.state.params['params']['autoencoder']['w_decoder']

    def get_w_decoder_from_params(self, params):
        """從指定的 params 獲取 w_decoder"""
        return params['params']['autoencoder']['w_decoder']

    def get_regularisation(self, params=None) -> jnp.ndarray:
        """計算正則化項"""
        if params is None:
            params = self.state.params
        leaves = jax.tree_util.tree_leaves(params)
        return sum(jnp.sum(leaf ** 2) for leaf in leaves)

    def get_acyclicity_violation(self, params=None) -> jnp.ndarray:
        """計算無環性違反"""
        if params is None:
            params = self.state.params
        adj = self.get_adj()
        adj_flat = adj[-1].reshape(self.d * self.d_z, self.d * self.d_z)
        return compute_dag_constraint(adj_flat)

    def get_ortho_violation(self, params=None) -> jnp.ndarray:
        """計算正交性違反"""
        if params is None:
            params = self.state.params
        w = params['params']['autoencoder']['w_decoder']
        return w[0].T @ w[0] - jnp.eye(w.shape[2])

    def _reset_optimizer(self):
        if self.hp.optimizer == "sgd":
            self.tx = optax.sgd(self.hp.lr)
        elif self.hp.optimizer == "rmsprop":
            self.tx = optax.rmsprop(self.hp.lr, decay=0.99, eps=1e-8)
        else:
            self.tx = optax.adam(self.hp.lr)

        params = self.state.params
        if self.cfg:
            from picabu_jax.model.picabu_latent import tsdcd_forward, get_adj
            def apply_fn(all_params, x=None, y=None, rng=None, deterministic=False, method=None):
                p = all_params['params']
                if method == 'get_adj' or (callable(method) and method.__name__ == 'get_adj'):
                    return get_adj(p, self.cfg)
                return tsdcd_forward(p, self.cfg, rng, x, y, deterministic=deterministic)
            self.state = TrainState.create(apply_fn=apply_fn, params=params, tx=self.tx, rng=self.state.rng)
        else:
            self.state = TrainState.create(apply_fn=self.model.apply, params=params, tx=self.tx, rng=self.state.rng)


if __name__ == "__main__":
    print("SAVAR-PICABU Training (對齊論文)")
    print("關鍵：target_edges = gt_graph.sum()，sparsity 對整個 adj")