"""
简化版 Plotter - 专门为 SAVAR 合成数据设计
移除了所有地图绘制功能，只保留核心训练监控和评估功能
"""
import os
import json
import numpy as np


def moving_average(a: np.ndarray, n: int = 10):
    """计算移动平均"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class Plotter:
    """简化的绘图器 - 延迟导入 matplotlib，避免 Jupyter crash"""

    def __init__(self):
        self.mcc = []
        self.assignments = []
        self._plt = None
        self._sns = None
        self._mcc_latent = None

    def _ensure_plotting_libs(self):
        """只在需要时才导入绘图库"""
        if self._plt is None:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式 backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            self._plt = plt
            self._sns = sns

    def _ensure_mcc_latent(self):
        """只在需要时才导入 mcc_latent"""
        if self._mcc_latent is None:
            # JAX 版本：从 picabu_jax.metrics_jax 导入
            from picabu_jax.metrics_jax import mcc_latent
            self._mcc_latent = mcc_latent

    def save(self, learner):
        """保存模型权重和图结构"""
        if learner.latent:
            # JAX 版本：直接获取并转换为 numpy
            w_decoder = np.array(learner.get_w_decoder())
            np.save(os.path.join(learner.hp.exp_path, "w_decoder"), w_decoder)

            # JAX 版本：需要从 params 中获取 w_encoder
            params = learner.state.params['params']
            if 'w_encoder' in params['autoencoder']:
                w_encoder = np.array(params['autoencoder']['w_encoder'])
            else:
                # tied weights: w_encoder = w_decoder.T
                w_encoder = np.transpose(w_decoder, (0, 2, 1))
            np.save(os.path.join(learner.hp.exp_path, "w_encoder"), w_encoder)

            adj = np.array(learner.get_adj())
            np.save(os.path.join(learner.hp.exp_path, "graphs"), adj)

    def plot(self, learner, save=False):
        """主绘图函数 - 简化版，只画核心曲线"""
        if save:
            self.save(learner)

        # 确保绘图库已导入
        self._ensure_plotting_libs()
        plt = self._plt

        # 绘制训练曲线
        if learner.latent:
            self._plot_learning_curves(
                train_loss=learner.train_loss_list,
                train_recons=learner.train_recons_list,
                train_kl=learner.train_kl_list,
                valid_loss=learner.valid_loss_list,
                valid_recons=learner.valid_recons_list,
                valid_kl=learner.valid_kl_list,
                best_metrics=learner.best_metrics,
                path=learner.hp.exp_path
            )

            # 绘制惩罚项
            losses = [
                {"name": "sparsity", "data": learner.train_sparsity_reg_list},
                {"name": "ortho_cons", "data": learner.train_ortho_cons_list},
                {"name": "mu_ortho", "data": learner.mu_ortho_list},
            ]
            self._plot_penalties(losses, learner.hp.exp_path)

        # 绘制邻接矩阵对比
        # JAX 版本：直接获取并转换为 numpy
        adj = np.array(learner.get_adj())

        if not learner.no_gt and learner.latent:
            # 计算 MCC 和潜变量对应关系
            self._ensure_mcc_latent()
            # JAX 版本：直接获取并转换为 numpy
            adj_w = np.array(learner.get_w_decoder())

            if learner.debug_gt_z:
                gt_dag = learner.gt_dag
                gt_w = learner.gt_w
                self.mcc.append(1.)
                assignments = np.arange(learner.gt_dag.shape[1])
            else:
                # JAX 版本：傳遞 learner（trainer）而不是 model
                score, _, assignments, _, _, _ = self._mcc_latent(learner, learner.data)
                permutation = np.zeros((learner.gt_dag.shape[1], learner.gt_dag.shape[1]))
                permutation[np.arange(learner.gt_dag.shape[1]), assignments[1]] = 1
                # JAX 版本：score 已經是 Python float 或 numpy scalar
                self.mcc.append(float(score))
                assignments = assignments[1]

                gt_dag = permutation.T @ learner.gt_dag @ permutation
                gt_w = learner.gt_w
                adj_w = adj_w[:, :, assignments]

            # 保存 MCC
            self._save_mcc(learner.hp.exp_path)

            # 绘制图结构对比
            self._plot_adjacency(adj, gt_dag, learner.hp.exp_path, 'transition')
            self._plot_adjacency_w(adj_w, gt_w, learner.hp.exp_path, 'decoder_w')
        else:
            self._plot_adjacency(adj, None, learner.hp.exp_path, 'transition', no_gt=True)

    def _plot_learning_curves(self, train_loss, train_recons, train_kl,
                             valid_loss, valid_recons, valid_kl, best_metrics, path):
        """绘制训练曲线"""
        plt = self._plt

        start = 1
        t_loss = moving_average(train_loss[start:])
        v_loss = moving_average(valid_loss[start:])
        t_recons = moving_average(train_recons[start:])
        t_kl = moving_average(train_kl[start:])

        plt.figure(figsize=(10, 6))
        plt.plot(v_loss, label="Valid ELBO", color="green", linewidth=2)
        plt.plot(t_loss, label="Train ELBO", color="purple", alpha=0.7)
        plt.plot(t_recons, label="Train Recons", color="blue", alpha=0.7)
        plt.plot(t_kl, label="Train KL", color="red", alpha=0.7)

        # 画基准线
        if best_metrics:
            plt.axhline(y=best_metrics.get("elbo", 0), color='purple', linestyle='dotted', alpha=0.5)
            plt.axhline(y=best_metrics.get("recons", 0), color='blue', linestyle='dotted', alpha=0.5)
            plt.axhline(y=best_metrics.get("kl", 0), color='red', linestyle='dotted', alpha=0.5)

        plt.title("Training Curves")
        plt.xlabel("Validation Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'training_loss.png'), dpi=150)
        plt.close()

    def _plot_penalties(self, losses, path):
        """绘制惩罚项曲线"""
        plt = self._plt

        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.set_yscale("log")

        for loss in losses:
            if len(loss["data"]) > 1:
                smoothed = moving_average(loss["data"][1:])
                plt.plot(smoothed, label=loss["name"])

        plt.title("Penalties and Constraints")
        plt.xlabel("Validation Steps")
        plt.ylabel("Value (log scale)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'penalties.png'), dpi=150)
        plt.close()

    def _plot_adjacency(self, learned, gt, path, name_suffix, no_gt=False):
        """绘制邻接矩阵对比"""
        plt = self._plt
        sns = self._sns

        tau = learned.shape[0]

        if no_gt:
            nrows = 1
        else:
            nrows = 3

        fig, axes = plt.subplots(nrows, tau, figsize=(tau*4, nrows*3))
        if nrows == 1:
            axes = [axes] if tau == 1 else axes

        for i in range(tau):
            if no_gt:
                ax = axes[i] if tau > 1 else axes[0]
                sns.heatmap(learned[tau-i-1], ax=ax, cmap="Blues",
                           vmin=0, vmax=1, cbar=True, square=True)
                ax.set_title(f"Learned (t-{i+1})")
            else:
                # Learned
                ax = axes[0, i] if tau > 1 else axes[0]
                sns.heatmap(learned[tau-i-1], ax=ax, cmap="Blues",
                           vmin=0, vmax=1, cbar=True, square=True)
                ax.set_title(f"Learned (t-{i+1})")

                # GT
                ax = axes[1, i] if tau > 1 else axes[1]
                sns.heatmap(gt[tau-i-1], ax=ax, cmap="Blues",
                           vmin=-1, vmax=1, cbar=True, square=True)
                ax.set_title(f"GT (t-{i+1})")

                # Difference
                ax = axes[2, i] if tau > 1 else axes[2]
                diff = learned[tau-i-1] - gt[tau-i-1]
                sns.heatmap(diff, ax=ax, cmap="RdBu_r",
                           vmin=-1, vmax=1, cbar=True, square=True, center=0)
                ax.set_title(f"Diff (t-{i+1})")

        plt.suptitle(f"Causal Graph: {name_suffix}")
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'adjacency_{name_suffix}.png'), dpi=150)
        plt.close()

    def _plot_adjacency_w(self, learned_w, gt_w, path, name_suffix):
        """绘制混合矩阵W对比"""
        plt = self._plt
        sns = self._sns

        d = learned_w.shape[0]

        fig, axes = plt.subplots(3, d, figsize=(d*3, 9))
        if d == 1:
            axes = axes.reshape(-1, 1)

        for i in range(d):
            # Learned
            sns.heatmap(learned_w[i], ax=axes[0, i], cmap="Blues",
                       vmin=0, vmax=1, cbar=True)
            axes[0, i].set_title(f"Learned W (d={i})")

            # GT
            sns.heatmap(gt_w[i], ax=axes[1, i], cmap="Blues",
                       vmin=0, vmax=1, cbar=True)
            axes[1, i].set_title(f"GT W (d={i})")

            # Difference
            diff = learned_w[i] - gt_w[i]
            sns.heatmap(diff, ax=axes[2, i], cmap="RdBu_r",
                       vmin=-1, vmax=1, cbar=True, center=0)
            axes[2, i].set_title(f"Diff (d={i})")

        plt.suptitle(f"Mixing Matrix: {name_suffix}")
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'adjacency_{name_suffix}.png'), dpi=150)
        plt.close()

    def _save_mcc(self, path):
        """保存 MCC 分数"""
        np.save(os.path.join(path, "mcc"), np.array(self.mcc))
        np.save(os.path.join(path, "assignments"), np.array(self.assignments))

        if len(self.mcc) > 1:
            plt = self._plt
            plt.figure()
            plt.plot(self.mcc)
            plt.title("MCC Score Over Time")
            plt.xlabel("Iteration")
            plt.ylabel("MCC")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(path, 'mcc.png'), dpi=150)
            plt.close()
