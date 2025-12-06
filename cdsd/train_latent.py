import os
from collections import deque

import torch
import numpy as np

from cdsd.dag_optim import compute_dag_constraint
from cdsd.utils import ALM
from cdsd.prox import monkey_patch_RMSprop

def _get_plotter():
    """Return a Plotter implementation (real or stub)."""
    disable = os.environ.get("CDSD_DISABLE_PLOTS", "").lower() in {"1", "true", "yes"}
    if disable:
        return lambda: _StubPlotter("CDSD_DISABLE_PLOTS is set")
    try:
        from cdsd.plot import Plotter as _Plotter  # type: ignore
        return _Plotter
    except Exception as exc:  # pragma: no cover - plotting disabled fallback
        return lambda: _StubPlotter(str(exc))


class _StubPlotter:  # pylint: disable=too-few-public-methods
    """Fallback Plotter so importing TrainingLatent never crashes."""

    def __init__(self, reason):
        self.reason = reason
        print(f"Plotting disabled: {reason}")

    def plot(self, *_, **__):
        return

    def save(self, *_, **__):
        return


Plotter = _get_plotter()


class TrainingLatent:
    def __init__(self, model, data, hp, best_metrics):
        self.model = model
        self.data = data
        self.hp = hp
        self.best_metrics = best_metrics

        self.latent = hp.latent
        self.no_gt = hp.no_gt
        self.debug_gt_z = hp.debug_gt_z
        self.gt_dag = data.gt_graph
        self.gt_w = data.gt_w
        self.d_z = hp.d_z
        self.no_w_constraint = hp.no_w_constraint

        self.d = data.x.shape[2]
        self.patience = hp.patience
        self.best_valid_loss = np.inf
        self.batch_size = hp.batch_size
        self.tau = hp.tau
        self.d_x = hp.d_x
        self.instantaneous = hp.instantaneous

        self.patience_freq = 50
        self.iteration = 1
        self.logging_iter = 0
        self.converged = False
        self.thresholded = False
        self.ended = False

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

        self.valid_loss_list = []
        self.valid_elbo_list = []
        self.valid_recons_list = []
        self.valid_kl_list = []
        self.valid_sparsity_reg_list = []
        self.valid_connect_reg_list = []
        self.valid_ortho_cons_list = []
        self.valid_ortho_vector_cons_list = []
        self.valid_acyclic_cons_list = []

        self.plotter = Plotter()

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

        # optimizer
        if hp.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=hp.lr)
        elif hp.optimizer == "rmsprop":
            # TODO: put back
            monkey_patch_RMSprop(torch.optim.RMSprop)
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=hp.lr)
        else:
            raise NotImplementedError("optimizer {} is not implemented".format(hp.optimizer))

        # compute constraint normalization
        with torch.no_grad():
            d = model.d * model.d_z
            full_adjacency = torch.ones((d, d)) - torch.eye(d)
            self.acyclic_constraint_normalization = compute_dag_constraint(full_adjacency).item()

            if self.latent:
                self.ortho_normalization = self.d_x * self.d_z

    def train_with_QPM(self):
        """
        Optimize a problem under constraint using the Augmented Lagragian
        method (or QPM). We train in 3 phases: first with ALM, then until
        the likelihood remain stable, then continue after thresholding
        the adjacency matrix
        """

        # initialize ALM/QPM for orthogonality and acyclicity constraints
        self.ALM_ortho = ALM(self.hp.ortho_mu_init,
                             self.hp.ortho_mu_mult_factor,
                             self.hp.ortho_omega_gamma,
                             self.hp.ortho_omega_mu,
                             self.hp.ortho_h_threshold,
                             self.hp.ortho_min_iter_convergence,
                             dim_gamma=(self.d_z, self.d_z))
        if self.instantaneous:
            # add the acyclicity constraint if the instantaneous connections
            # are considered
            self.QPM_acyclic = ALM(self.hp.acyclic_mu_init,
                                   self.hp.acyclic_mu_mult_factor,
                                   self.hp.acyclic_omega_gamma,
                                   self.hp.acyclic_omega_mu,
                                   self.hp.acyclic_h_threshold,
                                   self.hp.acyclic_min_iter_convergence)

        while self.iteration < self.hp.max_iteration and not self.ended:

            # train and valid step
            self.train_step()
            if self.iteration % self.hp.valid_freq == 0:
                self.logging_iter += 1
                x, y, y_pred = self.valid_step()
                self.log_losses()

                # print and plot losses
                if self.iteration % (self.hp.valid_freq * self.hp.print_freq) == 0:
                    self.print_results()
                if self.logging_iter > 0 and self.iteration % (self.hp.valid_freq * self.hp.plot_freq) == 0:
                    self.plotter.plot(self)

            if not self.converged:
                # train with penalty method
                if self.iteration % self.hp.valid_freq == 0:
                    self.ALM_ortho.update(self.iteration,
                                          self.valid_ortho_vector_cons_list,
                                          self.valid_loss_list)
                    if self.iteration > 1000:
                        if not self.no_w_constraint:
                            ortho_converged = self.ALM_ortho.has_converged
                        else:
                            self.converged = True
                    else:
                        ortho_converged = False

                    if self.ALM_ortho.has_increased_mu:
                        if self.hp.optimizer == "sgd":
                            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hp.lr)
                        elif self.hp.optimizer == "rmsprop":
                            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.hp.lr)

                    if self.instantaneous:
                        self.QPM_acyclic.update(self.iteration,
                                                self.valid_acyclic_cons_list,
                                                self.valid_loss_list)
                        acyclic_converged = self.QPM_acyclic.has_converged
                        if self.QPM_acyclic.has_increased_mu:
                            if self.hp.optimizer == "sgd":
                                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hp.lr)
                            elif self.hp.optimizer == "rmsprop":
                                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.hp.lr)
                        self.converged = ortho_converged & acyclic_converged
                    else:
                        self.converged = ortho_converged
            else:
                # continue training without penalty method
                if not self.thresholded and self.iteration % self.patience_freq == 0:
                    if not self.has_patience(self.hp.patience, self.valid_loss):
                        self.threshold()
                        self.patience = self.hp.patience_post_thresh
                        self.best_valid_loss = np.inf
                # continue training after thresholding
                else:
                    if self.iteration % self.patience_freq == 0:
                        if not self.has_patience(self.hp.patience_post_thresh, self.valid_loss):
                            self.ended = True

            self.iteration += 1

        if self.iteration >= self.hp.max_iteration:
            self.threshold()

        # final plotting and printing
        self.plotter.plot(self, save=True)
        self.print_results()

        valid_loss = {"valid_loss": self.valid_loss,
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
                      "valid_ortho_cons": torch.sum(self.valid_ortho_cons).item()}

        return valid_loss

    def train_step(self):
        self.model.train()

        # sample data
        x, y, z = self.data.sample(self.batch_size, valid=False)
        nll, recons, kl, y_pred = self.get_nll(x, y, z)

        # compute regularisations (sparsity and connectivity)
        sparsity_reg = self.get_regularisation()
        connect_reg = torch.tensor([0.])

        # compute constraints (acyclicity and orthogonality)
        h_acyclic = torch.tensor([0.])
        if self.instantaneous and not self.converged:
            h_acyclic = self.get_acyclicity_violation()
        # if self.hp.reg_coeff_connect:
        h_ortho = self.get_ortho_violation(self.model.autoencoder.get_w_decoder())

        # compute total loss
        loss = nll + sparsity_reg + connect_reg
        if not self.no_w_constraint:
            loss = loss + torch.sum(self.ALM_ortho.gamma * h_ortho) + \
                0.5 * self.ALM_ortho.mu * torch.sum(h_ortho ** 2)
        if self.instantaneous:
            loss = loss + 0.5 * self.QPM_acyclic.mu * h_acyclic ** 2

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        _, _ = self.optimizer.step() if self.hp.optimizer == "rmsprop" else self.optimizer.step(), self.hp.lr

        # projection of the gradient for w
        if self.model.autoencoder.use_grad_project and not self.no_w_constraint:
            with torch.no_grad():
                self.model.autoencoder.get_w_decoder().clamp_(min=0.)
            assert torch.min(self.model.autoencoder.get_w_decoder()) >= 0.

        self.train_loss = loss.item()
        self.train_nll = nll.item()
        self.train_recons = recons.item()
        self.train_kl = kl.item()
        self.train_sparsity_reg = sparsity_reg.item()
        self.train_connect_reg = connect_reg.item()
        self.train_ortho_cons = h_ortho.detach()
        self.train_acyclic_cons = h_acyclic.item()

        return x, y, y_pred

    def valid_step(self):
        self.model.eval()

        with torch.no_grad():
            # sample data
            x, y, z = self.data.sample(self.data.n_valid - self.data.tau, valid=True)
            nll, recons, kl, y_pred = self.get_nll(x, y, z)

            # compute regularisations (sparsity and connectivity)
            sparsity_reg = self.get_regularisation()
            connect_reg = torch.tensor([0.])

            # compute constraints (acyclicity and orthogonality)
            h_acyclic = torch.tensor([0.])
            if self.instantaneous and not self.converged:
                h_acyclic = self.get_acyclicity_violation()
            h_ortho = self.get_ortho_violation(self.model.autoencoder.get_w_decoder())

            # compute total loss
            loss = nll + sparsity_reg + connect_reg
            if self.instantaneous:
                loss = loss + 0.5 * self.QPM_acyclic.mu * h_acyclic ** 2

            self.valid_loss = loss.item()
            self.valid_nll = nll.item()
            self.valid_recons = recons.item()
            self.valid_kl = kl.item()
            self.valid_sparsity_reg = sparsity_reg.item()
            self.valid_ortho_cons = h_ortho.detach()
            self.valid_connect_reg = connect_reg.item()
            self.valid_acyclic_cons = h_acyclic.item()

        return x, y, y_pred

    def has_patience(self, patience_init, valid_loss):
        """
        Check if the validation loss has not improved for
        'patience' steps
        """
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
        """Consider that the graph has been found. Convert it to
        a binary graph and fix it."""
        with torch.no_grad():
            thresholded_adj = (self.model.get_adj() > 0.5).type(torch.Tensor)
            self.model.mask.fix(thresholded_adj)
        self.thresholded = True
        print("Thresholding ================")

    def log_losses(self):
        """Append in lists values of the losses and more"""
        # train
        self.train_loss_list.append(-self.train_loss)
        self.train_recons_list.append(self.train_recons)
        self.train_kl_list.append(self.train_kl)

        self.train_sparsity_reg_list.append(self.train_sparsity_reg)
        self.train_connect_reg_list.append(self.train_connect_reg)
        self.train_ortho_cons_list.append(torch.sum(self.train_ortho_cons))
        self.train_ortho_vector_cons_list.append(self.train_ortho_cons)
        self.train_acyclic_cons_list.append(self.train_acyclic_cons)

        # valid
        self.valid_loss_list.append(-self.valid_loss)
        self.valid_recons_list.append(self.valid_recons)
        self.valid_kl_list.append(self.valid_kl)

        self.valid_sparsity_reg_list.append(self.valid_sparsity_reg)
        self.valid_connect_reg_list.append(self.valid_connect_reg)
        self.valid_ortho_cons_list.append(torch.sum(self.valid_ortho_cons))
        self.valid_ortho_vector_cons_list.append(self.valid_ortho_cons)
        self.valid_acyclic_cons_list.append(self.valid_acyclic_cons)

        self.mu_ortho_list.append(self.ALM_ortho.mu)

        if self.adj_tt is not None:
            self.adj_tt.append(self.model.get_adj().detach().cpu().numpy())
        if self.adj_w_tt is not None:
            w = self.model.autoencoder.get_w_decoder().detach().cpu().numpy()
            self.adj_w_tt.append(w)
        self.logvar_decoder_tt.append(self.model.autoencoder.logvar_decoder[0].item())
        self.logvar_encoder_tt.append(self.model.autoencoder.logvar_encoder[0].item())
        self.logvar_transition_tt.append(self.model.transition_model.logvar[0, 0].item())

    def print_results(self):
        """Print values of many variable: losses, constraint violation, etc.
        at the frequency self.hp.print_freq"""
        print("============================================================")
        print(f"Iteration #{self.iteration}")
        print(f"Converged: {self.converged}")

        print(f"ELBO: {-self.train_nll:.4f}")
        print(f"Recons: {self.train_recons:.4f}")
        print(f"KL: {self.train_kl:.4f}")

        print(f"Sparsity_reg: {self.train_sparsity_reg:.1e}")

        print(f"ortho cons: {self.train_ortho_cons_list[-1]:.1e}")
        print(f"ortho mu: {self.ALM_ortho.mu}")

        if self.instantaneous:
            print(f"acyclic cons: {self.train_acyclic_cons:.4f}")
            print(f"acyclic mu: {self.QPM_acyclic.mu}")
        print("-------------------------------")

        print(f"valid_ELBO: {-self.valid_nll:.4f}")
        print(f"patience: {self.patience}")

    def get_nll(self, x, y, z=None) -> torch.Tensor:
        elbo, recons, kl, pred = self.model(x, y, z, self.iteration)
        return -elbo, recons, kl, pred

    def get_regularisation(self) -> float:
        if self.iteration > self.hp.schedule_reg:
            adj = self.model.get_adj()
            reg = self.hp.reg_coeff * torch.norm(adj, p=1)
        else:
            reg = torch.tensor([0.])

        return reg

    def get_acyclicity_violation(self) -> torch.Tensor:
        if self.iteration > 0:
            adj = self.model.get_adj()[-1].view(self.d * self.d_z, self.d * self.d_z)
            h = compute_dag_constraint(adj) / self.acyclic_constraint_normalization
        else:
            h = torch.tensor([0.])

        return h

    def get_ortho_violation(self, w: torch.Tensor) -> float:
        if self.iteration > self.hp.schedule_ortho:
            k = w.size(2)
            i = 0
            constraint = w[i].T @ w[i] - torch.eye(k)
            h = constraint / self.ortho_normalization
        else:
            h = torch.tensor([0.])
        return h
