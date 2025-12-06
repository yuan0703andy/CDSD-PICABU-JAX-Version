import os
import torch
import numpy as np
from scipy.linalg import expm


def _torch_trace_expm(mat: torch.Tensor) -> torch.Tensor:
    """Differentiable trace(exp(mat)) using torch.linalg.matrix_exp."""
    expm_t = torch.linalg.matrix_exp(mat)
    return torch.trace(expm_t)


class TrExpScipy(torch.autograd.Function):
    """
    autograd.Function to compute trace of an exponential of a matrix via scipy.
    Kept for backward compatibility; by default we prefer the torch path.
    """

    @staticmethod
    def forward(ctx, input):
        with torch.no_grad():
            expm_input = expm(input.detach().cpu().numpy())
            expm_input = torch.as_tensor(expm_input, device=input.device, dtype=input.dtype)
            ctx.save_for_backward(expm_input)
            return torch.trace(expm_input)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            (expm_input,) = ctx.saved_tensors
            return expm_input.t() * grad_output


def compute_dag_constraint(w_adj: torch.Tensor, force_scipy: bool = False):
    """
    Compute the DAG constraint of w_adj.
    Uses torch.matrix_exp by default (robust inside Jupyter),
    falls back to scipy if force_scipy is True.
    """
    if not torch.is_tensor(w_adj):
        w_adj = torch.tensor(w_adj)
    assert (w_adj >= 0).detach().cpu().numpy().all()

    use_scipy = force_scipy or os.environ.get("CDSD_FORCE_SCIPY_EXPM", "").lower() in {"1", "true", "yes"}
    if use_scipy:
        h = TrExpScipy.apply(w_adj) - w_adj.shape[0]
    else:
        h = _torch_trace_expm(w_adj) - w_adj.shape[0]
    return h


def is_acyclic(adjacency):
    """
    Return true if adjacency is a acyclic
    :param np.ndarray adjacency: adjacency matrix
    """
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0:
            return False
    return True
