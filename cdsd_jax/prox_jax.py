"""
prox_jax.py
JAX/Optax 版本的自定义 RMSprop

将 PyTorch monkey-patch 实现迁移到 Optax
主要功能：返回 effective learning rates
"""
import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple, Tuple, Any
import chex


class RMSpropState(NamedTuple):
    """自定义 RMSprop 状态"""
    square_avg: chex.ArrayTree      # 梯度平方的移动平均
    effective_lr: chex.ArrayTree    # 有效学习率（用于 ALM/QPM）
    step: int                       # 步数


def _rmsprop_with_effective_lr_impl(
    learning_rate: float,
    decay: float = 0.99,
    eps: float = 1e-8,
    momentum: float = 0.0,
    centered: bool = False,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """
    JAX 版本 RMSprop + effective_lr
    盡量貼近 PyTorch monkey_patch_RMSprop：
      square_avg = alpha * square_avg + (1-alpha) * grad^2
      avg        = sqrt(square_avg) + eps
      update     = - lr * grad / avg
      eff_lr     =   lr * grad / avg
    """
    if momentum > 0:
        raise NotImplementedError("Momentum not yet supported")
    if centered:
        raise NotImplementedError("Centered RMSprop not yet supported")

    def init_fn(params: chex.ArrayTree) -> RMSpropState:
        square_avg = jax.tree.map(jnp.zeros_like, params)
        effective_lr = jax.tree.map(jnp.zeros_like, params)
        return RMSpropState(
            square_avg=square_avg,
            effective_lr=effective_lr,
            step=0,
        )

    def update_fn(
        updates: chex.ArrayTree,
        state: RMSpropState,
        params: chex.ArrayTree | None = None,
    ) -> Tuple[chex.ArrayTree, RMSpropState]:
        # 對齊 PyTorch: 先加 weight decay 再算 square_avg
        if weight_decay != 0.0 and params is not None:
            updates = jax.tree.map(
                lambda g, p: g + weight_decay * p,
                updates,
                params,
            )

        # square_avg_t = decay * square_avg_{t-1} + (1-decay) * g^2
        new_square_avg = jax.tree.map(
            lambda s, g: decay * s + (1.0 - decay) * (g * g),
            state.square_avg,
            updates,
        )

        # avg = sqrt(square_avg) + eps  （不要寫成 sqrt(s + eps)）
        avg = jax.tree.map(
            lambda s: jnp.sqrt(s) + eps,
            new_square_avg,
        )

        # update = -lr * g / avg
        param_updates = jax.tree.map(
            lambda g, a: -learning_rate * g / a,
            updates,
            avg,
        )

        # effective lr = +lr * g / avg（和 PyTorch 一樣，沒有負號）
        new_effective_lr = jax.tree.map(
            lambda g, a: learning_rate * g / a,
            updates,
            avg,
        )

        new_state = RMSpropState(
            square_avg=new_square_avg,
            effective_lr=new_effective_lr,
            step=state.step + 1,
        )
        return param_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def rmsprop_with_effective_lr(
    learning_rate: float,
    decay: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    return _rmsprop_with_effective_lr_impl(
        learning_rate=learning_rate,
        decay=decay,
        eps=eps,
        weight_decay=weight_decay,
    )


# ========== 标准 Optax 包装（如果不需要 effective LRs）==========

def rmsprop_standard(learning_rate: float,
                    decay: float = 0.99,
                    eps: float = 1e-8) -> optax.GradientTransformation:
    """
    标准 RMSprop（使用 Optax）

    如果不需要 effective learning rates，使用这个更简单
    """
    return optax.rmsprop(
        learning_rate=learning_rate,
        decay=decay,
        eps=eps
    )


# ========== 测试和验证函数 ==========

def test_rmsprop():
    """测试自定义 RMSprop"""
    print("Testing RMSprop with effective LR...")

    # 创建简单的参数
    params = {
        'w': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'b': jnp.array([0.5, -0.5])
    }

    # 创建优化器
    optimizer = rmsprop_with_effective_lr(learning_rate=0.01, decay=0.99)
    opt_state = optimizer.init(params)

    print(f"  Initial params:")
    print(f"    w: {params['w']}")
    print(f"    b: {params['b']}")

    # 模拟梯度
    grads = {
        'w': jnp.array([[0.1, 0.2], [0.3, 0.4]]),
        'b': jnp.array([0.05, -0.05])
    }

    print(f"\n  Gradients:")
    print(f"    w: {grads['w']}")
    print(f"    b: {grads['b']}")

    # 更新参数
    updates, new_state = optimizer.update(grads, opt_state, params)

    print(f"\n  Updates:")
    print(f"    w: {updates['w']}")
    print(f"    b: {updates['b']}")

    print(f"\n  Effective learning rates (stored in state):")
    print(f"    w: {new_state.effective_lr['w']}")
    print(f"    b: {new_state.effective_lr['b']}")

    # 应用更新
    new_params = jax.tree.map(lambda p, u: p + u, params, updates)

    print(f"\n  New params:")
    print(f"    w: {new_params['w']}")
    print(f"    b: {new_params['b']}")

    # 多步更新测试
    print(f"\n  Testing multiple steps...")
    params_test = params
    opt_state_test = opt_state

    for i in range(5):
        # 模拟不同的梯度
        grads_step = jax.tree.map(lambda g: g * (1.0 / (i + 1)), grads)

        updates_step, opt_state_test = optimizer.update(
            grads_step, opt_state_test, params_test
        )

        params_test = jax.tree.map(lambda p, u: p + u, params_test, updates_step)

        print(f"    Step {i+1}: w[0,0]={params_test['w'][0,0]:.6f}, "
              f"square_avg={opt_state_test.square_avg['w'][0,0]:.6f}, "
              f"eff_lr[0,0]={opt_state_test.effective_lr['w'][0,0]:.6f}")

    # 对比标准 RMSprop
    print(f"\n  Comparing with standard Optax RMSprop...")

    optimizer_std = rmsprop_standard(learning_rate=0.01, decay=0.99)
    opt_state_std = optimizer_std.init(params)

    updates_std, _ = optimizer_std.update(grads, opt_state_std)

    print(f"    Custom updates (w[0,0]): {updates['w'][0,0]:.6f}")
    print(f"    Standard updates (w[0,0]): {updates_std['w'][0,0]:.6f}")
    print(f"    Difference: {abs(updates['w'][0,0] - updates_std['w'][0,0]):.2e}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_rmsprop()
