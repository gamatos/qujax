from typing import Mapping, Tuple, Union, Any, Sequence

import jax
from jax.typing import ArrayLike
import jax.numpy as jnp

from qujax import gates

from qujax.experimental.typing import ParamInds


def get_params(
    param_inds: ParamInds,
    params: Union[Mapping[str, ArrayLike], ArrayLike],
    root: bool = True,
) -> Tuple[Any, ...]:
    """
    Extracts parameters from `params` using indices specified by `param_inds`.

    Args:
        param_inds: Indices of parameters. Can be
            - None (results in an empty jax.Array)
            - an integer, when `params` is an indexable array
            - a dictionary, when `params` is also a dictionary
            - nested list or tuples of the above
        params: Parameters from which a subset is picked. Can be either an array or a dictionary
            of arrays
    Returns:
        Tuple of indexed parameters respeciting the structure of nested lists/tuples of param_inds.

    """
    op_params: Tuple[Any, ...]
    if param_inds is None:
        op_params = jnp.empty(0)
    elif param_inds == 0 and jnp.isscalar(params):
        op_params = jnp.array([params])
    elif isinstance(param_inds, int) and isinstance(params, jax.Array):
        op_params = jnp.array([params[param_inds]])
    elif isinstance(params, dict) and isinstance(param_inds, str):
        op_params = params[param_inds]
    elif isinstance(param_inds, dict) and isinstance(params, dict):
        op_params = tuple(
            get_params(param_inds[k], params[k], False) for k in param_inds
        )
        if len(op_params) == 1:
            op_params = op_params[0]
    elif isinstance(param_inds, (list, tuple)):
        if len(param_inds):
            if all(isinstance(x, int) for x in param_inds):
                op_params = jnp.take(params, jnp.array(param_inds), axis=0)
            else:
                op_params = tuple(get_params(p, params, False) for p in param_inds)
        else:
            op_params = jnp.array([])
    else:
        raise TypeError(
            f"Invalid specification for parameters: {type(param_inds)=} {type(params)=}."
        )

    if root and not isinstance(op_params, tuple):
        op_params = (op_params,)

    return op_params


def get_default_gates() -> dict:
    """
    Returns dictionary of default gates supported by qujax.
    """
    return {
        k: v for k, v in gates.__dict__.items() if not k.startswith(("_", "jax", "jnp"))
    }


def apply_gate(
    statetensor: jax.Array, gate_unitary: jax.Array, qubit_inds: Sequence[int]
) -> jax.Array:
    """
    Applies gate to statetensor and returns updated statetensor.
    Gate is represented by a unitary matrix in tensor form.

    Args:
        statetensor: Input statetensor.
        gate_unitary: Unitary array representing gate
            must be in tensor form with shape (2,2,...).
        qubit_inds: Sequence of indices for gate to be applied to.
            Must have 2 * len(qubit_inds) = gate_unitary.ndim

    Returns:
        Updated statetensor.
    """
    statetensor = jnp.tensordot(
        gate_unitary, statetensor, axes=(list(range(-len(qubit_inds), 0)), qubit_inds)
    )
    statetensor = jnp.moveaxis(statetensor, list(range(len(qubit_inds))), qubit_inds)
    return statetensor
