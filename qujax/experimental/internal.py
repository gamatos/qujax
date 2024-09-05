from typing import Callable, Mapping, Union, Sequence, Tuple, Any, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from qujax.statetensor import apply_gate
from qujax.typing import Gate, GateFunction, KrausOp, GateParameterIndices
from qujax.experimental.typing import PyTree, GateDict


from qujax.utils import _arrayify_inds


def _to_kraus_operator_seq_funcs(
    kraus_op: KrausOp,
    param_inds: Optional[Union[GateParameterIndices, Sequence[GateParameterIndices]]],
    tensor_dict: GateDict,
) -> Tuple[Sequence[GateFunction], Sequence[jax.Array]]:
    """
    Ensures Kraus operators are a sequence of functions that map (possibly empty) parameters to
    tensors and that each element of param_inds_seq is a sequence of arrays that correspond to the
    parameter indices of each Kraus operator.

    Args:
        kraus_op: Either a normal Gate or a sequence of Gates representing Kraus operators.
        param_inds: If kraus_op is a normal Gate then a sequence of parameter indices,
            if kraus_op is a sequence of Kraus operators then a sequence of sequences of
            parameter indices

    Returns:
        Tuple containing sequence of functions mapping to Kraus operators
        and sequence of arrays with parameter indices

    """
    if isinstance(kraus_op, (list, tuple)):
        kraus_op_funcs = [_to_gate_func(ko, tensor_dict) for ko in kraus_op]
        if param_inds is None:
            param_inds = [None for _ in kraus_op]
    elif isinstance(kraus_op, (str, jax.Array)) or callable(kraus_op):
        kraus_op_funcs = [_to_gate_func(kraus_op, tensor_dict)]
        param_inds = [param_inds]
    else:
        raise ValueError(f"Invalid Kraus operator specification: {kraus_op}")
    return kraus_op_funcs, _arrayify_inds(param_inds)


def _gate_func_to_unitary(
    gate_func: GateFunction,
    n_qubits: int,
    params: jax.Array,
) -> jax.Array:
    """
    Compute tensor representing parameterised unitary for specific parameters.

    Args:
        gate_func: Function that maps a (possibly empty) parameter array to a unitary tensor
        n_qubts: Number of qubits unitary acts on
        params: Parameter vector

    Returns:
        Array containing gate unitary in tensor form.
    """
    gate_unitary = gate_func(*params)
    gate_unitary = gate_unitary.reshape(
        (2,) * (2 * n_qubits)
    )  # Ensure gate is in tensor form
    return gate_unitary


def _array_to_callable(arr: jax.Array) -> Callable[[], jax.Array]:
    """
    Wraps array `arr` in a callable that takes no parameters and returns `arr`.
    """

    def _no_param_tensor():
        return arr

    return _no_param_tensor


def _to_gate_func(
    gate: Gate,
    tensor_dict: Mapping[str, Union[Callable, jax.Array]],
) -> GateFunction:
    """
    Converts a gate specification to a callable function that takes the gate parameters and returns
    the corresponding unitary.

    Args:
        gate: Gate specification. Can be either a string, a callable or a jax.Array.

    Returns:
        Callable taking gate parameters and returning
    """

    if isinstance(gate, str):
        gate = tensor_dict[gate]
    if isinstance(gate, jax.Array):
        gate = _array_to_callable(gate)
    if callable(gate):
        return gate
    else:
        raise TypeError(
            f"Unsupported gate type - gate must be either a string in qujax.gates, a jax.Array or "
            f"callable: {type(gate)}"
        )


def _wrap_parameterised_tensor(
    gate_func: Callable, qubit_inds: Sequence[int]
) -> Callable:
    """
    Takes a callable representing a parameterised gate and wraps it in a function that takes
    the returned jax.Array and applies it to the qubits specified by `qubit_inds`.

    Args:
        gate_func: Callable representing parameterised gate.
        qubit_inds: Indices gate is to be applied to.

    Returns:
        Callable taking in gate parameters, input statetensor and input classical registers,
        and returning updated statetensor after applying parameterized gate to specified qubits.
    """

    def unitary_op(
        params: Tuple[jax.Array],
        statetensor_in: jax.Array,
        classical_registers_in: PyTree,
    ):
        gate_unitary = gate_func(*params[0])
        statetensor = apply_gate(statetensor_in, gate_unitary, qubit_inds)

        return statetensor, classical_registers_in

    return unitary_op
