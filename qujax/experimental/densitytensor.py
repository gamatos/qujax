from typing import Any, Callable, Sequence, Tuple, Optional, Mapping, Union

# Backwards compatibility with Python <3.10
from typing_extensions import TypeVarTuple, Unpack

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from qujax.typing import Gate

from qujax.densitytensor import kraus

from qujax.typing import (
    KrausOp,
)

from qujax.experimental.internal import _to_kraus_operator_seq_funcs


from qujax.experimental.utils import get_default_gates, get_params
from qujax.experimental.typing import (
    MetaparameterisedOperation,
    ParamInds,
    GateDict,
    PyTree,
)

DensitytensorOperationSpecifier = Union[
    Gate,
    KrausOp,
    str,
]


def wrap_parameterised_tensor_list(
    gate_func: Sequence[Callable], qubit_inds: Sequence[int]
) -> Callable[[Tuple[jax.Array], jax.Array, PyTree], Tuple[jax.Array, jax.Array]]:
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

    def kraus_op(
        params: Tuple[jax.Array],
        densitytensor_in: jax.Array,
        classical_registers_in: PyTree,
    ):
        gate_matrices = [g(*params[0]) for g in gate_func]
        densitytensor = kraus(densitytensor_in, gate_matrices, qubit_inds)

        return densitytensor, classical_registers_in

    return kraus_op


def get_default_densitytensor_operations(
    gate_dict: GateDict,
) -> Mapping[str, MetaparameterisedOperation]:
    """
    Returns dictionary of default operations supported by qujax. Each operation is a function
    that takes a set of metaparemeters and returns another function. The returned function
    must have three arguments: `op_params`, `statetensor_in` and `classical_registers_in`.
    `op_params` holds parameters that are passed when the circuit is executed, while
    `statetensor_in` and `classical_registers_in` correspond to the statetensor
    and classical registers, respectively, being modified by the circuit.

    Parameters:
        `gate_dict`: Dictionary encoding quantum gates that the circuit can use. This
            dictionary maps strings to a callable in the case of parameterized gates or to a
            jax.Array in the case of unparameterized gates.
    """

    return {}


def parse_densitytensor_op(
    op: DensitytensorOperationSpecifier,
    metaparams: Sequence[Any],
    params: Sequence[Any],
    gate_dict: GateDict,
    op_dict: Mapping[str, Callable],
) -> Tuple[Callable, Any]:
    """
    Parses operation specified by `op`, applying relevant metaparameters and returning a callable
    retpresenting the operation to be applied to the circuit.

    Args:
        op: Operation specification. Can be:
            - A string, in which case we first check whether it is a gate by looking it up in
            `tensor_dict` and then check whether it is a more general operation by looking it up
            in `op_dict`.
            - A jax.Array, which we assume to represent a gate.
            - A callable, which we assume to represent a parameterized gate.
        metaparams: Operator metaparameters. For gates, these are the qubit indices the gate is to
            be applied to.
        tensor_dict: Dictionary mapping strings to gates.
        op_dict: Dictionary mapping strings to callables that take operation metaparameters and
            return a function representing the operation to be applied to the circuit.

    Returns:
        A callable encoding the operation to be applied to the circuit.
    """
    # Gates, Kraus Operations
    if (
        (isinstance(op, str) and op in gate_dict)
        or isinstance(op, jax.Array)
        or isinstance(op, (list, tuple))
        or callable(op)
    ):
        op_list, params = _to_kraus_operator_seq_funcs(op, params, gate_dict)
        return wrap_parameterised_tensor_list(op_list, metaparams), params

    if isinstance(op, str) and op in op_dict:
        return op_dict[op](*metaparams), params

    if isinstance(op, str):
        raise ValueError(f"String {op} not a known gate or operation")
    else:
        raise TypeError(
            f"Invalid specification for `op`, got type {type(op)} with value {op}"
        )


def get_params_to_densitytensor_func(
    op_seq: Sequence[DensitytensorOperationSpecifier],
    op_metaparams_seq: Sequence[Sequence[Any]],
    param_pos_seq: Sequence[ParamInds],
    op_dict: Optional[Mapping[str, MetaparameterisedOperation]] = None,
    gate_dict: Optional[GateDict] = None,
):
    """
    Creates a function that maps circuit parameters to a densitytensor.

    Args:
        op_seq: Sequence of operations to be executed.
            Can be either
            - a string specifying a gate in `gate_dict`
            - a jax.Array specifying a gate
            - a function returning a jax.Array specifying a parameterized gate.
            - a string specifying an operation in `op_dict`
        op_params_seq: Sequence of operation meta-parameters. Each element corresponds to one
            operation in `op_seq`. For gates, this will be the qubit indices the gate is applied to.
        param_pos_seq: Sequence of indices specifying the positions of the parameters each gate
            or operation takes.
            Note that these are parameters of the circuit, and are distinct from the meta-parameters
            fixed in `op_params_seq`.
        op_dict: Dictionary mapping strings to operations. Each operation is a function
            taking metaparameters (which are specified in `op_params_seq`) and returning another
            function. This returned function encodes the operation, and takes an array of
            parameters, a densitytensor and classical registers, and returns the updated densitytensor
            and classical registers after the operation is applied.
        gate_dict: Dictionary mapping strings to gates. Each gate is either a jax.Array or a
            function taking a number of parameters and returning a jax.Array.
            Defaults to qujax's dictionary of gates.
    Returns:
        Function that takes a number of parameters, an input densitytensor and an input set of
        classical registers, and returns the updated densitytensor and classical registers
        after the specified gates and operations are applied.
    """
    if gate_dict is None:
        gate_dict = get_default_gates()
    if op_dict is None:
        op_dict = get_default_densitytensor_operations(gate_dict)

    repeated_ops = set(gate_dict.keys()) & set(op_dict.keys())
    if repeated_ops:
        raise ValueError(
            f"Operation list and gate list have repeated operation(s): {repeated_ops}"
        )

    parsed_op_and_param_pos_seq = [
        parse_densitytensor_op(op, metaparams, params, gate_dict, op_dict)
        for op, metaparams, params in zip(op_seq, op_metaparams_seq, param_pos_seq)
    ]
    parsed_op_seq = [x[0] for x in parsed_op_and_param_pos_seq]
    parsed_param_pos_seq = [x[1] for x in parsed_op_and_param_pos_seq]

    def params_to_densitytensor_func(
        params: Union[Mapping[str, ArrayLike], ArrayLike],
        densitytensor_in: jax.Array,
        classical_registers_in: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, PyTree]:
        """
        Applies parameterised circuit to the quantum state represented by `densitytensor_in`.

        Args:
            params: Parameters to be passed to the circuit
            densitytensor_in: Input state in tensor form.
            classical_registers_in: Classical registers that can store intermediate results
                (e.g. measurements), possibly to later reuse them
        Returns:
            Resulting quantum state and classical registers after applying the circuit.

        """
        densitytensor = densitytensor_in
        classical_registers = classical_registers_in
        for (
            op,
            param_pos,
        ) in zip(
            parsed_op_seq,
            parsed_param_pos_seq,
        ):
            op_params = get_params(param_pos, params)
            densitytensor, classical_registers = op(
                op_params, densitytensor, classical_registers
            )

        return densitytensor, classical_registers

    return params_to_densitytensor_func
