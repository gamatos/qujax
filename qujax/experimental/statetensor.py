from typing import Any, Callable, Sequence, Tuple, Optional, Mapping, Union

# Backwards compatibility with Python <3.10
from typing_extensions import TypeVarTuple, Unpack

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from qujax.typing import Gate, GateFunction

from qujax.statetensor import apply_gate

import qujax.experimental.statetensor_operations as statetensor_operations
from qujax.experimental.typing import MetaparameterisedOperation, GateDict, PyTree
from qujax.experimental.internal import _to_gate_func, _wrap_parameterised_tensor
from qujax.experimental.utils import get_default_gates, get_params

StatetensorOperationSpecifier = Union[
    Gate,
    str,
]


def get_default_statetensor_operations(
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

    op_dict: dict[str, MetaparameterisedOperation] = dict()

    op_dict["Generic"] = statetensor_operations.generic_op

    def _conditional_gate(gates: Sequence[Gate], qubit_inds: Sequence[int]):
        return statetensor_operations.conditional_gate(gates, qubit_inds, gate_dict)

    op_dict["ConditionalGate"] = _conditional_gate
    op_dict["Measure"] = statetensor_operations.measure
    op_dict["Reset"] = statetensor_operations.reset

    return op_dict


def parse_statetensor_op(
    op: StatetensorOperationSpecifier,
    params: Sequence[Any],
    gate_dict: GateDict,
    op_dict: Mapping[str, Callable],
) -> Callable:
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
        params: Operator metaparameters. For gates, these are the qubit indices the gate is to
            be applied to.
        tensor_dict: Dictionary mapping strings to gates.
        op_dict: Dictionary mapping strings to callables that take operation metaparameters and
            return a function representing the operation to be applied to the circuit.

    Returns:
        A callable encoding the operation to be applied to the circuit.
    """
    # Gates
    if (
        (isinstance(op, str) and op in gate_dict)
        or isinstance(op, jax.Array)
        or callable(op)
    ):
        op = _to_gate_func(op, gate_dict)
        return _wrap_parameterised_tensor(op, params)

    if isinstance(op, str) and op in op_dict:
        return op_dict[op](*params)

    if isinstance(op, str):
        raise ValueError(f"String {op} not a known gate or operation")
    else:
        raise TypeError(
            f"Invalid specification for `op`, got type {type(op)} with value {op}"
        )


ParamInds = Optional[
    Union[
        int,
        Sequence[int],
        Sequence[Sequence[int]],
        Mapping[str, int],
        Mapping[str, Sequence[int]],
    ]
]


def get_params_to_statetensor_func(
    op_seq: Sequence[StatetensorOperationSpecifier],
    op_metaparams_seq: Sequence[Sequence[Any]],
    param_pos_seq: Sequence[ParamInds],
    op_dict: Optional[Mapping[str, MetaparameterisedOperation]] = None,
    gate_dict: Optional[GateDict] = None,
):
    """
    Creates a function that maps circuit parameters to a statetensor.

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
            parameters, a statetensor and classical registers, and returns the updated statetensor
            and classical registers after the operation is applied.
        gate_dict: Dictionary mapping strings to gates. Each gate is either a jax.Array or a
            function taking a number of parameters and returning a jax.Array.
            Defaults to qujax's dictionary of gates.
    Returns:
        Function that takes a number of parameters, an input statetensor and an input set of
        classical registers, and returns the updated statetensor and classical registers
        after the specified gates and operations are applied.
    """
    if gate_dict is None:
        gate_dict = get_default_gates()
    if op_dict is None:
        op_dict = get_default_statetensor_operations(gate_dict)

    repeated_ops = set(gate_dict.keys()) & set(op_dict.keys())
    if repeated_ops:
        raise ValueError(
            f"Operation list and gate list have repeated operation(s): {repeated_ops}"
        )

    parsed_op_seq = [
        parse_statetensor_op(op, params, gate_dict, op_dict)
        for op, params in zip(op_seq, op_metaparams_seq)
    ]

    def params_to_statetensor_func(
        params: Union[Mapping[str, ArrayLike], ArrayLike],
        statetensor_in: jax.Array,
        classical_registers_in: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, PyTree]:
        """
        Applies parameterised circuit to the quantum state represented by `statetensor_in`.

        Args:
            params: Parameters to be passed to the circuit
            statetensor_in: Input state in tensor form.
            classical_registers_in: Classical registers that can store intermediate results
                (e.g. measurements), possibly to later reuse them
        Returns:
            Resulting quantum state and classical registers after applying the circuit.

        """
        statetensor = statetensor_in
        classical_registers = classical_registers_in
        for (
            op,
            param_pos,
        ) in zip(
            parsed_op_seq,
            param_pos_seq,
        ):
            op_params = get_params(param_pos, params)
            statetensor, classical_registers = op(
                op_params, statetensor, classical_registers
            )

        return statetensor, classical_registers

    return params_to_statetensor_func
