from typing import Callable, Sequence, Tuple, Optional, Mapping, Union, Any
import operator

import jax
import jax.numpy as jnp

from qujax.typing import KrausOp, GateParameterIndices

import qujax.gates
from qujax.statetensor import apply_gate
from qujax.densitytensor import kraus
from qujax.kraus import Reset

from qujax.experimental.internal import (
    _to_gate_func,
    _gate_func_to_unitary,
    parse_densitytensor_op,
    _to_kraus_operator_seq_funcs,
)
from qujax.experimental.typing import (
    Operation,
    GateMapping,
    DensitytensorOperationSpecifier,
    ParamInds,
    MetaparameterisedOperation,
    GateDict,
)
from qujax.experimental.utils import get_params


def _get_pexb(tensor, d):
    identity = jnp.diag(jnp.ones(tensor.shape[0]))

    def _pexb(p) -> jax.Array:
        a = -1 / 2 * jnp.pi * p
        gate = jnp.cos(a) * identity + 1j * jnp.sin(a) * tensor
        gate = gate.reshape((2,) * 2 * d)
        return gate

    return _pexb


def conditional_operation(
    op_seq: Sequence[str],
    op_metaparams_seq: Sequence[Sequence[Any]],
    param_pos: Sequence[ParamInds],
    gate_dict: GateDict,
    op_dict: Mapping[str, MetaparameterisedOperation],
) -> Operation:
    """
    Apply conditional operation.

    Args:
        qubit_index: index of qubit to reset
    """

    parsed_op_and_param_pos_seq = [
        parse_densitytensor_op(op, metaparams, param_pos, gate_dict, op_dict)
        for op, metaparams in zip(op_seq, op_metaparams_seq)
    ]
    parsed_op_seq = [x[0] for x in parsed_op_and_param_pos_seq]

    def _get_wrapped_f(f):
        def _wrapped_f(args):
            return f(*args)

        return _wrapped_f

    wrapped_parsed_op_seq = [_get_wrapped_f(f) for f in parsed_op_seq]
    parsed_param_pos_seq = [x[1] for x in parsed_op_and_param_pos_seq]

    parsed_param_pos = parsed_param_pos_seq[0]
    if not all(parsed_param_pos == x for x in parsed_param_pos_seq):
        raise ValueError()

    def apply_conditional_operation(
        op_params: Tuple[jax.Array] | Tuple[jax.Array, jax.Array],
        densitytensor_in: jax.Array,
        classical_registers_in: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """ """
        if len(op_params) > 1:
            index, params = op_params
            parsed_op_params = get_params(parsed_param_pos, params)
        else:
            index, params = parsed_op_params[0], (jnp.array([]),)
        i = index.reshape()
        grouped_params = [x for x in zip(*parsed_op_params[0])]
        chosen_params = tuple(jnp.stack(jnp.array(g))[i] for g in grouped_params)

        # get param_pos from params
        res = jax.lax.switch(
            i,
            wrapped_parsed_op_seq,
            (chosen_params, densitytensor_in, classical_registers_in),
        )
        return res

    return apply_conditional_operation


def pauliexpbox(
    pauli_string: str,
    qubit_inds: Sequence[int],
    op_dict: Optional[Mapping[str, MetaparameterisedOperation]] = None,
    gate_dict: Optional[GateDict] = None,
) -> Operation:
    """
    Apply PauliExpBox.

    Args:
        qubit_index: index of qubit to reset
    """

    tensor = jnp.ones(1)
    # Build tensor product of Pauli matrices
    for p in pauli_string:
        # TODO: Handle op_dict and gate_dict
        m = qujax.gates.__dict__[p]
        tensor = jnp.kron(tensor, m)

    identity = jnp.diag(jnp.ones(tensor.shape[0]))
    n_paulis = len(pauli_string)

    def apply_pauliexpbox(
        op_params: Tuple[jax.Array],
        densitytensor_in: jax.Array,
        classical_registers_in: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """ """

        a = -1 / 2 * jnp.pi * op_params[0]
        gate = jnp.cos(a) * identity + 1j * jnp.sin(a) * tensor
        gate = gate.reshape((2,) * 2 * n_paulis)

        res = kraus(densitytensor_in, gate, qubit_inds)

        return (res, classical_registers_in)

    return apply_pauliexpbox


def reset(
    qubit_index: int, classical_register_index: Optional[int] = None
) -> Operation:
    """
    Reset qubit.

    Args:
        qubit_index: index of qubit to reset
    """

    def apply_reset(
        op_params: Tuple[jax.Array],
        densitytensor_in: jax.Array,
        classical_registers_in: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """ """

        res = kraus(densitytensor_in, Reset, [qubit_index])

        return (res, classical_registers_in)

    return apply_reset


def conditional_gate(
    kraus_ops: Sequence[KrausOp], qubit_inds: Sequence[int], gate_dict: GateMapping
) -> Operation:
    """
    Operation applying one of the gates in `gates` according to an index passed as a
    circuit parameter.

    Args:
        gates: gates from which one is selected to be applied
        qubit_indices: indices of qubits the selected gate is to be applied to
    """
    gate_funcs_and_dummy_params = [
        _to_kraus_operator_seq_funcs(g, None, gate_dict) for g in kraus_ops
    ]
    gate_funcs = [x[0] for x in gate_funcs_and_dummy_params]

    # Check if Kraus length choices are all the same
    kraus_lengths = None
    for k in kraus_ops:
        if isinstance(k, (list, tuple)):
            l = len(k)
        else:
            l = 1
        if kraus_lengths is None:
            kraus_lengths = l
        elif l != kraus_lengths:
            raise ValueError()

    def apply_conditional_gate(
        op_params: Union[Tuple[jax.Array], Tuple[jax.Array, jax.Array]],
        densitytensor_in: jax.Array,
        classical_registers_in: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Applies a gate specified by an index passed in `op_params` to a statetensor.

        Args:
            op_params: gates from which one is` selected to be applied
            statetensor_in: indices of qubits the selected gate is to be applied to
            classical_registers_in: indices of qubits the selected gate is to be applied to
        """
        if len(op_params) == 1:
            ind, gate_params = op_params[0][0], jnp.empty((len(kraus_ops), 0))
        elif len(op_params) == 2:
            ind, gate_params = op_params[0][0], op_params[1]
        else:
            raise ValueError("Invalid number of parameters for ConditionalGate")

        stacked_kraus_ops = jnp.stack(
            [
                jnp.stack(
                    [
                        _gate_func_to_unitary(
                            ko[i], len(qubit_inds), gate_params[i]
                        )
                        for i in range(len(ko))
                    ]
                )
                for ko in gate_funcs
            ]
        )

        chosen_kraus = stacked_kraus_ops[ind]

        densitytensor = kraus(densitytensor_in, chosen_kraus, qubit_inds)
        return densitytensor, classical_registers_in

    return apply_conditional_gate
