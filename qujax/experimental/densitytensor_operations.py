from typing import Callable, Sequence, Tuple, Optional, Mapping, Union

import jax
import jax.numpy as jnp

from qujax.typing import KrausOp

from qujax.statetensor import apply_gate

from qujax.experimental.internal import _to_gate_func, _gate_func_to_unitary
from qujax.experimental.typing import Operation


GateMapping = Mapping[str, Union[Callable, jax.Array]]


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
    gate_funcs = [_to_gate_func(g, gate_dict) for g in kraus_ops]

    # Check if Kraus choices are all the same
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
        statetensor_in: jax.Array,
        classical_registers_in: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Applies a gate specified by an index passed in `op_params` to a statetensor.

        Args:
            op_params: gates from which one is selected to be applied
            statetensor_in: indices of qubits the selected gate is to be applied to
            classical_registers_in: indices of qubits the selected gate is to be applied to
        """
        if len(op_params) == 1:
            ind, gate_params = op_params[0][0], jnp.empty((len(kraus_ops), 0))
        elif len(op_params) == 2:
            ind, gate_params = op_params[0][0], op_params[1]
        else:
            raise ValueError("Invalid number of parameters for ConditionalGate")

        unitaries = jnp.stack(
            [
                _gate_func_to_unitary(gate_funcs[i], len(qubit_inds), gate_params[i])
                for i in range(len(gate_funcs))
            ]
        )

        chosen_unitary = unitaries[ind]

        statevector = apply_gate(statetensor_in, chosen_unitary, qubit_inds)
        return statevector, classical_registers_in

    return apply_conditional_gate