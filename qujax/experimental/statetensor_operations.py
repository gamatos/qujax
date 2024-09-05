from typing import Callable, Sequence, Tuple, Optional, Mapping, Union


import jax
import jax.numpy as jnp

from qujax.typing import Gate

from qujax.statetensor import apply_gate

from qujax.experimental.internal import _to_gate_func, _gate_func_to_unitary
from qujax.experimental.typing import Operation

GateMapping = Mapping[str, Union[Callable, jax.Array]]


def generic_op(f: Operation) -> Operation:
    """
    Generic operation to be applied to the circuit, passed as a metaparameter `f`.
    """
    return f


def conditional_gate(
    gates: Sequence[Gate], qubit_inds: Sequence[int], gate_dict: GateMapping
) -> Operation:
    """
    Operation applying one of the gates in `gates` according to an index passed as a
    circuit parameter.

    Args:
        gates: gates from which one is selected to be applied
        qubit_indices: indices of qubits the selected gate is to be applied to
    """
    gate_funcs = [_to_gate_func(g, gate_dict) for g in gates]

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
            ind, gate_params = op_params[0][0], jnp.empty((len(gates), 0))
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


def measure(qubit_index: int, classical_register_index: int) -> Operation:
    """
    Measure qubit.

    Args:
        qubit_index: index of qubit to measure
    """

    def apply_measure(
        op_params: Tuple[jax.Array],
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

        rng = op_params[0][0]
        sum_axes = tuple(x for x in range(statetensor_in.ndim) if x != qubit_index)

        probabilities = jnp.sum(jnp.square(jnp.abs(statetensor_in)), axis=sum_axes)
        projection_on_zero_state = jnp.array(([[1, 0], [0, 0]]))
        projection_on_one_state = jnp.array(([[0, 0], [0, 1]]))
        projection_array = jnp.array(
            [projection_on_zero_state, projection_on_one_state]
        )

        measurement = jax.random.choice(rng, jnp.array([1, -1]), p=probabilities)
        projection = projection_array[(1 - measurement) // 2]
        statetensor_in = apply_gate(statetensor_in, projection, [qubit_index])
        statetensor_in /= jnp.linalg.norm(statetensor_in)

        classical_registers_in = classical_registers_in.at[
            classical_register_index
        ].set(measurement)
        return statetensor_in, classical_registers_in

    return apply_measure


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
        statetensor_in: jax.Array,
        classical_registers_in: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """ """

        statetensor_in, measurement = measure(qubit_index, 0)(
            op_params, statetensor_in, jnp.zeros(1)
        )

        if classical_register_index is not None:
            classical_registers_in = classical_registers_in.at[
                classical_register_index
            ].set(measurement.item())

        axes_order = jnp.array([[0, 1], [1, 0]])
        rescaled_measurement = (1 - measurement) // 2
        index = rescaled_measurement.astype(int)
        chosen_axes_order = axes_order[index].reshape(2)
        # conditional rotation to zero state
        return (
            jnp.take(statetensor_in, chosen_axes_order, axis=qubit_index),
            classical_registers_in,
        )

    return apply_reset
