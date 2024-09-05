from typing import Any, Literal, Sequence, Optional, Final

# from jax import Trac

from qujax.typing import MixedCircuitFunction, PureCircuitFunction

from qujax.experimental.utils import get_default_gates
from qujax.experimental.typing import GateDict, ParamInds, StatetensorOperationSpecifier

from qujax.experimental.statetensor import get_params_to_statetensor_func
from qujax.experimental.densitytensor import get_params_to_densitytensor_func

CircuitType = Literal["statetensor", "densitytensor"]


class Circuit:
    _circuit_type: CircuitType
    _operations: Sequence[StatetensorOperationSpecifier]
    _operation_metaparameters: Sequence[Sequence[Any]]
    _param_pos_seq: Sequence[ParamInds]
    _gate_dict: GateDict
    n_qubits: int

    gates: GateDict

    def __init__(
        self,
        circuit_type: Literal["statetensor", "densitytensor"],
        gate_dict: Optional[GateDict],
    ) -> None:
        self._circuit_type = circuit_type
        if gate_dict is not None:
            self._gate_dict = gate_dict
        else:
            self._gate_dict = get_default_gates()

    def __getattr__(self, name: str) -> Any:
        gates = get_default_gates()
        if name in gates:
            pass

        if self._circuit_type == "statetensor":
            d = 1
        elif self._circuit_type == "densitytensor":
            d = 1
        else:
            raise ValueError()

    def get_function(self):
        if self.circuit_type == "statetensor":
            return get_params_to_statetensor_func(
                self._operations, self._operation_metaparameters, self._param_pos_seq
            )
        elif self.circuit_type == "densitytensor":
            return get_params_to_densitytensor_func(
                self._operations, self._operation_metaparameters, self._param_pos_seq
            )

    def get_rng_parameter_names():
        pass

    @property
    def circuit_type(self):
        return self._circuit_type


def check_arguments(c: Circuit) -> None:
    pass
