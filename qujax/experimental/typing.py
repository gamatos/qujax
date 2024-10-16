from typing import Mapping, Any, Callable, Optional, Sequence, Union, Tuple

# Backwards compatibility with Python <3.10
from typing_extensions import TypeVarTuple, Unpack

import jax

from qujax.typing import GateFunction, KrausOp, Gate

ParamInds = Optional[
    Union[
        int,
        Sequence[int],
        Sequence[Sequence[int]],
        Mapping[str, int],
        Mapping[str, Sequence[int]],
    ]
]


PyTree = Any

Operation = Callable[
    [Tuple[jax.Array, ...], jax.Array, jax.Array], Tuple[jax.Array, jax.Array]
]

OperationMetaparameters = TypeVarTuple("OperationMetaparameters")
MetaparameterisedOperation = Callable[[Unpack[OperationMetaparameters]], Operation]

GateDict = Mapping[str, Union[GateFunction, jax.Array]]
GateMapping = Mapping[str, Union[Callable, jax.Array]]

DensitytensorOperationSpecifier = Union[
    Gate,
    KrausOp,
    str,
]

