import jax
import jax.numpy as jnp
from jax import jit

import qujax
from qujax import all_zeros_statetensor, apply_gate, all_zeros_densitytensor, statetensor_to_densitytensor
from qujax.experimental.statetensor import get_params_to_statetensor_func
from qujax.experimental.densitytensor import get_params_to_densitytensor_func

def test_get_params_to_statetensor_func():
    ops = ["H", "H", "H", "CX", "Rz", "CY"]
    op_params = [[0], [1], [2], [0, 1], [1], [1, 2]]
    param_inds = [[], [], [], None, [0], []]

    param_to_st = get_params_to_statetensor_func(ops, op_params, param_inds)
    param_to_st = jax.jit(param_to_st)
    param = jnp.array([0.1])
    st_in = all_zeros_statetensor(3)
    st, _ = param_to_st(param, st_in)

    true_sv = jnp.array(
        [
            0.34920055 - 0.05530793j,
            0.34920055 - 0.05530793j,
            0.05530793 - 0.34920055j,
            -0.05530793 + 0.34920055j,
            0.34920055 - 0.05530793j,
            0.34920055 - 0.05530793j,
            0.05530793 - 0.34920055j,
            -0.05530793 + 0.34920055j,
        ],
        dtype="complex64",
    )

    assert st.size == true_sv.size
    assert jnp.allclose(st.flatten(), true_sv)


def test_get_params_to_densitytensor_func():
    n_qubits = 2

    gate_seq = ["Rx" for _ in range(n_qubits)]
    qubit_inds_seq = [(i,) for i in range(n_qubits)]
    param_inds_seq = [(i,) for i in range(n_qubits)]

    gate_seq += ["CZ" for _ in range(n_qubits - 1)]
    qubit_inds_seq += [(i, i + 1) for i in range(n_qubits - 1)]
    param_inds_seq += [() for _ in range(n_qubits - 1)]

    params_to_dt = get_params_to_densitytensor_func(
        gate_seq, qubit_inds_seq, param_inds_seq
    )
    params_to_st = get_params_to_statetensor_func(
        gate_seq, qubit_inds_seq, param_inds_seq
    )

    params = jnp.arange(1, n_qubits + 1) / 10.0

    st_in = all_zeros_statetensor(n_qubits)
    st, _ = params_to_st(params, st_in)
    dt_in = all_zeros_densitytensor(n_qubits)
    dt_test = qujax.statetensor_to_densitytensor(st)

    dt, _ = params_to_dt(params, dt_in)

    assert jnp.allclose(dt, dt_test)

    jit_dt, _ = jit(params_to_dt)(params, dt_in)
    assert jnp.allclose(jit_dt, dt_test)


def test_conditional_gate_st():
    ops = ["ConditionalGate"]
    op_params = [[["X", "Y", "Z"], [0]]]
    param_inds = [{"op_ind": 0}]

    st_in = all_zeros_statetensor(1)
    X_apply = apply_gate(st_in, qujax.gates.X, [0])
    Y_apply = apply_gate(st_in, qujax.gates.Y, [0])
    Z_apply = apply_gate(st_in, qujax.gates.Z, [0])

    param_to_st = get_params_to_statetensor_func(ops, op_params, param_inds)
    param_to_st = jax.jit(param_to_st)

    st_in = all_zeros_statetensor(1)

    st_X, _ = param_to_st({"op_ind": 0}, st_in)
    st_Y, _ = param_to_st({"op_ind": 1}, st_in)
    st_Z, _ = param_to_st({"op_ind": 2}, st_in)

    assert jnp.allclose(X_apply, st_X)
    assert jnp.allclose(Y_apply, st_Y)
    assert jnp.allclose(Z_apply, st_Z)


def test_conditional_gate_dt():
    ops = ["ConditionalGate"]
    op_params = [[["X", "Y", "Z"], [0]]]
    param_inds = [{"op_ind": 0}]

    st_in = all_zeros_statetensor(1)
    X_apply = apply_gate(st_in, qujax.gates.X, [0])
    Y_apply = apply_gate(st_in, qujax.gates.Y, [0])
    Z_apply = apply_gate(st_in, qujax.gates.Z, [0])

    dt_X_apply = statetensor_to_densitytensor(X_apply)
    dt_Y_apply = statetensor_to_densitytensor(Y_apply)
    dt_Z_apply = statetensor_to_densitytensor(Z_apply)

    param_to_dt = get_params_to_densitytensor_func(ops, op_params, param_inds)
    # param_to_dt = jax.jit(param_to_dt)

    dt_in = all_zeros_densitytensor(1)

    dt_X, _ = param_to_dt({"op_ind": 0}, dt_in)
    dt_Y, _ = param_to_dt({"op_ind": 1}, dt_in)
    dt_Z, _ = param_to_dt({"op_ind": 2}, dt_in)

    assert jnp.allclose(dt_X_apply, dt_X)
    assert jnp.allclose(dt_Y_apply, dt_Y)
    assert jnp.allclose(dt_Z_apply, dt_Z)


def test_parameterised_conditional_gate_st():
    ops = ["ConditionalGate"]
    op_params = [[["Rx", "Ry", "Rz"], [0]]]
    param_inds = [[{"op_ind": 0}, [{"angles": 0}, {"angles": 1}, {"angles": 2}]]]

    st_in = all_zeros_statetensor(1)
    params = jnp.array([0.1, 0.2, 0.3])

    CX_apply = apply_gate(st_in, qujax.gates.Rx(params[0].item()), [0])
    CY_apply = apply_gate(st_in, qujax.gates.Ry(params[1].item()), [0])
    CZ_apply = apply_gate(st_in, qujax.gates.Rz(params[2].item()), [0])

    param_to_st = get_params_to_statetensor_func(ops, op_params, param_inds)

    st_in = all_zeros_statetensor(1)

    st_CX, _ = param_to_st({"angles": params, "op_ind": 0}, st_in)
    st_CY, _ = param_to_st({"angles": params, "op_ind": 1}, st_in)
    st_CZ, _ = param_to_st({"angles": params, "op_ind": 2}, st_in)

    assert jnp.allclose(CX_apply, st_CX)
    assert jnp.allclose(CY_apply, st_CY)
    assert jnp.allclose(CZ_apply, st_CZ)

    batched_op_inds = jnp.array([[0], [1], [2]])

    batched_param_to_st = jax.vmap(
        param_to_st, in_axes=({"angles": None, "op_ind": 0}, None)
    )

    batched_st, _ = batched_param_to_st(
        {"angles": params, "op_ind": batched_op_inds}, st_in
    )

    assert jnp.allclose(batched_st, jnp.stack([st_CX, st_CY, st_CZ]))


def test_measure():
    ops = ["Measure"]
    op_params = [[0, 0]]
    param_inds = [[0]]

    state = 1 / jnp.sqrt(2) * jnp.array([1, 0, 0, 1]).reshape(2, 2)
    seed = 0

    param_to_st = get_params_to_statetensor_func(ops, op_params, param_inds)
    rng = jax.random.PRNGKey(seed)
    for rng_i in jax.random.split(rng, 10):
        params = rng_i.reshape(1, *rng_i.shape)
        classical_registers = jnp.zeros(1)
        result, measurement = param_to_st(params, state, classical_registers)

        if measurement == 1:
            assert jnp.allclose(result, jnp.array([1, 0, 0, 0]).reshape(2, 2))
        elif measurement == -1:
            assert jnp.allclose(result, jnp.array([0, 0, 0, 1]).reshape(2, 2))
        else:
            raise ValueError("Measurement is not 1 or -1.")


def test_measure_probability():
    ops = ["Measure"]
    op_params = [[0, 0]]
    param_inds = [[0]]

    state = 1 / jnp.sqrt(2) * jnp.array([1, 0, 0, 1]).reshape(2, 2)
    seed = 0

    param_to_st = get_params_to_statetensor_func(ops, op_params, param_inds)
    vectorized_param_to_st = jax.jit(jax.vmap(param_to_st, (0, None, None)))
    rng = jax.random.PRNGKey(seed)
    n_samples = int(1e5)
    vectorized_rng = jax.random.split(rng, n_samples)

    vectorized_params = vectorized_rng.reshape(n_samples, 1, *rng.shape)
    classical_registers = jnp.zeros(1)
    result, measurement = vectorized_param_to_st(
        vectorized_params, state, classical_registers
    )

    avg = sum(measurement) / n_samples
    assert jnp.allclose(avg, 0, atol=1e-2)


def test_reset():
    ops = ["Reset"]
    op_metaparams = [[0, 0]]
    param_inds = [[0]]

    state = 1 / jnp.sqrt(2) * jnp.array([1, 0, 0, 1]).reshape(2, 2)
    seed = 0

    param_to_st = get_params_to_statetensor_func(ops, op_metaparams, param_inds)
    rng = jax.random.PRNGKey(seed)

    for rng_i in jax.random.split(rng, 10):
        params = rng_i.reshape(1, *rng_i.shape)
        classical_registers = jnp.zeros(1)
        result, measurement = param_to_st(params, state, classical_registers)

        if measurement == 1:
            assert jnp.allclose(result, jnp.array([1, 0, 0, 0]).reshape(2, 2))
        elif measurement == -1:
            assert jnp.allclose(result, jnp.array([0, 1, 0, 0]).reshape(2, 2))
        else:
            raise ValueError("Measurement is not 1 or -1.")


def test_conditional_operation_dt():
    ops = ["ConditionalOperation"]
    op_metaparams = [[["X", "H"], [[0], [0]], [[], []]]]
    param_inds = [[0]]
    param_to_dt = get_params_to_densitytensor_func(ops, op_metaparams, param_inds)
    initial_densitytensor = all_zeros_densitytensor(1)
    jitted_param_to_dt = jax.jit(param_to_dt)
    res_1 = jitted_param_to_dt(jnp.array([0]), initial_densitytensor, jnp.array([]))[0]
    res_2 = jitted_param_to_dt(jnp.array([1]), initial_densitytensor, jnp.array([]))[0]
    assert jnp.allclose(res_1, jnp.array([[0.0, 0.0], [0.0, 1.0]]))
    assert jnp.allclose(res_2, jnp.array([[0.5, 0.5], [0.5, 0.5]]))


def test_double_conditional_dt():
    ops = ["ConditionalOperation"]
    op_metaparams = [
        [
            ["ConditionalGate", "ConditionalGate"],
            [[[["Rx"], ["Ry"]], [0]], [[["Rx"], ["Ry"]], [1]]],
            [[[{"conditional_ind":0}, {"angles": [[0]]}], [{"conditional_ind":1}, {"angles": [[1]]}]]]
        ]
    ]
    param_inds = [[{"op_ind": 0}, "params"]]

    rotation_params = jnp.array([.1, .2])
    params_1 = {"op_ind":0, "params":{"conditional_ind":jnp.array([0, 0]), "angles":rotation_params}}
    params_2 = {"op_ind":1, "params":{"conditional_ind":jnp.array([1, 1]), "angles":rotation_params}}

    param_to_dt = get_params_to_densitytensor_func(ops, op_metaparams, param_inds)
    jitted_param_to_dt = jax.jit(param_to_dt)

    initial_densitytensor = all_zeros_densitytensor(2)

    dt_1, _ = jitted_param_to_dt(params_1, initial_densitytensor)
    test_ops_1 = ["Rx"]
    test_op_metaparams_1 = [[0]]
    test_op_params_1 = [[0]]
    test_param_to_dt_1 = get_params_to_densitytensor_func(test_ops_1, test_op_metaparams_1, test_op_params_1)
    dt_test_1, _ = test_param_to_dt_1(rotation_params, initial_densitytensor)
    assert jnp.allclose(dt_1, dt_test_1)

    dt_2, _ = jitted_param_to_dt(params_2, initial_densitytensor)
    test_ops_2 = ["Ry"]
    test_op_metaparams_2 = [[1]]
    test_op_params_2 = [[1]]
    test_param_to_dt_2 = get_params_to_densitytensor_func(test_ops_2, test_op_metaparams_2, test_op_params_2)
    dt_test_2, _ = test_param_to_dt_2(rotation_params, initial_densitytensor)
    assert jnp.allclose(dt_2, dt_test_2)


