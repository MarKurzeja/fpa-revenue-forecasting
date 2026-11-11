import numpy as np

from fpa.hierarchy import is_coherent, node_id, store_path_parents


def test_levels_and_summing_matrix(hier):
    counts = hier.nodes.groupby("level").size().to_dict()
    # 1 total, 2 states, 3 stores, 2 cats, 3 depts, 4 state x cat, 6 store x cat, 7 bottom
    assert counts == {1: 1, 2: 2, 3: 3, 4: 2, 5: 3, 6: 4, 7: 6, 8: 7}
    assert hier.S.shape == (28, 7)
    assert np.allclose(hier.S[hier.bottom_idx], np.eye(7))
    assert hier.S[hier.total_idx].sum() == 7
    tx = hier.node_index("TX")
    assert hier.S[tx].sum() == 3
    assert hier.nodes.loc[hier.node_index("CA_1/FOODS_1"), "state_id"] == "CA"


def test_aggregate_and_coherence(hier):
    yb = np.arange(1, 8, dtype=float)
    y = hier.aggregate(yb)
    assert y[hier.total_idx] == yb.sum()
    assert is_coherent(hier, y)
    y[hier.total_idx] += 1
    assert not is_coherent(hier, y)


def test_node_id_and_paths(hier):
    assert node_id(1, {}) == "Total"
    assert node_id(7, {"store_id": "CA_1", "cat_id": "FOODS"}) == "CA_1/FOODS"
    paths = store_path_parents(hier)
    j = hier.node_index("TX_1/FOODS_2")
    tot, st, so = paths[j]
    assert hier.nodes.loc[st, "node_id"] == "TX" and hier.nodes.loc[so, "node_id"] == "TX_1"
