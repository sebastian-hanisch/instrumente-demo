"""Instrument-Kriterium: Hand-Diagramme, Zerlegung in die zwei Bedingungen, Brute-Force-Gegenprobe über networkx; Frontdoor-Prüfung (wortgleich aus Stück 8)."""

import itertools

import numpy as np
import pytest

import ins_criterion as K

nx = pytest.importorskip("networkx")


def test_classic_diagram(classic):
    assert K.instrument_check(classic, 2, 3, 1) == {"relevant": True, "excluded": True, "valid": True}
    assert K.instruments(classic, 2, 3, [0, 1, 2, 3]) == [1] and K.instruments(classic, 2, 3, [1]) == [1] and K.instruments(classic, 2, 3, [0]) == []


def test_the_hidden_confounder_is_no_instrument(classic):
    chk = K.instrument_check(classic, 2, 3, 0)
    assert chk["relevant"] is True and chk["excluded"] is False and chk["valid"] is False                 # U -> Y direkt


def test_direct_edge_breaks_the_exclusion():
    parents = ((), (), (0, 1), (0, 1, 2))                                                                # zusätzlich Z -> Y
    chk = K.instrument_check(parents, 2, 3, 1)
    assert chk["relevant"] is True and chk["excluded"] is False and chk["valid"] is False


def test_common_cause_of_z_and_y_breaks_the_independence():
    parents = ((), (0,), (0, 1), (0, 2))                                                                 # U -> Z
    chk = K.instrument_check(parents, 2, 3, 1)
    assert chk["relevant"] is True and chk["excluded"] is False and chk["valid"] is False


def test_irrelevant_node_is_no_instrument():
    parents = ((), (), (0,), (0, 2))                                                                     # Z hat keinen Pfeil zu X
    chk = K.instrument_check(parents, 2, 3, 1)
    assert chk["relevant"] is False and chk["excluded"] is True and chk["valid"] is False


def test_a_descendant_of_x_is_no_instrument():
    parents = ((), (0,), (1,), (0, 1, 2))                                                                # 0 = U, 1 = X, 2 = Nachfahre von X, 3 = Y
    assert K.instrument_check(parents, 1, 3, 2)["valid"] is False


def test_a_cause_of_the_instrument_is_also_one():
    parents = ((), (), (1,), (0, 2), (0, 3))                                                             # 1 -> 2 -> 3 -> 4 und 0 -> 3, 0 -> 4: Instrument 1 über 2, x = 3, y = 4
    assert K.instrument_valid(parents, 3, 4, 2) and K.instrument_valid(parents, 3, 4, 1)


def test_endpoints_are_rejected(classic):
    with pytest.raises(ValueError):
        K.instrument_check(classic, 2, 3, 2)
    with pytest.raises(ValueError):
        K.instrument_check(classic, 2, 2, 1)


def _surgery(parents, x, y, z):
    """Zweiter Rechenweg über networkx: dieselben zwei Bedingungen mit nx.is_d_separator."""
    n = len(parents)
    edges = [(p, c) for c, ps in enumerate(parents) for p in ps]
    full = nx.DiGraph()
    full.add_nodes_from(range(n))
    full.add_edges_from(edges)
    cut = nx.DiGraph()
    cut.add_nodes_from(range(n))
    cut.add_edges_from([(p, c) for p, c in edges if c != x])
    return (not nx.is_d_separator(full, {z}, {x}, set())) and nx.is_d_separator(cut, {z}, {y}, set())


def test_criterion_against_networkx_on_random_diagrams():
    rng = np.random.default_rng(5)
    checked = valid = 0
    for _ in range(300):
        n = 6
        p = rng.uniform(0.25, 0.55)
        parents = tuple(tuple(i for i in range(j) if rng.random() < p) for j in range(n))
        x, y = sorted(int(v) for v in rng.choice(n, 2, replace=False))
        for z in (v for v in range(n) if v not in (x, y)):
            ok = K.instrument_valid(parents, x, y, z)
            assert ok == _surgery(parents, x, y, z), (parents, x, y, z)
            checked += 1
            valid += ok
    assert checked == 1200 and valid > 20


def test_frontdoor_check_on_the_classic_mediator_diagram():
    parents = ((), (0,), (1,), (0, 2))                                                                   # U, X, M, Y
    assert K.frontdoor_check(parents, 1, 3, (2,)) == {"intercepts": True, "no_backdoor": True, "blocked": True, "valid": True}
    assert K.frontdoor_sets(parents, 1, 3, [1, 2, 3]) == [(2,)] and K.frontdoor_sets(parents, 1, 3, [0]) == []
    direct = ((), (0,), (1,), (0, 1, 2))
    assert K.frontdoor_check(direct, 1, 3, (2,))["intercepts"] is False
    with pytest.raises(ValueError):
        K.frontdoor_check(parents, 1, 3, (1,))


def test_instrument_and_frontdoor_are_different_corridors():
    instrument_only = ((), (), (0, 1), (0, 2))
    frontdoor_only = ((), (0,), (1,), (0, 2))
    assert K.instruments(instrument_only, 2, 3, [1]) == [1] and K.frontdoor_sets(instrument_only, 2, 3, [1]) == []
    assert K.instruments(frontdoor_only, 1, 3, [2]) == [] and K.frontdoor_sets(frontdoor_only, 1, 3, [2]) == [(2,)]
    both = ((), (), (0, 1), (2,), (0, 3))                                                                # Z -> X -> M -> Y mit U -> X, U -> Y: beides
    assert K.instruments(both, 2, 4, [1, 3]) == [1] and K.frontdoor_sets(both, 2, 4, [1, 3]) == [(3,)]
