"""Vehikel: Koeffizientenmatrix, Wahrheit, Population, Stichprobe und die binäre Ermutigung."""

import numpy as np
import pytest

import ins_constants as C
import ins_world as W
from kausal_generator import SplitMix64


def test_coefficient_matrix_and_parents():
    w = W.World(a=0.5, pi=0.7, beta=0.9, g=0.2, h=0.3, pi2=0.4, g2=0.1, h2=0.6)
    b = W.coefficient_matrix(w)
    assert b[W.X, W.U] == b[W.Y, W.U] == 0.5 and b[W.X, W.Z] == 0.7 and b[W.Y, W.X] == 0.9 and b[W.Y, W.Z] == 0.2 and b[W.Z, W.U] == 0.3
    assert b[W.X, W.Z2] == 0.4 and b[W.Y, W.Z2] == 0.1 and b[W.Z2, W.U] == 0.6 and b.sum() == pytest.approx(0.5 * 2 + 0.7 + 0.9 + 0.2 + 0.3 + 0.4 + 0.1 + 0.6)
    assert W.parents(W.World()) == ((), (), (), (0, 1), (0, 3))
    assert W.parents(W.World(g=0.1, h=0.1)) == ((), (0,), (), (0, 1), (0, 1, 3))


def test_nodes_are_in_topological_order():
    for j, ps in enumerate(W.parents(W.World(g=0.1, h=0.1, pi2=0.1, g2=0.1, h2=0.1))):
        assert all(p < j for p in ps)


def test_truth_is_beta_whatever_the_violations():
    assert W.truth(W.World()) == pytest.approx(1.0) and W.truth(W.World(beta=0.4)) == pytest.approx(0.4)
    assert W.truth(W.World(g=0.3, h=0.3, a=2.0, pi=0.1)) == pytest.approx(1.0)
    assert W.truth(W.World(pi2=0.5, g2=0.5, h2=0.5, beta=1.3)) == pytest.approx(1.3)


def test_population_by_hand(unit_world):
    cov, total = W.population(unit_world)
    assert cov[W.Z, W.Z] == pytest.approx(1.0) and cov[W.X, W.X] == pytest.approx(3.0) and cov[W.Z, W.X] == pytest.approx(1.0) and cov[W.U, W.X] == pytest.approx(1.0)
    assert cov[W.X, W.Y] == pytest.approx(4.0) and cov[W.Z, W.Y] == pytest.approx(1.0) and cov[W.U, W.Y] == pytest.approx(2.0) and cov[W.U, W.Z] == pytest.approx(0.0)
    assert total[W.Y, W.X] == pytest.approx(1.0) and total[W.X, W.Z] == pytest.approx(1.0) and total[W.Y, W.Z] == pytest.approx(1.0) and total[W.X, W.Y] == 0.0


def test_total_effects_have_a_zero_diagonal(unit_world):
    _, total = W.population(unit_world)
    assert np.diag(total) == pytest.approx(np.zeros(5)) and total[W.Y, W.U] == pytest.approx(2.0)          # U -> Y direkt 1 plus U -> X -> Y 1


def test_simulate_shapes_and_streams():
    w = W.World()
    v = W.simulate(w, 30, SplitMix64(3), reps=4)
    assert v.shape == (4, 30, 5)
    assert (W.simulate(w, 30, SplitMix64(3), reps=4) == v).all() and not (W.simulate(w, 30, SplitMix64(4), reps=4) == v).all()


def test_simulation_has_the_population_covariance(unit_world):
    emp = np.cov(W.simulate(unit_world, 200000, SplitMix64(6))[0].T)
    assert emp == pytest.approx(W.population(unit_world)[0], abs=0.06)


def test_late_population_by_hand():
    p = W.late_population()
    assert p["ate"] == pytest.approx(0.5 * 0.5 + 0.2 * 0.05 + 0.3 * 0.0) and p["late"] == pytest.approx(0.5)
    assert p["ey"][1] == pytest.approx(0.5 * 0.7 + 0.2 * 0.95 + 0.3 * 0.1) and p["ey"][0] == pytest.approx(0.5 * 0.2 + 0.2 * 0.95 + 0.3 * 0.1)
    assert p["ex"][1] == pytest.approx(0.7) and p["ex"][0] == pytest.approx(0.2) and p["wald"] == pytest.approx(0.5)
    ey_x1 = (0.2 * 0.95 + 0.5 * 0.5 * 0.7) / 0.45
    ey_x0 = (0.3 * 0.1 + 0.5 * 0.5 * 0.2) / 0.55
    assert p["naive"] == pytest.approx(ey_x1 - ey_x0)


def test_late_simulation_matches_the_probabilities():
    typ, z, x, y = (a[0] for a in W.simulate_late(400000, SplitMix64(7)))
    assert [np.mean(typ == k) for k in range(3)] == pytest.approx(list(C.LATE_SHARES), abs=0.005) and z.mean() == pytest.approx(0.5, abs=0.005)
    assert (x[typ == 0] == z[typ == 0]).all() and (x[typ == 1] == 1).all() and (x[typ == 2] == 0).all()
    assert y[(typ == 0) & (x == 1)].mean() == pytest.approx(0.7, abs=0.006) and y[(typ == 0) & (x == 0)].mean() == pytest.approx(0.2, abs=0.006)
    assert y[(typ == 1) & (x == 1)].mean() == pytest.approx(0.95, abs=0.006) and y[typ == 2].mean() == pytest.approx(0.1, abs=0.006)
    p = W.late_population()
    assert (y[z == 1].mean() - y[z == 0].mean()) / (x[z == 1].mean() - x[z == 0].mean()) == pytest.approx(p["wald"], abs=0.01)
    assert y[x == 1].mean() - y[x == 0].mean() == pytest.approx(p["naive"], abs=0.006)


def test_late_types_follow_the_constants_not_a_symmetry(monkeypatch):
    """Mit anderen Anteilen (0,2 / 0,5 / 0,3) und P(Z = 1) = 0,3 ändern sich die exakten Größen; die Rechnung benutzt die Konstanten."""
    monkeypatch.setattr(C, "LATE_SHARES", (0.2, 0.5, 0.3))
    monkeypatch.setattr(C, "LATE_PZ", 0.3)
    p = W.late_population()
    assert p["ate"] == pytest.approx(0.2 * 0.5 + 0.5 * 0.05) and p["ex"][1] == pytest.approx(0.2 + 0.5) and p["ex"][0] == pytest.approx(0.5)
    px1 = 0.5 + 0.2 * 0.3
    assert p["naive"] == pytest.approx((0.5 * 0.95 + 0.2 * 0.3 * 0.7) / px1 - (0.3 * 0.1 + 0.2 * 0.7 * 0.2) / (1 - px1))
