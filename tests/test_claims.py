"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SplitMix64-Ströme (ganzzahlig, plattformstabil); Fließkomma-Reihenfolgen können je Bibliotheksversion die letzte Stelle ändern, deshalb
Toleranzen von einer halben Einheit der gedruckten Stelle. Bei schwachem Instrument stehen nur Median, Quartile, Deckung und Breiten im Befund, nie das Mittel des 2SLS.
"""

import pytest

import ins_constants as C
import ins_evaluation as ev


@pytest.fixture(scope="module")
def base():
    return {r["pi"]: r for r in ev.base_experiment()}


@pytest.fixture(scope="module")
def viol():
    return {r["label"]: r for r in ev.violations_experiment()}


@pytest.fixture(scope="module")
def over():
    return {r["label"]: r for r in ev.overid_experiment()}


@pytest.fixture(scope="module")
def late():
    return ev.late_experiment()


@pytest.fixture(scope="module")
def diagrams():
    return ev.diagram_experiment()


def test_default_sample():
    a = ev.analyse(ev.Settings())
    by = {r.name: r for r in a.rows}
    assert a.truth == pytest.approx(1.0) and by["Naiver Vergleich"].estimate == pytest.approx(1.43, abs=0.005) and by["Kontrolle für das Instrument"].estimate == pytest.approx(1.50, abs=0.005)
    assert by["2SLS (Instrument)"].estimate == pytest.approx(1.04, abs=0.005) and by["2SLS (Instrument)"].se == pytest.approx(0.113, abs=0.0006) and by["2SLS (Instrument)"].covers is True
    assert by["Orakel mit U (nicht beobachtbar)"].estimate == pytest.approx(1.02, abs=0.005) and by["Naiver Vergleich"].covers is False and by["Kontrolle für das Instrument"].covers is False
    assert a.first["f"] == pytest.approx(79.1, abs=0.05) and a.first["pi"] == pytest.approx(0.550, abs=0.0006) and a.first["se_pi"] == pytest.approx(0.062, abs=0.0006)
    assert a.ar["kind"] == "interval" and a.ar["lo"] == pytest.approx(0.79, abs=0.005) and a.ar["hi"] == pytest.approx(1.25, abs=0.005) and a.ar["covers"] is True
    assert ev.verdict_instance(a)[0] == "success"


def test_experiment_1_strong_instrument(base):
    s = base[1.0]
    assert C.EXP_N == 500 and C.EXP_REPS == 500 and s["f_mean"] == pytest.approx(253.1, abs=0.05) and s["share_weak"] == 0.0
    assert s["iv"]["median_bias"] == pytest.approx(0.000, abs=0.0006) and s["iv"]["sd"] == pytest.approx(0.065, abs=0.0006) and s["oracle"]["sd"] == pytest.approx(0.032, abs=0.0006)
    assert s["ols"]["median_bias"] == pytest.approx(0.334, abs=0.0006) and s["wald_cover"] == pytest.approx(0.95, abs=0.005) and s["ar_cover"] == pytest.approx(0.95, abs=0.005)
    d = base[0.5]
    assert d["f_mean"] == pytest.approx(63.8, abs=0.05) and d["iv"]["median_bias"] == pytest.approx(-0.001, abs=0.0006) and d["iv"]["sd"] == pytest.approx(0.132, abs=0.0006)
    assert d["oracle"]["sd"] == pytest.approx(0.040, abs=0.0006) and d["ols"]["median_bias"] == pytest.approx(0.445, abs=0.0006) and d["wald_width"] == pytest.approx(0.50, abs=0.005)
    assert d["iv"]["sd"] / d["oracle"]["sd"] == pytest.approx(3.3, abs=0.05) and s["iv"]["sd"] / s["oracle"]["sd"] == pytest.approx(2.0, abs=0.05)


def test_experiment_1_weakening_instrument(base):
    for pi, f, weak in ((0.3, 22.5, 0.07), (0.2, 11.1, 0.53), (0.1, 3.6, 0.94), (0.05, 1.6, 1.0)):
        assert base[pi]["f_mean"] == pytest.approx(f, abs=0.05) and base[pi]["share_weak"] == pytest.approx(weak, abs=0.005), pi
    for pi, med, iqr in ((0.3, -0.024, 0.288), (0.2, 0.022, 0.477), (0.1, 0.022, 0.921), (0.05, 0.314, 1.322)):
        assert base[pi]["iv"]["median_bias"] == pytest.approx(med, abs=0.0006) and base[pi]["iv"]["iqr"] == pytest.approx(iqr, abs=0.0006), pi
    assert base[0.2]["iv"]["sd"] == pytest.approx(0.40, abs=0.005) and base[0.1]["iv"]["sd"] == pytest.approx(4.9, abs=0.05) and base[0.05]["iv"]["sd"] > 100
    assert base[0.05]["ols"]["median_bias"] == pytest.approx(0.50, abs=0.005) and base[0.05]["iv"]["median_bias"] < base[0.05]["ols"]["median_bias"]


def test_experiment_1_intervals(base):
    for pi, cover in ((0.3, 0.958), (0.2, 0.946), (0.1, 0.968), (0.05, 0.982)):
        assert base[pi]["wald_cover"] == pytest.approx(cover, abs=0.0006), pi
    for pi, width in ((0.5, 0.50), (0.3, 0.86), (0.2, 1.25), (0.1, 2.74), (0.05, 5.32)):
        assert base[pi]["wald_width"] == pytest.approx(width, abs=0.005), pi
    for pi, bounded in ((0.2, 0.89), (0.1, 0.372), (0.05, 0.136)):
        assert base[pi]["ar_bounded"] == pytest.approx(bounded, abs=0.0006), pi
    assert all(0.93 <= base[pi]["ar_cover"] <= 0.96 for pi in base) and base[0.1]["ar_width"] == pytest.approx(2.62, abs=0.005)


def test_experiment_1_pretest_on_f(base):
    p2, p1 = base[0.2]["pretest"], base[0.1]["pretest"]
    assert p2["share"] == pytest.approx(0.472, abs=0.0006) and p2["median_bias"] == pytest.approx(0.132, abs=0.0006) and p2["cover"] == pytest.approx(0.89, abs=0.005)
    assert p1["share"] == pytest.approx(0.058, abs=0.0006) and p1["median_bias"] == pytest.approx(0.349, abs=0.0006) and p1["cover"] == pytest.approx(0.72, abs=0.005)
    assert base[0.05]["pretest"] is None and abs(base[0.3]["pretest"]["median_bias"]) < 0.01


def test_experiment_2_violations(viol):
    assert viol["Z wirkt direkt auf y (0,1), π = 1,0"]["iv_error"] == pytest.approx(0.10, abs=0.005) and viol["Z wirkt direkt auf y (0,1), π = 0,5"]["iv_error"] == pytest.approx(0.20, abs=0.005)
    assert viol["Z wirkt direkt auf y (0,1), π = 0,3"]["iv_error"] == pytest.approx(0.33, abs=0.005) and viol["Z wirkt direkt auf y (0,1), π = 0,1"]["iv_error"] == pytest.approx(1.00, abs=0.005)
    assert viol["Z wirkt direkt auf y (0,1), π = 0,1"]["naive_error"] == pytest.approx(0.50, abs=0.005) and viol["Z wirkt direkt auf y (0,1), π = 0,1"]["ratio"] == pytest.approx(1.99, abs=0.005)
    assert viol["Z wirkt direkt auf y (0,1), π = 1,0"]["ratio"] == pytest.approx(0.27, abs=0.005) and viol["Z wirkt direkt auf y (0,1), π = 1,0"]["naive_error"] == pytest.approx(0.37, abs=0.005)
    assert viol["U wirkt auf Z (0,1), π = 1,0"]["iv_error"] == pytest.approx(0.09, abs=0.005) and viol["U wirkt auf Z (0,1), π = 0,3"]["iv_error"] == pytest.approx(0.25, abs=0.005)
    assert viol["U wirkt auf Z (0,3), π = 1,0"]["iv_error"] == pytest.approx(0.22, abs=0.005)
    strong = viol["starker Confounder (U wirkt mit 2,0)"]
    assert strong["iv_error"] == pytest.approx(0.0, abs=0.005) and strong["naive_error"] == pytest.approx(0.76, abs=0.005) and strong["control"] == pytest.approx(1.8, abs=0.005)


def test_a_connection_from_u_to_z_inflates_the_first_stage():
    import ins_estimators as E
    from ins_world import World

    assert E.expected_f(World(pi=0.2), 500) == pytest.approx(10.0, abs=0.05) and E.expected_f(World(pi=0.2, h=0.3), 500) == pytest.approx(64.2, abs=0.05)
    assert E.expected_f(World(pi=0.2, h=0.3), 500) > 5 * E.expected_f(World(pi=0.2), 500)


def test_experiment_3_overidentification(over):
    assert C.EXP_OVER_N == 500 and C.EXP_OVER_REPS == 1000
    for label, reject, med, iqr in (("beide gültig", 0.054, 0.005, 0.084), ("eines ungültig (Z₂ → Y 0,3)", 0.95, 0.216, 0.079), ("beide gleich ungültig (je 0,3)", 0.059, 0.429, 0.075),
                                    ("eines schwach ungültig (Z₂ → Y 0,15)", 0.401, 0.112, 0.086), ("beide gültig, aber schwach (π = 0,15)", 0.051, 0.059, 0.364)):
        assert over[label]["reject"] == pytest.approx(reject, abs=0.0006) and over[label]["iv"]["median_bias"] == pytest.approx(med, abs=0.0006), label
        assert over[label]["iv"]["iqr"] == pytest.approx(iqr, abs=0.0006), label
    assert over["beide gültig"]["f_mean"] == pytest.approx(124.2, abs=0.05) and over["beide gültig, aber schwach (π = 0,15)"]["f_mean"] == pytest.approx(6.8, abs=0.05)


def test_experiment_4_late(late):
    p = late["population"]
    assert p["ate"] == pytest.approx(0.26, abs=0.005) and p["late"] == pytest.approx(0.50, abs=0.005) and p["naive"] == pytest.approx(0.67, abs=0.005) and p["wald"] == pytest.approx(0.50, abs=0.005)
    r500, r2000 = late["rows"]
    assert (r500["n"], r2000["n"]) == (500, 2000) and r500["reps"] == 500
    assert r500["wald"]["mean"] == pytest.approx(0.498, abs=0.0006) and r500["wald"]["sd"] == pytest.approx(0.069, abs=0.0006) and r2000["wald"]["mean"] == pytest.approx(0.503, abs=0.0006)
    assert r2000["wald"]["sd"] == pytest.approx(0.034, abs=0.0006) and r500["naive"]["mean"] == pytest.approx(0.667, abs=0.0006) and r500["naive"]["sd"] == pytest.approx(0.035, abs=0.0006)
    assert r2000["naive"]["sd"] == pytest.approx(0.017, abs=0.0006)


def test_experiment_5_diagrams(diagrams):
    assert diagrams["diagrams"] == 3000 and diagrams["nodes"] == 7 and diagrams["backdoor"] == 0 and diagrams["instrument"] == 304 and diagrams["frontdoor"] == 265
    assert diagrams["both"] == 15 and diagrams["either"] == 554 and diagrams["instrument"] / 3000 == pytest.approx(0.101, abs=0.0005) and diagrams["frontdoor"] / 3000 == pytest.approx(0.088, abs=0.0005)
    assert diagrams["either"] / 3000 == pytest.approx(0.185, abs=0.0005)
    k = diagrams["per_diagram"]
    assert k.count(0) == 2696 and k.count(1) == 225 and k.count(2) == 58 and sum(1 for v in k if v >= 3) == 21
    assert diagrams["checked"] == 410 and diagrams["wrong"] == 0


def test_the_text_numbers_of_the_app_follow_the_findings(base, viol, diagrams):
    assert "bei π = 1,0 um +0,10" in ev.verdict_violations(list(viol.values()))[1] and "bei π = 0,1 um +1,00" in ev.verdict_violations(list(viol.values()))[1]
    assert "10,1 %" in ev.verdict_diagrams(diagrams)[1] and "8,8 %" in ev.verdict_diagrams(diagrams)[1]
    assert "in 63 % unbeschränkt" in ev.verdict_base(list(base.values()))[1]
