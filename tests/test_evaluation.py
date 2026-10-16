"""Auswertung: Einstellungen, eine Stichprobe, Zusammenfassung, die fünf Experimente (klein gefahren) und die Zufallsdiagramme."""

import numpy as np
import pytest

import ins_constants as C
import ins_evaluation as ev
import ins_world as W
from kausal_generator import SplitMix64


def test_number_formats():
    assert ev.de(-1.234) == "−1,23" and ev.de(0.5, 1) == "0,5" and ev.de(3.0, 0) == "3"
    assert ev.signed(0.3) == "+0,30" and ev.signed(-0.3) == "−0,30" and ev.signed(1e-12) == "+0,00" and ev.signed(-1e-12) == "+0,00"


def test_settings_build_the_world_and_default_to_the_base_case():
    s = ev.Settings(a=0.5, pi=0.7, beta=0.9, g=0.1, h=0.2)
    assert s.world == W.World(0.5, 0.7, 0.9, 0.1, 0.2)
    d = ev.Settings()
    assert (d.a, d.pi, d.beta, d.g, d.h, d.n, d.seed) == (1.0, 0.5, 1.0, 0.0, 0.0, C.DEFAULT_N, C.DEFAULT_SEED)


def test_analyse_rows_and_limits():
    a = ev.analyse(ev.Settings())
    assert [r.name for r in a.rows] == list(ev.ROW_NAMES) and a.truth == pytest.approx(1.0)
    by = {r.name: r for r in a.rows}
    assert by["Naiver Vergleich"].limit == pytest.approx(1 / 2.25 + 1.0) and by["Kontrolle für das Instrument"].limit == pytest.approx(1.5)
    assert by["2SLS (Instrument)"].limit == pytest.approx(1.0) and by["Orakel mit U (nicht beobachtbar)"].limit == pytest.approx(1.0)
    for r in a.rows:
        assert r.lo == pytest.approx(r.estimate - C.Z_CRIT * r.se) and r.hi == pytest.approx(r.estimate + C.Z_CRIT * r.se) and r.error == pytest.approx(r.estimate - a.truth)
        assert r.covers == (r.lo <= a.truth <= r.hi)
    assert a.first["pi_limit"] == pytest.approx(0.5) and a.first["expected_f"] == pytest.approx(500 * (0.25 / 2.25) / (1 - 0.25 / 2.25)) and a.first["f"] > 10
    assert a.ar["bounded"] is True and a.ar["covers"] == (a.ar["lo"] <= a.truth <= a.ar["hi"])


def test_analyse_is_deterministic_and_seed_dependent():
    assert ev.analyse(ev.Settings()).rows[2].estimate == ev.analyse(ev.Settings()).rows[2].estimate
    assert ev.analyse(ev.Settings(seed=1)).rows[2].estimate != ev.analyse(ev.Settings(seed=2)).rows[2].estimate


def test_ar_text_for_each_kind():
    assert ev.ar_text({"kind": "interval", "lo": 0.5, "hi": 1.25}) == "0,50 bis 1,25"
    assert ev.ar_text({"kind": "outside", "lo": 0.5, "hi": 1.25}) == "unbeschränkt (alles außer 0,50 bis 1,25)"
    assert ev.ar_text({"kind": "all", "lo": -np.inf, "hi": np.inf}) == "die ganze Gerade"


def test_summarize_by_hand():
    out = ev.summarize([0.0, 1.0, 2.0, 3.0, 100.0], 1.0)
    assert out["median"] == 2.0 and out["median_bias"] == 1.0 and out["mean"] == 21.2 and out["bias"] == pytest.approx(20.2) and out["iqr"] == 2.0
    assert out["sd"] == pytest.approx(np.std([0, 1, 2, 3, 100]))


def test_base_experiment_small_run():
    rows = ev.base_experiment(pi_values=(1.0, 0.05), n=300, reps=80)
    strong, weak = rows
    assert [r["pi"] for r in rows] == [1.0, 0.05] and strong["truth"] == pytest.approx(1.0)
    assert strong["share_weak"] == 0.0 and abs(strong["iv"]["median_bias"]) < 0.05 and strong["ols"]["median_bias"] == pytest.approx(1 / 3, abs=0.04)
    assert strong["iv"]["sd"] > 1.5 * strong["oracle"]["sd"] and strong["ar_bounded"] == 1.0 and strong["pretest"]["share"] == 1.0 and 0.85 <= strong["wald_cover"] <= 1.0
    assert weak["share_weak"] > 0.9 and weak["ar_bounded"] < 0.5 and weak["wald_width"] > 5 * strong["wald_width"] and weak["pretest"] is None
    assert rows == ev.base_experiment(pi_values=(1.0, 0.05), n=300, reps=80)


def test_violations_experiment_matches_the_closed_forms():
    by = {r["label"]: r for r in ev.violations_experiment()}
    assert len(by) == len(C.VIOLATIONS) and by["Grundfall (π = 0,5)"]["iv_error"] == pytest.approx(0.0, abs=1e-9)
    assert by["Z wirkt direkt auf y (0,1), π = 1,0"]["iv_error"] == pytest.approx(0.1) and by["Z wirkt direkt auf y (0,1), π = 0,5"]["iv_error"] == pytest.approx(0.2)
    assert by["Z wirkt direkt auf y (0,1), π = 0,3"]["iv_error"] == pytest.approx(0.1 / 0.3) and by["Z wirkt direkt auf y (0,1), π = 0,1"]["iv_error"] == pytest.approx(1.0)
    assert by["U wirkt auf Z (0,1), π = 1,0"]["iv_error"] == pytest.approx(0.1 / (1.01 + 0.1)) and by["U wirkt auf Z (0,3), π = 1,0"]["iv_error"] == pytest.approx(0.3 / (1.09 + 0.3))
    strong = by["starker Confounder (U wirkt mit 2,0)"]
    assert strong["iv_error"] == pytest.approx(0.0, abs=1e-9) and strong["naive_error"] == pytest.approx(4 / (0.25 + 5)) and strong["ratio"] == pytest.approx(0.0, abs=1e-9)
    assert by["Z wirkt direkt auf y (0,1), π = 0,1"]["ratio"] == pytest.approx(1.0 / by["Z wirkt direkt auf y (0,1), π = 0,1"]["naive_error"])


def test_overid_experiment_small_run():
    rows = {r["label"]: r for r in ev.overid_experiment(n=500, reps=150)}
    assert list(rows) == list(C.OVER_CASES) and rows["beide gültig"]["reject"] <= 0.15 and rows["beide gültig"]["truth"] == pytest.approx(1.0)
    assert rows["eines ungültig (Z₂ → Y 0,3)"]["reject"] >= 0.8 and rows["beide gleich ungültig (je 0,3)"]["reject"] <= 0.15
    assert rows["beide gleich ungültig (je 0,3)"]["iv"]["median_bias"] > 0.3 and rows["beide gültig, aber schwach (π = 0,15)"]["f_mean"] < 10 and rows["beide gültig"]["reps"] == 150


def test_late_experiment_small_run():
    res = ev.late_experiment(ns=(500, 1000), reps=100)
    p = res["population"]
    assert [r["n"] for r in res["rows"]] == [500, 1000] and p["late"] == pytest.approx(0.5) and p["ate"] == pytest.approx(0.26)
    for r in res["rows"]:
        assert r["wald"]["mean"] == pytest.approx(0.5, abs=0.05) and r["naive"]["mean"] == pytest.approx(p["naive"], abs=0.03) and r["used"] == 100
    assert res["rows"][1]["wald"]["sd"] < res["rows"][0]["wald"]["sd"]


def test_random_dag_is_topological_and_reproducible():
    parents = ev.random_dag(7, 0.4, SplitMix64(4))
    assert len(parents) == 7 and all(all(p < j for p in ps) for j, ps in enumerate(parents)) and parents == ev.random_dag(7, 0.4, SplitMix64(4))
    assert ev.random_dag(5, 0.0, SplitMix64(1)) == ((),) * 5 and sum(len(p) for p in ev.random_dag(5, 1.0, SplitMix64(1))) == 10


def test_diagram_experiment_small_run():
    res = ev.diagram_experiment(diagrams=200, nodes=7)
    assert res["diagrams"] == 200 and res["backdoor"] == 0 and 0 < res["instrument"] < 100 and 0 < res["frontdoor"] < 100 and res["wrong"] == 0
    assert res["either"] == res["instrument"] + res["frontdoor"] - res["both"] and res["both"] <= min(res["instrument"], res["frontdoor"])
    assert len(res["per_diagram"]) == 200 and sum(1 for k in res["per_diagram"] if k) == res["instrument"] and res["checked"] == sum(res["per_diagram"])
