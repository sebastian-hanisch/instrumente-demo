"""Verdikte: Art und Text je Lage; jede Zahl im Text stammt aus dem Befund, nicht aus einer Konstanten."""

import ins_evaluation as ev


def test_instance_verdict_in_the_base_case_is_a_success():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings()))
    assert kind == "success" and "Der 2SLS schätzt" in text and "deckt" in text and "verfehlt die Wahrheit" not in text and "F = " in text
    assert "naive Vergleich liegt um +0," in text and "Kontrolle für das Instrument um +0," in text


def test_instance_verdict_names_every_active_violation():
    for kw, phrase in ((dict(g=0.1), "direkten Pfeil"), (dict(h=0.1), "Standortqualität")):
        kind, text = ev.verdict_instance(ev.analyse(ev.Settings(**kw)))
        assert kind == "warning" and "Eine Annahme ist verletzt" in text and phrase in text, kw
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(g=0.1, h=0.2)))
    assert "direkten Pfeil" in text and "Standortqualität" in text


def test_instance_verdict_warns_about_a_weak_instrument():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(pi=0.1)))
    assert kind == "warning" and "Das Instrument ist schwach" in text and "Anderson-Rubin-Menge ist" in text and "Eine Annahme ist verletzt" not in text


def test_a_violation_is_named_before_a_weak_instrument():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(pi=0.1, g=0.1)))
    assert kind == "warning" and "Eine Annahme ist verletzt" in text and "schwach" not in text


def test_instance_verdict_is_info_when_the_interval_misses_without_a_violation():
    for seed in range(300):
        a = ev.analyse(ev.Settings(n=200, seed=seed))
        if a.first["f"] >= 10 and not next(r for r in a.rows if r.name == "2SLS (Instrument)").covers:
            kind, text = ev.verdict_instance(a)
            assert kind == "info" and "verfehlt" in text
            return
    raise AssertionError("bei n = 200 und 300 Seeds sollte mindestens ein Intervall bei starkem Instrument verfehlen")


def test_base_verdict_reads_the_first_and_the_weak_row():
    rows = ev.base_experiment(pi_values=(1.0, 0.1), n=300, reps=60)
    kind, text = ev.verdict_base(rows)
    first, weak = rows
    assert kind == "info" and ev.de(first["iv"]["sd"], 3) in text and ev.de(first["oracle"]["sd"], 3) in text and ev.de(weak["f_mean"], 1) in text and "π = 0,1" in text
    assert ev.de(weak["iv"]["iqr"], 2) in text and f"in {ev.de(100 * (1 - weak['ar_bounded']), 0)} % unbeschränkt" in text


def test_violation_verdict_quotes_the_exact_values():
    kind, text = ev.verdict_violations(ev.violations_experiment())
    assert kind == "info" and "bei π = 1,0 um +0,10" in text and "bei π = 0,1 um +1,00" in text and "g / π" in text and "(+0,00)" in text


def test_overid_verdict_quotes_the_rows():
    rows = ev.overid_experiment(n=500, reps=60)
    kind, text = ev.verdict_overid(rows)
    by = {r["label"]: r for r in rows}
    assert kind == "info" and f"lehnt der J-Test in {ev.de(100 * by['beide gültig']['reject'], 0)} % ab" in text and "blind" in text
    assert f"nur in {ev.de(100 * by['beide gleich ungültig (je 0,3)']['reject'], 0)} % ab" in text


def test_late_verdict_names_late_ate_and_naive():
    res = ev.late_experiment(ns=(500, 800), reps=40)
    kind, text = ev.verdict_late(res)
    assert kind == "info" and "bei den Compliern (0,50, Mittel" in text and "n = 800" in text and "(0,26)" in text and "naive Vergleich zeigt 0,67" in text


def test_diagram_verdict_counts():
    res = ev.diagram_experiment(diagrams=100, nodes=7)
    kind, text = ev.verdict_diagrams(res)
    assert kind == "info" and "In 100 Zufallsdiagrammen" in text and f"({res['backdoor']})" in text and f"in {res['instrument']} (" in text and f"in nur {res['both']} beides" in text
    assert f"{res['checked'] - res['wrong']} von {res['checked']}" in text
