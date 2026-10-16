"""Auswertung: Einstellungen, eine Stichprobe mit allen Schätzern, die fünf Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import bdr_backdoor as B
import ins_constants as C
import ins_criterion as K
import ins_estimators as E
import ins_world as W
from kausal_generator import SplitMix64


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    x = 0.0 if abs(x) < 0.5 * 10 ** -digits * 1e-3 else x          # −0,00 aus Rundungsresten vermeiden
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


@dataclass(frozen=True)
class Settings:
    a: float = C.DEFAULT_A
    pi: float = C.DEFAULT_PI
    beta: float = C.DEFAULT_BETA
    g: float = 0.0
    h: float = 0.0
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED

    @property
    def world(self) -> W.World:
        return W.World(self.a, self.pi, self.beta, self.g, self.h)


ROW_NAMES = ("Naiver Vergleich", "Kontrolle für das Instrument", "2SLS (Instrument)", "Orakel mit U (nicht beobachtbar)")


@dataclass
class Row:
    name: str
    estimate: float
    se: float
    lo: float
    hi: float
    error: float
    covers: bool
    limit: float          # Grenzwert bei n → ∞


@dataclass
class Analysis:
    settings: Settings
    rows: list
    truth: float
    first: dict           # erste Stufe: F, π̂ mit Standardfehler, Grenzwert, erwartetes F
    ar: dict              # Anderson-Rubin-Menge und ob sie die Wahrheit enthält
    population: dict


def _row(name, est, se, truth, limit) -> Row:
    return Row(name, est, se, est - C.Z_CRIT * se, est + C.Z_CRIT * se, est - truth, bool(abs(est - truth) <= C.Z_CRIT * se), limit)


def ar_text(ar: dict) -> str:
    if ar["kind"] == "interval":
        return f"{de(ar['lo'], 2)} bis {de(ar['hi'], 2)}"
    if ar["kind"] == "outside":
        return f"unbeschränkt (alles außer {de(ar['lo'], 2)} bis {de(ar['hi'], 2)})"
    return "die ganze Gerade" if ar["kind"] == "all" else "leer"


def analyse(s: Settings) -> Analysis:
    w = s.world
    v = W.simulate(w, s.n, SplitMix64(s.seed))[0]
    x, y, z, u = v[:, W.X], v[:, W.Y], v[:, W.Z], v[:, W.U]
    truth = W.truth(w)
    pop = E.population_estimates(w)
    t = E.tsls(x, [z], y)
    beta1, cov1 = E.ols(x, [z])
    rows = [_row("Naiver Vergleich", *E.naive(x, y), truth, pop["naive"]), _row("Kontrolle für das Instrument", *E.control_instrument(x, z, y), truth, pop["control"]),
            _row("2SLS (Instrument)", t["estimate"], t["se"], truth, pop["iv"]), _row("Orakel mit U (nicht beobachtbar)", *E.oracle(x, u, y), truth, pop["oracle"])]
    ar = E.anderson_rubin(x, z, y)
    ar = {**ar, "covers": E.ar_contains(ar, truth), "bounded": E.ar_bounded(ar)}
    first = {"f": t["f"], "pi": float(beta1[1]), "se_pi": float(np.sqrt(cov1[1, 1])), "pi_limit": pop["pi"], "expected_f": E.expected_f(w, s.n)}
    return Analysis(s, rows, truth, first, ar, pop)


def summarize(values, truth: float) -> dict:
    """Median, Streuung und Quartile: Mittel und SD allein sind bei schwachem Instrument wertlos (die Momente des 2SLS existieren nicht)."""
    v = np.asarray(values, dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"mean": float(v.mean()), "median": float(med), "bias": float(v.mean() - truth), "median_bias": float(med - truth), "sd": float(v.std()), "iqr": float(q3 - q1)}


# ----------------------------------------------------------------------------- Experiment 1: Stärke des Instruments

def base_experiment(pi_values=C.EXP_PI_VALUES, n: int = C.EXP_N, reps: int = C.EXP_REPS) -> list[dict]:
    """Je Stärke π des Instruments: OLS, 2SLS und Orakel über feste Wiederholungen; erste Stufe (F), Intervalle (Wald und Anderson-Rubin) und der Vortest „F > 10“."""
    rows = []
    for i, pi in enumerate(pi_values):
        w = W.World(pi=pi)
        truth = W.truth(w)
        v = W.simulate(w, n, SplitMix64(C.SEED_BASE + i), reps)
        ols_est, iv_est, orc_est, f_all, wald_cover, wald_width, ar_cover, ar_bounded, ar_width = [], [], [], [], [], [], [], [], []
        for r in range(reps):
            x, y, z, u = v[r, :, W.X], v[r, :, W.Y], v[r, :, W.Z], v[r, :, W.U]
            t = E.tsls(x, [z], y)
            ar = E.anderson_rubin(x, z, y)
            ols_est.append(E.naive(x, y)[0])
            orc_est.append(E.oracle(x, u, y)[0])
            iv_est.append(t["estimate"])
            f_all.append(t["f"])
            wald_cover.append(abs(t["estimate"] - truth) <= C.Z_CRIT * t["se"])
            wald_width.append(2 * C.Z_CRIT * t["se"])
            ar_cover.append(E.ar_contains(ar, truth))
            ar_bounded.append(E.ar_bounded(ar))
            if E.ar_bounded(ar):
                ar_width.append(ar["hi"] - ar["lo"])
        f_all, iv_est, wald_cover = np.asarray(f_all), np.asarray(iv_est), np.asarray(wald_cover)
        strong = f_all > C.F_RULE
        pretest = None
        if strong.sum() >= 20:
            pretest = {"share": float(strong.mean()), "median_bias": float(np.median(iv_est[strong]) - truth), "cover": float(wald_cover[strong].mean())}
        rows.append({"pi": pi, "truth": truth, "f_mean": float(f_all.mean()), "f_median": float(np.median(f_all)), "share_weak": float(np.mean(f_all < C.F_RULE)),
                     "ols": summarize(ols_est, truth), "iv": summarize(iv_est, truth), "oracle": summarize(orc_est, truth),
                     "wald_cover": float(wald_cover.mean()), "wald_width": float(np.median(wald_width)), "ar_cover": float(np.mean(ar_cover)), "ar_bounded": float(np.mean(ar_bounded)),
                     "ar_width": float(np.median(ar_width)) if ar_width else float("nan"), "pretest": pretest})
    return rows


# ----------------------------------------------------------------------------- Experiment 2: Verletzungen (Population)

def violations_experiment() -> list[dict]:
    """Exakte Grenzwerte der Schätzer, wenn eine Annahme verletzt ist; die Wahrheit ist die Wirkung β (sie ändert sich durch die Verletzung nicht)."""
    out = []
    for label, kw in C.VIOLATIONS.items():
        pop = E.population_estimates(W.World(**kw))
        iv_error, naive_error = pop["iv"] - pop["truth"], pop["naive"] - pop["truth"]
        out.append({"label": label, "truth": pop["truth"], "naive": pop["naive"], "control": pop["control"], "iv": pop["iv"], "iv_error": iv_error, "naive_error": naive_error,
                    "ratio": abs(iv_error) / abs(naive_error) if naive_error != 0 else float("nan")})
    return out


# ----------------------------------------------------------------------------- Experiment 3: zwei Instrumente und der J-Test

def overid_experiment(n: int = C.EXP_OVER_N, reps: int = C.EXP_OVER_REPS) -> list[dict]:
    rows = []
    for i, (label, kw) in enumerate(C.OVER_CASES.items()):
        w = W.World(**kw)
        v = W.simulate(w, n, SplitMix64(C.SEED_BASE + 50 + i), reps)
        est, f_all, reject = [], [], 0
        for r in range(reps):
            t = E.tsls(v[r, :, W.X], [v[r, :, W.Z], v[r, :, W.Z2]], v[r, :, W.Y])
            est.append(t["estimate"])
            f_all.append(t["f"])
            reject += E.sargan(t["resid"], t["zc"]) > C.CHI2_CRIT
        rows.append({"label": label, "truth": W.truth(w), "reject": reject / reps, "reps": reps, "iv": summarize(est, W.truth(w)), "f_mean": float(np.mean(f_all))})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: binäre Ermutigung, Complier und LATE

def late_experiment(ns=C.EXP_LATE_NS, reps: int = C.EXP_LATE_REPS) -> dict:
    pop = W.late_population()
    rows = []
    for i, n in enumerate(ns):
        _, z, x, y = W.simulate_late(n, SplitMix64(C.SEED_BASE + 60 + i), reps)
        wald, naive = [], []
        for r in range(reps):
            wd, nv = E.wald_binary(z[r], x[r], y[r]), E.naive_binary(x[r], y[r])
            if wd is not None and nv is not None:
                wald.append(wd)
                naive.append(nv)
        rows.append({"n": n, "wald": summarize(wald, pop["late"]), "naive": summarize(naive, pop["late"]), "reps": reps, "used": len(wald)})
    return {"population": pop, "rows": rows}


# ----------------------------------------------------------------------------- Experiment 5: Zufallsdiagramme mit verborgener gemeinsamer Ursache

def random_dag(n: int, p: float, rng: SplitMix64) -> tuple:
    """Zufälliges Diagramm in der Knotenreihenfolge 0..n−1 mit Kantenwahrscheinlichkeit p (ein Block Gleichverteilte, feste Reihenfolge)."""
    u = rng.uniform(n * (n - 1) // 2)
    k = 0
    parents = []
    for j in range(n):
        ps = []
        for i in range(j):
            if u[k] < p:
                ps.append(i)
            k += 1
        parents.append(tuple(ps))
    return tuple(parents)


def diagram_experiment(diagrams: int = C.EXP_DIAGRAMS, nodes: int = C.EXP_NODES) -> dict:
    """Zufallsdiagramme mit der verborgenen gemeinsamen Ursache u (Knoten 0) von x und y und mindestens einem kausalen Pfad; alle anderen Knoten sind beobachtet.
    Gezählt wird, wie oft ein gültiges Instrument, eine Frontdoor-Menge und eine Backdoor-Menge existiert; Gegenprobe: In einer linearen Population mit zufälligen
    Koeffizienten ist Cov(Z, Y) / Cov(Z, X) für jedes gültige Instrument gleich der Wirkung."""
    rng = SplitMix64(C.SEED_BASE + 200)
    done = iv = fd = bd = both = checked = wrong = 0
    per_diagram = []
    while done < diagrams:
        p = C.RANDOM_P_LO + C.RANDOM_P_WIDTH * float(rng.uniform(1)[0])
        base = [list(ps) for ps in random_dag(nodes, p, rng)]
        draw = rng.uniform(2)
        x = 1 + int(draw[0] * (nodes - 1))
        y = 1 + int(draw[1] * (nodes - 1))
        if x == y:
            continue
        x, y = sorted((x, y))
        for v in (x, y):
            if 0 not in base[v]:
                base[v].append(0)
        parents = tuple(tuple(sorted(ps)) for ps in base)
        if not B.directed_paths(parents, x, y):
            continue
        done += 1
        observed = [v for v in range(1, nodes) if v not in (x, y)]
        ins = K.instruments(parents, x, y, observed)
        has_fd = bool(K.frontdoor_sets(parents, x, y, observed))
        bd += any(B.adjustment_valid(parents, x, y, z) for z in B.all_sets(observed))
        iv += bool(ins)
        fd += has_fd
        both += bool(ins) and has_fd
        per_diagram.append(len(ins))
        if ins:
            bm = np.zeros((nodes, nodes))
            for j, ps in enumerate(parents):
                if ps:
                    dr = rng.uniform(2 * len(ps))
                    for k, q in enumerate(ps):
                        bm[j, q] = (C.COEF_LO + (C.COEF_HI - C.COEF_LO) * dr[k]) * (1.0 if dr[len(ps) + k] < 0.5 else -1.0)
            inv = np.linalg.inv(np.eye(nodes) - bm)
            cov, total = inv @ inv.T, inv - np.eye(nodes)
            for z in ins:
                checked += 1
                wrong += abs(cov[z, y] / cov[z, x] - total[y, x]) > 1e-8
    return {"diagrams": diagrams, "nodes": nodes, "instrument": iv, "frontdoor": fd, "backdoor": bd, "both": both, "either": iv + fd - both, "per_diagram": per_diagram,
            "checked": checked, "wrong": int(wrong)}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    by = {r.name: r for r in a.rows}
    iv, naive, ctrl = by["2SLS (Instrument)"], by["Naiver Vergleich"], by["Kontrolle für das Instrument"]
    text = (f"Der 2SLS schätzt {de(iv.estimate)} (Wahrheit {de(a.truth)}, Fehler {signed(iv.error)}), das Intervall {'deckt' if iv.covers else 'verfehlt'} die Wahrheit; "
            f"die erste Stufe hat F = {de(a.first['f'], 1)}. Der naive Vergleich liegt um {signed(naive.error)} daneben, die Kontrolle für das Instrument um {signed(ctrl.error)}.")
    s = a.settings
    active = [name for name, v in (("direkten Pfeil vom Anstoß auf den Servicegrad", s.g), ("Einfluss der Standortqualität auf den Anstoß", s.h)) if v > 0]
    if active:
        return "warning", text + " Eine Annahme ist verletzt: " + ", ".join(active) + "."
    if a.first["f"] < C.F_RULE:
        return "warning", text + f" Das Instrument ist schwach (F unter {de(C.F_RULE, 0)}): Schätzung und Intervall sind unzuverlässig, die Anderson-Rubin-Menge ist {ar_text(a.ar)}."
    if abs(iv.limit - a.truth) < 1e-9 and iv.covers:
        return "success", text
    return "info", text


def verdict_base(rows: list) -> tuple[str, str]:
    by = {r["pi"]: r for r in rows}
    first, weak = rows[0], by[0.1]
    return "info", (f"Bei starkem Instrument (π = {de(first['pi'], 1)}, F im Mittel {de(first['f_mean'], 0)}) trifft der 2SLS (Median-Verzerrung {signed(first['iv']['median_bias'], 3)}), streut aber "
                    f"{de(first['iv']['sd'], 3)} gegen {de(first['oracle']['sd'], 3)} beim Orakel. Bei π = 0,1 ist F im Mittel {de(weak['f_mean'], 1)}, der Median liegt {signed(weak['iv']['median_bias'], 2)} "
                    f"daneben, der Streubereich der mittleren Hälfte (IQR) ist {de(weak['iv']['iqr'], 2)} und die Anderson-Rubin-Menge ist in {de(100 * (1 - weak['ar_bounded']), 0)} % unbeschränkt.")


def verdict_violations(rows: list) -> tuple[str, str]:
    by = {r["label"]: r for r in rows}
    strong, weak = by["Z wirkt direkt auf y (0,1), π = 1,0"], by["Z wirkt direkt auf y (0,1), π = 0,1"]
    return "info", (f"Ein direkter Pfeil von 0,1 verfehlt den 2SLS bei π = 1,0 um {signed(strong['iv_error'])}, bei π = 0,1 um {signed(weak['iv_error'])}: Der Fehler ist g / π. "
                    f"Die Stärke des Confounders ändert nichts ({signed(by['starker Confounder (U wirkt mit 2,0)']['iv_error'])}).")


def verdict_overid(rows: list) -> tuple[str, str]:
    by = {r["label"]: r for r in rows}
    ok, one, both = by["beide gültig"], by["eines ungültig (Z₂ → Y 0,3)"], by["beide gleich ungültig (je 0,3)"]
    return "info", (f"Sind beide Instrumente gültig, lehnt der J-Test in {de(100 * ok['reject'], 0)} % ab; ist eines ungültig, in {de(100 * one['reject'], 0)} % (Fehler {signed(one['iv']['median_bias'], 2)}). "
                    f"Sind beide gleich ungültig, lehnt er nur in {de(100 * both['reject'], 0)} % ab, obwohl der Fehler {signed(both['iv']['median_bias'], 2)} beträgt: Er ist blind.")


def verdict_late(res: dict) -> tuple[str, str]:
    p, last = res["population"], res["rows"][-1]
    return "info", (f"Der Wald-Schätzer trifft die Wirkung bei den Compliern ({de(p['late'], 2)}, Mittel {de(last['wald']['mean'], 3)} bei n = {last['n']}), nicht die durchschnittliche Wirkung "
                    f"({de(p['ate'], 2)}). Der naive Vergleich zeigt {de(p['naive'], 2)}.")


def verdict_diagrams(res: dict) -> tuple[str, str]:
    share = lambda k: de(100 * res[k] / res["diagrams"], 1)      # noqa: E731
    return "info", (f"In {res['diagrams']} Zufallsdiagrammen mit verborgener gemeinsamer Ursache gibt es nie eine gültige Backdoor-Menge ({res['backdoor']}), aber in {res['instrument']} ({share('instrument')} %) "
                    f"ein gültiges Instrument und in {res['frontdoor']} ({share('frontdoor')} %) eine Frontdoor-Menge, in nur {res['both']} beides. In allen {res['checked'] - res['wrong']} von "
                    f"{res['checked']} geprüften Paaren aus Instrument und linearer Population stimmt der Quotient mit der Wirkung.")
