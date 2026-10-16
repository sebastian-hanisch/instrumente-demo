"""Vehikel: ein Standort wählt die neue Regel B (X) abhängig von der verborgenen Standortqualität U; ein Anstoß von außen (Z) verschiebt die Regelwahl, wirkt aber nicht selbst
auf den Servicegrad (Y). Lineares Modell mit Einheitsrauschen; die Wahrheit ist die Wirkung β von X auf Y. Nur numpy und der Generator der Linie, kein Streamlit.

Knoten (topologisch): 0 U (verborgen), 1 Z (Instrument), 2 Z2 (zweites Instrument, nur Experiment 3), 3 X, 4 Y.
Kanten: U -> X und U -> Y (je a), Z -> X (π), X -> Y (β); Verletzungen: Z -> Y (g), U -> Z (h); zweites Instrument: Z2 -> X (π₂), Z2 -> Y (g₂), U -> Z2 (h₂).
Dazu das binäre Beispiel: Ermutigung Z ~ B(0,5), Typen Complier, Always-Taker, Never-Taker (keine Defier)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import ins_constants as C
from kausal_generator import SplitMix64

U, Z, Z2, X, Y = 0, 1, 2, 3, 4
NAMES = ("Standortqualität (verborgen)", "Anstoß Z", "zweiter Anstoß Z₂", "Regel B", "Servicegrad")


@dataclass(frozen=True)
class World:
    a: float = C.DEFAULT_A
    pi: float = C.DEFAULT_PI
    beta: float = C.DEFAULT_BETA
    g: float = 0.0
    h: float = 0.0
    pi2: float = 0.0
    g2: float = 0.0
    h2: float = 0.0


def coefficient_matrix(w: World) -> np.ndarray:
    """B[j, i] = Koeffizient der Kante i -> j."""
    b = np.zeros((5, 5))
    b[X, U], b[Y, U], b[X, Z], b[Y, X], b[Y, Z], b[Z, U] = w.a, w.a, w.pi, w.beta, w.g, w.h
    b[X, Z2], b[Y, Z2], b[Z2, U] = w.pi2, w.g2, w.h2
    return b


def parents(w: World) -> tuple:
    b = coefficient_matrix(w)
    return tuple(tuple(i for i in range(5) if b[j, i] != 0.0) for j in range(5))


def population(w: World):
    """Kovarianz Σ der fünf Knoten und Gesamtwirkungen T = (I − B)⁻¹ − I bei Einheitsrauschen."""
    inv = np.linalg.inv(np.eye(5) - coefficient_matrix(w))
    return inv @ inv.T, inv - np.eye(5)


def truth(w: World) -> float:
    """Wirkung von X auf Y (X hat nur den einen Pfeil nach Y): β."""
    return float(population(w)[1][Y, X])


def simulate(w: World, n: int, rng: SplitMix64, reps: int = 1) -> np.ndarray:
    """Alle Knoten, Form (reps, n, 5)."""
    z = rng.normal(reps * n * 5).reshape(reps, n, 5)
    return z @ np.linalg.inv(np.eye(5) - coefficient_matrix(w)).T


# ----------------------------------------------------------------------------- binäre Ermutigung (Complier, LATE)

def late_population() -> dict:
    """Exakte Größen des binären Beispiels: Anteile, Wirkungen je Typ, ATE, LATE (Complier), die Erwartungen je Z, die Wald-Größe und der naive Vergleich."""
    shares = dict(zip(C.LATE_TYPES, C.LATE_SHARES))
    effect = {t: C.LATE_PY[t][1] - C.LATE_PY[t][0] for t in shares}
    ate = sum(shares[t] * effect[t] for t in shares)

    def x_of(t, z):
        return z if t == "Complier" else (1 if t == "Always-Taker" else 0)

    ey = {z: sum(shares[t] * C.LATE_PY[t][x_of(t, z)] for t in shares) for z in (0, 1)}
    ex = {z: sum(shares[t] * x_of(t, z) for t in shares) for z in (0, 1)}
    px1 = sum(shares[t] * (C.LATE_PZ * x_of(t, 1) + (1 - C.LATE_PZ) * x_of(t, 0)) for t in shares)
    ey_x1 = sum(shares[t] * (C.LATE_PZ * x_of(t, 1) * C.LATE_PY[t][1] + (1 - C.LATE_PZ) * x_of(t, 0) * C.LATE_PY[t][1]) for t in shares) / px1
    ey_x0 = sum(shares[t] * (C.LATE_PZ * (1 - x_of(t, 1)) * C.LATE_PY[t][0] + (1 - C.LATE_PZ) * (1 - x_of(t, 0)) * C.LATE_PY[t][0]) for t in shares) / (1 - px1)
    return {"shares": shares, "effect": effect, "ate": ate, "late": effect["Complier"], "ey": ey, "ex": ex, "wald": (ey[1] - ey[0]) / (ex[1] - ex[0]), "naive": ey_x1 - ey_x0}


def simulate_late(n: int, rng: SplitMix64, reps: int = 1):
    """(typ, z, x, y) mit Form (reps, n); Typ 0 = Complier, 1 = Always-Taker, 2 = Never-Taker."""
    size = reps * n
    cut = np.cumsum(C.LATE_SHARES)
    typ = np.searchsorted(cut, rng.uniform(size).reshape(reps, n), side="right")
    z = (rng.uniform(size).reshape(reps, n) < C.LATE_PZ).astype(int)
    x = np.where(typ == 0, z, np.where(typ == 1, 1, 0))
    table = np.array([C.LATE_PY[t] for t in C.LATE_TYPES])
    y = (rng.uniform(size).reshape(reps, n) < table[typ, x]).astype(int)
    return typ, z, x, y
