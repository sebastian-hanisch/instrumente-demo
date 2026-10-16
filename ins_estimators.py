"""Schätzer: naiver Vergleich, Kontrolle für das Instrument, 2SLS (Wald bei einem Instrument, mit erster Stufe und F), Anderson-Rubin-Menge, Sargan-J-Test, Orakel mit dem verborgenen
Confounder; dazu die Population (exakte Grenzwerte aus der Kovarianz) und der Wald-Schätzer für binäre Daten. Nur numpy, kein Streamlit."""

from __future__ import annotations

import numpy as np

import ins_constants as C
import ins_world as W


def ols(y: np.ndarray, columns) -> tuple[np.ndarray, np.ndarray]:
    """Kleinste Quadrate mit Achsenabschnitt: (Koeffizienten, Kovarianzmatrix der Koeffizienten); `columns` sind die Regressoren ohne Achsenabschnitt."""
    d = np.column_stack([np.ones(len(y))] + list(columns))
    beta = np.linalg.solve(d.T @ d, d.T @ y)
    res = y - d @ beta
    return beta, np.linalg.inv(d.T @ d) * (res @ res) / (len(y) - d.shape[1])


def naive(x, y) -> tuple[float, float]:
    beta, cov = ols(y, [x])
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def control_instrument(x, z, y) -> tuple[float, float]:
    """Koeffizient von X in Y ~ X + Z (die schlechte Kontrolle aus Stück 4: Bias-Verstärkung)."""
    beta, cov = ols(y, [x, z])
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def oracle(x, u, y) -> tuple[float, float]:
    """Koeffizient von X in Y ~ X + U: das Backdoor-Orakel, das den verborgenen Confounder kennt (in der Praxis nicht möglich)."""
    beta, cov = ols(y, [x, u])
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def tsls(x, zs, y) -> dict:
    """2SLS mit Achsenabschnitt; `zs` sind die Instrumente. Bei einem Instrument ist die Schätzung der Wald-Quotient Cov(Z, Y) / Cov(Z, X).
    Standardfehler mit den Resten y − β̂ x (nicht mit den angepassten x); erste Stufe: F-Statistik aller Instrumente gemeinsam. `resid` und `zc` für den J-Test."""
    n = len(y)
    zc = np.column_stack([z - z.mean() for z in zs])
    xc, yc = x - x.mean(), y - y.mean()
    coef = np.linalg.lstsq(zc, xc, rcond=None)[0]
    xh = zc @ coef
    est = float((xh @ yc) / (xh @ xh))
    resid = yc - est * xc
    se = float(np.sqrt(resid @ resid / (n - 2) / (xh @ xh)))
    k = zc.shape[1]
    rss1 = float((xc - xh) @ (xc - xh))
    f = float(((xh @ xh) / k) / (rss1 / (n - k - 1)))
    return {"estimate": est, "se": se, "f": f, "pi": coef, "resid": resid, "zc": zc}


def sargan(resid: np.ndarray, zc: np.ndarray) -> float:
    """J = n R² der Regression der 2SLS-Reste auf die Instrumente (χ² mit (Instrumente − 1) Freiheitsgraden unter der Nullhypothese gültiger Instrumente)."""
    coef = np.linalg.lstsq(zc, resid, rcond=None)[0]
    r2 = 1.0 - float(((resid - zc @ coef) ** 2).sum() / (resid @ resid))
    return len(resid) * r2


def anderson_rubin(x, z, y, crit: float = C.CHI2_CRIT) -> dict:
    """Anderson-Rubin-Menge bei einem Instrument: alle β₀, bei denen die Regression von y − β₀ x auf z keine Wirkung von z zeigt (t² ≤ crit). Das ist eine quadratische
    Ungleichung: `kind` ist "interval" (beschränkt), "outside" (alles außer einem Intervall, unbeschränkt) oder "all" (die ganze Gerade). Leer kann die Menge nicht sein:
    die Schätzung selbst löst die Ungleichung."""
    n = len(y)
    zc, xc, yc = z - z.mean(), x - x.mean(), y - y.mean()
    zz, zx, zy, xx, xy, yy = zc @ zc, zc @ xc, zc @ yc, xc @ xc, xc @ yc, yc @ yc
    k = crit / (n - 2) * zz
    s = 1.0 + k / zz
    q2 = s * zx * zx - k * xx
    q1 = -2.0 * s * zy * zx + 2.0 * k * xy
    q0 = s * zy * zy - k * yy
    disc = q1 * q1 - 4.0 * q2 * q0
    if q2 == 0.0 or disc < 0.0:                                                   # keine Nullstelle (oder keine Krümmung): jedes β₀ wird nicht abgelehnt
        return {"kind": "all", "lo": float("-inf"), "hi": float("inf")}
    r = float(np.sqrt(disc))
    lo, hi = sorted(((-q1 - r) / (2 * q2), (-q1 + r) / (2 * q2)))
    return {"kind": "interval" if q2 > 0 else "outside", "lo": float(lo), "hi": float(hi)}


def ar_contains(ar: dict, value: float) -> bool:
    if ar["kind"] == "all":
        return True
    inside = ar["lo"] <= value <= ar["hi"]
    return inside if ar["kind"] == "interval" else not (ar["lo"] < value < ar["hi"])


def ar_bounded(ar: dict) -> bool:
    return ar["kind"] == "interval"


# ----------------------------------------------------------------------------- Population

def coefficients(cov: np.ndarray, y: int, xs: list[int]) -> np.ndarray:
    return np.linalg.solve(cov[np.ix_(xs, xs)], cov[xs, y])


def population_estimates(w: W.World) -> dict:
    """Grenzwerte (n → ∞) der Schätzer und die Wahrheit, exakt aus der Kovarianz."""
    cov = W.population(w)[0]
    return {"truth": W.truth(w), "naive": float(coefficients(cov, W.Y, [W.X])[0]), "control": float(coefficients(cov, W.Y, [W.X, W.Z])[0]),
            "iv": float(cov[W.Z, W.Y] / cov[W.Z, W.X]), "oracle": float(coefficients(cov, W.Y, [W.X, W.U])[0]), "pi": float(cov[W.Z, W.X] / cov[W.Z, W.Z])}


def expected_f(w: W.World, n: int) -> float:
    """Erwartete erste-Stufe-F-Statistik (Näherung n R² / (1 − R²), R² = Korrelation(Z, X)²)."""
    cov = W.population(w)[0]
    r2 = cov[W.Z, W.X] ** 2 / (cov[W.Z, W.Z] * cov[W.X, W.X])
    return float(n * r2 / (1.0 - r2))


# ----------------------------------------------------------------------------- binär

def wald_binary(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> float | None:
    """Wald-Schätzer für binäre Daten: (E[Y | Z = 1] − E[Y | Z = 0]) / (E[X | Z = 1] − E[X | Z = 0]); None, wenn Z x nicht bewegt (Nenner 0) oder eine Gruppe leer ist."""
    if not (z == 1).any() or not (z == 0).any():
        return None
    dx = float(x[z == 1].mean() - x[z == 0].mean())
    if dx == 0.0:
        return None
    return float(y[z == 1].mean() - y[z == 0].mean()) / dx


def naive_binary(x: np.ndarray, y: np.ndarray) -> float | None:
    if not (x == 1).any() or not (x == 0).any():
        return None
    return float(y[x == 1].mean() - y[x == 0].mean())
