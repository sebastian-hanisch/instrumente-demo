"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def unit_world():
    """Mini-Instanz 1 (a = π = β = 1, Einheitsrauschen): U -> X, U -> Y, Z -> X, X -> Y, kein weiterer Pfeil.
    Handrechnung: Var Z = 1; Var X = π² + a² + 1 = 3; Cov(Z, X) = π = 1; Cov(U, X) = a = 1; Cov(X, Y) = β Var X + a·a = 3 + 1 = 4; Cov(Z, Y) = β Cov(Z, X) = 1; Cov(U, Y) = β a + a = 2.
    Naiv 4/3. Kontrolle für Z: [[3, 1], [1, 1]] β = (4, 1): det 2, β_X = (4 − 1)/2 = 1,5. IV 1/1 = 1. Orakel mit U: [[3, 1], [1, 1]] β = (4, 2): β_X = (4 − 2)/2 = 1."""
    from ins_world import World

    return World(a=1.0, pi=1.0, beta=1.0)


@pytest.fixture
def classic():
    """Diagramm 0 = U (verborgen), 1 = Z, 2 = X, 3 = Y: U -> X, U -> Y, Z -> X, X -> Y. Gültiges Instrument: Z (Knoten 1)."""
    return ((), (), (0, 1), (0, 2))


@pytest.fixture
def four_units():
    """Mini-Instanz 2 (vier Einheiten): z = (0, 0, 1, 1), x = (0, 1, 1, 1), y = (0, 1, 1, 2).
    zc = (−½, −½, ½, ½), xc = x − ¾ = (−¾, ¼, ¼, ¼), yc = y − 1 = (−1, 0, 0, 1). Erste Stufe: π̂ = zc·xc / zc·zc = ½ / 1 = ½; x̂ = (−¼, −¼, ¼, ¼); β̂ = x̂·yc / x̂·x̂ = ½ / ¼ = 2.
    Reste yc − 2 xc = (½, −½, −½, ½), Summe der Quadrate 1; Standardfehler √(1 / (4 − 2) / ¼) = √2. Erste-Stufe-F: rss = (−½, ½, 0, 0)·(…) = ½; F = (¼ / 1) / (½ / 2) = 1."""
    z = np.array([0.0, 0.0, 1.0, 1.0])
    x = np.array([0.0, 1.0, 1.0, 1.0])
    y = np.array([0.0, 1.0, 1.0, 2.0])
    return z, x, y


@pytest.fixture
def binary_sample():
    """Mini-Instanz 3 (binär, acht Einheiten): z = (0,0,0,0,1,1,1,1), x = (0,0,0,1,0,1,1,1), y = (0,0,1,1,1,1,1,0).
    E[X | Z=1] = ¾, E[X | Z=0] = ¼; E[Y | Z=1] = ¾, E[Y | Z=0] = ½: Wald (¾ − ½)/(¾ − ¼) = ½. Naiv: E[Y | X=1] = ¾ (y der Einheiten 3, 5, 6, 7 = 1, 1, 1, 0), E[Y | X=0] = ½: ¼."""
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    x = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    y = np.array([0, 0, 1, 1, 1, 1, 1, 0])
    return z, x, y
