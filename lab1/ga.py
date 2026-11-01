"""ЛР1, вариант 16: вещественный ГА для функции «изогнутая сигара» (bent cigar).
f(x) = x1^2 + 1e6 * sum_{i=2..d} x_i^2,  d = 9,  x_i in [-100, 100],  min f = 0 в x = 0.
Ключевые операторы (селекция, кроссовер, мутация, границы, элитизм) реализованы самостоятельно;
numpy используется только для массивов и генерации случайных чисел.
"""
from dataclasses import dataclass
import numpy as np

V, D = 16, 5 + (16 % 6)          # вариант и размерность: d = 9
LOW, HIGH = -100.0, 100.0


def bent_cigar(X: np.ndarray) -> np.ndarray:
    """Целевая функция для матрицы особей (n, d) -> вектор (n,)."""
    return X[:, 0] ** 2 + 1e6 * np.sum(X[:, 1:] ** 2, axis=1)


@dataclass
class Config:
    name: str = "base"
    pop_size: int = 100
    budget: int = 100_000          # лимит вычислений целевой функции
    p_cross: float = 0.9
    p_mut: float = 1.0 / D         # вероятность мутации гена
    sigma0: float = 0.01           # начальный масштаб мутации (доля от ширины области)
    adaptive: bool = False         # адаптация масштаба мутации (доп. задание)
    selection: str = "tournament"  # tournament | rank
    tour_k: int = 3
    alpha: float = 0.3             # параметр BLX-alpha
    elite: int = 2
    stop_f: float = 1e-12          # досрочная остановка по достижении значения


# ---------- селекция ----------
def select_tournament(fit, n, k, rng):
    idx = rng.integers(0, len(fit), size=(n, k))
    return idx[np.arange(n), np.argmin(fit[idx], axis=1)]


def select_rank(fit, n, rng, s=1.8):
    """Линейная ранговая селекция (минимизация: лучшая особь получает ранг N)."""
    N = len(fit)
    order = np.argsort(-fit)                    # от худшей к лучшей
    rank = np.empty(N); rank[order] = np.arange(N)   # 0 .. N-1, лучшая = N-1
    p = (2 - s) / N + 2 * rank * (s - 1) / (N * (N - 1))
    return rng.choice(N, size=n, p=p / p.sum())


# ---------- рекомбинация и мутация ----------
def blx_alpha(p1, p2, alpha, p_cross, rng):
    """BLX-alpha: потомок равномерно из [min-α·d, max+α·d] покоординатно."""
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    d = hi - lo
    child = rng.uniform(lo - alpha * d, hi + alpha * d)
    do = rng.random(len(p1)) < p_cross
    return np.where(do[:, None], child, p1)     # без кроссовера — копия родителя


def gauss_mutation(X, p_mut, sigma_abs, rng):
    mask = rng.random(X.shape) < p_mut
    return X + mask * rng.normal(0.0, sigma_abs, X.shape)


def reflect(X):
    """Обработка границ: отражение от границы (с гарантией попадания в [LOW, HIGH])."""
    w = HIGH - LOW
    Y = np.mod(X - LOW, 2 * w)
    Y = np.where(Y > w, 2 * w - Y, Y)
    return Y + LOW


# ---------- основной алгоритм ----------
def run_ga(cfg: Config, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(LOW, HIGH, (cfg.pop_size, D))
    fit = bent_cigar(X); evals = cfg.pop_size
    sigma = cfg.sigma0
    hist = [(evals, fit.min(), fit.mean(), fit.max())]
    prev_best = fit.min()
    n_child = cfg.pop_size - cfg.elite
    while evals + n_child <= cfg.budget and fit.min() > cfg.stop_f:
        if cfg.selection == "tournament":
            i1 = select_tournament(fit, n_child, cfg.tour_k, rng)
            i2 = select_tournament(fit, n_child, cfg.tour_k, rng)
        else:
            i1, i2 = select_rank(fit, n_child, rng), select_rank(fit, n_child, rng)
        ch = blx_alpha(X[i1], X[i2], cfg.alpha, cfg.p_cross, rng)
        ch = gauss_mutation(ch, cfg.p_mut, sigma * (HIGH - LOW), rng)
        ch = reflect(ch)
        cf = bent_cigar(ch); evals += n_child
        el = np.argsort(fit)[:cfg.elite]                   # элитизм
        X = np.vstack([X[el], ch]); fit = np.concatenate([fit[el], cf])
        if cfg.adaptive:   # стагнация лучшего -> уменьшаем масштаб, улучшение -> слегка увеличиваем
            sigma = min(sigma * 1.05, 0.2) if fit.min() < 0.999 * prev_best else max(sigma * 0.9, 1e-12)
        prev_best = fit.min()
        hist.append((evals, fit.min(), fit.mean(), fit.max()))
    b = int(np.argmin(fit))
    return dict(best_f=float(fit[b]), best_x=X[b], evals=evals, hist=np.array(hist))


def run_random(budget, seed, batch=1000):
    """Базовая случайная выборка в области D с тем же бюджетом."""
    rng = np.random.default_rng(seed)
    best, evals, hist = np.inf, 0, []
    while evals < budget:
        n = min(batch, budget - evals)
        f = bent_cigar(rng.uniform(LOW, HIGH, (n, D))); evals += n
        best = min(best, f.min()); hist.append((evals, best))
    return dict(best_f=float(best), evals=evals, hist=np.array(hist))
