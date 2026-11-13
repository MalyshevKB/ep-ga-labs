"""Генетический алгоритм для раскраски графа (вектор целых чисел, вариант 16, ЛР №2).

Особь (генотип = фенотип): c[i] ∈ {0..K-1} — цвет вершины i.
Целевая функция (минимизируется):
    F(c) = k(c) + lam * conf(c) - eps * Q(c),
    k    — число различных использованных цветов,
    conf — число рёбер, концы которых покрашены одинаково (нарушения ограничений),
    Q    — сглаживающая добавка sum(s_j^2)/n^2 (опционально; при eps<1 порядок по k не меняется).
Допустимое решение: conf = 0.
"""
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class Config:
    name: str = "A2_penalty_lam2"
    pop_size: int = 100
    max_evals: int = 50_000          # критерий остановки: лимит вычислений фитнес-функции
    elite: int = 2
    tournament: int = 3
    p_cross: float = 0.9
    crossover: str = "uniform"       # uniform | onepoint | gpx
    mutation: str = "random"         # random | mincon | elim
    p_mut: float = 0.02              # вероятность мутации гена (random) / базовая доля вершин (mincon)
    constraint: str = "penalty"      # penalty | repair
    lam: float = 2.0                 # коэффициент штрафа за конфликт
    eps: float = 0.0                 # вес сглаживающего члена


class Problem:
    def __init__(self, g, K, lb):
        self.g, self.K, self.lb = g, K, lb
        self.n = g.n


# ----------------------------------------------------------- оценка
def metrics(pr, c):
    sizes = np.bincount(c, minlength=pr.K)
    k = int((sizes > 0).sum())
    conf = int((c[pr.g.eu] == c[pr.g.ev]).sum())
    q = float((sizes.astype(np.float64) ** 2).sum()) / (pr.n ** 2)
    return k, conf, q


def fitness(cfg, k, conf, q):
    return k + cfg.lam * conf - cfg.eps * q


# ----------------------------------------------------------- ремонт
def repair(pr, c, rng):
    """Восстановление допустимости: конфликтующие вершины (в случайном порядке) получают
    наименьший цвет, не занятый ни одним соседом. Такой цвет существует всегда (K >= Δ+1).
    Перекрашенная вершина не конфликтует с соседями, а последующие перекраски выбирают цвета,
    свободные у всех соседей, поэтому на выходе conf = 0."""
    g = pr.g
    mask = c[g.eu] == c[g.ev]
    if not mask.any():
        return c
    bad = np.zeros(pr.n, dtype=bool)
    bad[g.eu[mask]] = True
    bad[g.ev[mask]] = True
    idx = np.flatnonzero(bad)
    rng.shuffle(idx)
    for v in idx:
        occ = np.zeros(pr.K, dtype=bool)
        occ[c[g.adj_arr[v]]] = True
        if not occ[c[v]]:
            continue                      # конфликт уже снят перекраской соседа
        c[v] = int(np.argmin(occ))        # первый свободный цвет
    return c


# ----------------------------------------------------------- инициализация
def init_population(pr, cfg, rng):
    """Каждая особь — случайная раскраска k цветами, k ~ U[lb, K] (разнообразие по k)."""
    P = np.empty((cfg.pop_size, pr.n), dtype=np.int16)
    for i in range(cfg.pop_size):
        k = int(rng.integers(pr.lb, pr.K + 1))
        P[i] = rng.integers(0, k, pr.n)
    return P


# ----------------------------------------------------------- селекция
def tournament(F, cfg, rng):
    cand = rng.integers(0, len(F), cfg.tournament)
    return cand[np.argmin(F[cand])]


# ----------------------------------------------------------- кроссовер
def cx_uniform(a, b, rng):
    return np.where(rng.random(a.size) < 0.5, a, b)


def cx_onepoint(a, b, rng):
    pt = int(rng.integers(1, a.size))
    return np.concatenate([a[:pt], b[pt:]])


def cx_gpx(pr, a, b, rng):
    """Greedy Partition Crossover (Galinier & Hao): потомок наследует цветовые КЛАССЫ,
    а не метки цветов (метки симметричны: перестановка меток не меняет решение).
    Поочерёдно берётся наибольший класс одного из родителей, его вершины удаляются из
    классов обоих родителей. Оставшиеся вершины красятся случайно среди уже созданных классов."""
    n, K = pr.n, pr.K
    ca = [set(np.flatnonzero(a == j).tolist()) for j in range(K)]
    cb = [set(np.flatnonzero(b == j).tolist()) for j in range(K)]
    child = -np.ones(n, dtype=np.int16)
    parents = [ca, cb] if rng.random() < 0.5 else [cb, ca]
    used = 0
    for step in range(K):
        cls = parents[step % 2]
        j = max(range(K), key=lambda t: len(cls[t]))
        S = cls[j]
        if not S:
            break
        for v in S:
            child[v] = used
        S = set(S)
        for cl in (ca, cb):
            for t in range(K):
                cl[t] -= S
        used += 1
    rest = np.flatnonzero(child < 0)
    if rest.size:
        child[rest] = rng.integers(0, max(used, 1), rest.size)
    return child


# ----------------------------------------------------------- мутация
def mut_random(pr, c, cfg, rng):
    m = rng.random(c.size) < cfg.p_mut
    if m.any():
        k = int(c.max()) + 1
        c[m] = rng.integers(0, max(k, 2), int(m.sum()))
    return c


def mut_mincon(pr, c, cfg, rng):
    """Мутация «минимум конфликтов»: вершины выбираются (конфликтующие — с вероятностью 0.5,
    остальные — с p_mut), каждая перекрашивается в один из уже использованных цветов,
    дающий минимум конфликтов с соседями (ничья разрешается случайно)."""
    g = pr.g
    mask = c[g.eu] == c[g.ev]
    prob = np.full(pr.n, cfg.p_mut)
    prob[g.eu[mask]] = 0.5
    prob[g.ev[mask]] = 0.5
    sel = np.flatnonzero(rng.random(pr.n) < prob)
    if sel.size == 0:
        return c
    k = int(c.max()) + 1
    for v in sel:
        cnt = np.bincount(c[g.adj_arr[v]], minlength=k)[:k]
        best = np.flatnonzero(cnt == cnt.min())
        c[v] = int(best[rng.integers(best.size)])
    return c


def mut_elim(pr, c, cfg, rng):
    """Мутация «устранение цвета» (специализирована под цель «минимум цветов»): выбирается один
    цветовой класс (меньшие классы — вероятнее), все его вершины перекрашиваются в оставшиеся
    использованные цвета с минимумом конфликтов. Число цветов уменьшается на 1, возможные
    конфликты снимаются ремонтом или штрафом. С вероятностью 0.5 вместо этого — точечная мутация."""
    g = pr.g
    if rng.random() < 0.5:
        return mut_mincon(pr, c, cfg, rng)
    sizes = np.bincount(c, minlength=pr.K)
    used = np.flatnonzero(sizes > 0)
    if used.size <= 2:
        return c
    w = 1.0 / sizes[used]
    j = int(rng.choice(used, p=w / w.sum()))
    others = used[used != j]
    for v in rng.permutation(np.flatnonzero(c == j)):
        cnt = np.bincount(c[g.adj_arr[v]], minlength=pr.K)[others]
        best = others[cnt == cnt.min()]
        c[v] = int(best[rng.integers(best.size)])
    return c


# ----------------------------------------------------------- основной цикл
MUT = {"random": mut_random, "mincon": mut_mincon, "elim": mut_elim}


def run_ga(pr, cfg, seed):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    P = init_population(pr, cfg, rng)
    n_ev = 0
    best_feas = {"k": None, "c": None, "ev": None}

    def evaluate(c):
        nonlocal n_ev
        k, conf, q = metrics(pr, c)
        n_ev += 1
        if conf == 0 and (best_feas["k"] is None or k < best_feas["k"]):
            best_feas.update(k=k, c=c.copy(), ev=n_ev)
        return k, conf, fitness(cfg, k, conf, q)

    pk = np.zeros(cfg.pop_size, dtype=np.int32)
    pc = np.zeros(cfg.pop_size, dtype=np.int32)
    F = np.zeros(cfg.pop_size)
    for i in range(cfg.pop_size):
        if cfg.constraint == "repair":
            P[i] = repair(pr, P[i], rng)
        pk[i], pc[i], F[i] = evaluate(P[i])

    hist = []   # (evals, min F, mean F, max F, лучшее допустимое k или NaN, доля допустимых)

    def log():
        hist.append((n_ev, F.min(), F.mean(), F.max(),
                     np.nan if best_feas["k"] is None else best_feas["k"], float((pc == 0).mean())))
    log()

    # запускаются только полные поколения: неполное последнее поколение исказило бы
    # статистику финальной популяции (доля допустимых, min/mean/max F)
    while n_ev + cfg.pop_size - cfg.elite <= cfg.max_evals:
        order = np.argsort(F, kind="stable")[: cfg.elite]
        newP = [P[j].copy() for j in order]
        newk, newc, newF = [pk[j] for j in order], [pc[j] for j in order], [F[j] for j in order]
        while len(newP) < cfg.pop_size:
            a, b = P[tournament(F, cfg, rng)], P[tournament(F, cfg, rng)]
            if rng.random() < cfg.p_cross:
                if cfg.crossover == "uniform":
                    ch = cx_uniform(a, b, rng)
                elif cfg.crossover == "onepoint":
                    ch = cx_onepoint(a, b, rng)
                else:
                    ch = cx_gpx(pr, a, b, rng)
            else:
                ch = a.copy()
            ch = ch.astype(np.int16)
            ch = MUT[cfg.mutation](pr, ch, cfg, rng)
            if cfg.constraint == "repair":
                ch = repair(pr, ch, rng)
            k, conf, f = evaluate(ch)
            newP.append(ch); newk.append(k); newc.append(conf); newF.append(f)
        P = np.array(newP, dtype=np.int16)
        pk, pc, F = np.array(newk), np.array(newc), np.array(newF)
        log()

    i = int(np.argmin(F))
    return {"seed": seed, "best_k": best_feas["k"], "best_c": best_feas["c"],
            "evals_to_best": best_feas["ev"], "final_best_conf": int(pc[i]),
            "final_best_k": int(pk[i]), "final_feasible_share": float((pc == 0).mean()),
            "evals": n_ev, "time": time.perf_counter() - t0, "hist": np.array(hist)}


# ----------------------------------------------------------- baseline: случайный допустимый поиск
def run_random_feasible(pr, max_evals, seed):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best_k, best_c, ev_best, hist = None, None, None, []
    for ev in range(1, max_evals + 1):
        k0 = int(rng.integers(pr.lb, pr.K + 1))
        c = repair(pr, rng.integers(0, k0, pr.n).astype(np.int16), rng)
        k, conf, _ = metrics(pr, c)
        if conf == 0 and (best_k is None or k < best_k):
            best_k, best_c, ev_best = k, c.copy(), ev
        if ev % 100 == 0:
            hist.append((ev, np.nan, np.nan, np.nan, best_k, 1.0))
    return {"seed": seed, "best_k": best_k, "best_c": best_c, "evals_to_best": ev_best,
            "evals": max_evals, "time": time.perf_counter() - t0, "hist": np.array(hist),
            "final_feasible_share": 1.0}
