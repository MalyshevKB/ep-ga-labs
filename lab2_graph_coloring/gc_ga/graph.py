"""Граф: генерация, DSATUR, нижняя оценка (клика), точный хроматический номер."""
import json
import numpy as np


class Graph:
    def __init__(self, n, edges):
        self.n = n
        self.edges = np.array(sorted(map(tuple, edges)), dtype=np.int32)
        self.eu = self.edges[:, 0].copy()
        self.ev = self.edges[:, 1].copy()
        self.adj = [[] for _ in range(n)]
        for u, v in self.edges:
            self.adj[u].append(int(v))
            self.adj[v].append(int(u))
        self.adj_arr = [np.array(a, dtype=np.int32) for a in self.adj]
        self.deg = np.array([len(a) for a in self.adj])
        self.max_deg = int(self.deg.max())

    @property
    def m(self):
        return len(self.edges)


def generate_gnp(n, p, seed):
    """Случайный граф Эрдёша—Реньи G(n, p) с зафиксированным seed."""
    rng = np.random.default_rng(seed)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]


def save_graph(path, n, p, seed, edges):
    with open(path, "w") as f:
        json.dump({"n": n, "p": p, "seed": seed, "edges": [list(map(int, e)) for e in edges]}, f)


def load_graph(path):
    with open(path) as f:
        d = json.load(f)
    return Graph(d["n"], d["edges"]), d


# ---------- конструктивная эвристика DSATUR ----------
def dsatur(g):
    n = g.n
    color = -np.ones(n, dtype=np.int32)
    sat = [set() for _ in range(n)]
    for _ in range(n):
        best, bkey = -1, None
        for v in range(n):
            if color[v] >= 0:
                continue
            key = (len(sat[v]), g.deg[v])
            if bkey is None or key > bkey:
                best, bkey = v, key
        used = sat[best]
        c = 0
        while c in used:
            c += 1
        color[best] = c
        for u in g.adj[best]:
            sat[u].add(c)
    return color


# ---------- нижняя оценка: максимальная клика (Bron–Kerbosch с pivot) ----------
def max_clique(g):
    adj = [set(a) for a in g.adj]
    best = []

    def bk(R, P, X):
        nonlocal best
        if not P and not X:
            if len(R) > len(best):
                best = R[:]
            return
        if len(R) + len(P) <= len(best):
            return
        u = max(P | X, key=lambda w: len(adj[w] & P))
        for v in list(P - adj[u]):
            bk(R + [v], P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    bk([], set(range(g.n)), set())
    return best


# ---------- точный хроматический номер: DSATUR branch & bound ----------
def exact_chromatic(g, ub_colors, lb, node_limit=5_000_000):
    """Возвращает (chi, доказан_ли, число узлов). Ветвление по вершине максимальной насыщенности."""
    n = g.n
    best = [ub_colors]
    nodes = [0]
    color = [-1] * n
    cnt = [[0] * (n + 1) for _ in range(n)]
    sat = [0] * n
    exhausted = [False]

    def rec(colored, used):
        nodes[0] += 1
        if nodes[0] > node_limit:
            exhausted[0] = True
            return
        if used >= best[0]:
            return
        if colored == n:
            best[0] = used
            return
        v, key = -1, None
        for u in range(n):
            if color[u] < 0:
                k = (sat[u], g.deg[u])
                if key is None or k > key:
                    v, key = u, k
        for c in range(min(used + 1, best[0] - 1)):
            if cnt[v][c] == 0:
                color[v] = c
                for w in g.adj[v]:
                    if cnt[w][c] == 0:
                        sat[w] += 1
                    cnt[w][c] += 1
                rec(colored + 1, max(used, c + 1))
                for w in g.adj[v]:
                    cnt[w][c] -= 1
                    if cnt[w][c] == 0:
                        sat[w] -= 1
                color[v] = -1
                if best[0] <= lb or exhausted[0]:
                    return

    rec(0, 0)
    proven = (not exhausted[0])
    return best[0], proven, nodes[0]
