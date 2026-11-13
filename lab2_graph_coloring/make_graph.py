"""Генерация входных данных: случайный граф G(n,p), seed фиксирован."""
import json, sys
from gc_ga.graph import *

N, P, SEED = 80, 0.30, 2026
edges = generate_gnp(N, P, SEED)
save_graph("data/graph.json", N, P, SEED, edges)
with open("data/graph_edges.csv", "w", newline="\n") as f:     # LF на любой ОС: файлы совпадают побайтно
    f.write("u,v\n")
    for u, v in edges:
        f.write(f"{u},{v}\n")
g, _ = load_graph("data/graph.json")
ds = dsatur(g)
cl = max_clique(g)
lb = len(cl)
ub = int(ds.max()) + 1
chi, proven, nodes = exact_chromatic(g, ub, lb, node_limit=3_000_000)
info = dict(n=g.n, m=g.m, density=2*g.m/(g.n*(g.n-1)), max_deg=g.max_deg, min_deg=int(g.deg.min()),
            mean_deg=float(g.deg.mean()), dsatur_colors=ub, clique_lb=lb, clique=[int(x) for x in cl],
            chi_found=chi, chi_proven=proven, bb_nodes=nodes)
json.dump(info, open("data/graph_info.json", "w", newline="\n"), indent=2)
print(info)
