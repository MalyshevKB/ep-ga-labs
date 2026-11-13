"""Статистика, графики и примеры решений по results/. Запуск: python analyze.py"""
import csv, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import mannwhitneyu
from gc_ga.graph import load_graph, dsatur
from gc_ga.ga import Problem, metrics

info = json.load(open("data/graph_info.json"))
g, _ = load_graph("data/graph.json")
rows = list(csv.DictReader(open("results/runs.csv")))
by = defaultdict(list)
for r in rows:
    by[r["config"]].append(r)
order = [c["name"] for c in json.load(open("config.json"))["configs"]] + ["RandomFeasible"]
order = [c for c in order if c in by]
CHI = info["chi_found"]

def f(x):
    return None if x in ("", "None") else float(x)

stats = {}
for name in order:
    R = by[name]
    k = np.array([f(r["best_k"]) if f(r["best_k"]) is not None else np.nan for r in R])
    feas = ~np.isnan(k)
    kk = np.where(feas, k, np.nan)
    ev = np.array([f(r["evals_to_best"]) or np.nan for r in R])
    stats[name] = dict(
        runs=len(R), best=np.nanmin(kk), mean=np.nanmean(kk), median=np.nanmedian(kk),
        std=np.nanstd(kk, ddof=1) if len(R) > 1 else 0.0, worst=np.nanmax(kk),
        hit_opt=float(np.mean(kk == CHI)), feasible_runs=float(feas.mean()),
        final_pop_feasible=float(np.mean([f(r["final_feasible_share"]) for r in R])),
        final_best_conf=float(np.mean([f(r["final_best_conf"]) for r in R])),
        evals_to_best=float(np.nanmean(ev)), time=float(np.mean([f(r["time_s"]) for r in R])),
        ks=kk)

# DSATUR baseline
ds = dsatur(g); ds_k = int(ds.max()) + 1

with open("results/summary.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["config", "runs", "best", "mean", "median", "std", "worst", "share_reach_chi",
                "share_runs_with_feasible", "mean_final_pop_feasible_share", "mean_evals_to_best", "mean_time_s"])
    for n in order:
        s = stats[n]
        w.writerow([n, s["runs"], int(s["best"]), round(s["mean"], 2), s["median"], round(s["std"], 2),
                    int(s["worst"]), s["hit_opt"], s["feasible_runs"], round(s["final_pop_feasible"], 3),
                    round(s["evals_to_best"]), round(s["time"], 1)])
    w.writerow(["DSATUR", 1, ds_k, ds_k, ds_k, 0, ds_k, float(ds_k == CHI), 1, 1, 0, 0])

# Манн–Уитни: сравнения «меняется один фактор»
pairs = [("P_lam0.5", "P_lam2"), ("P_lam2", "P_lam10"), ("P_lam2", "R_repair"), ("P_lam2", "P_lam2_eps"),
         ("R_repair", "R_onepoint"), ("R_repair", "R_gpx"), ("R_repair", "R_mincon"), ("R_repair", "R_elim"),
         ("R_gpx", "R_gpx_elim"), ("R_elim", "R_gpx_elim"), ("R_gpx_elim", "P_eps_gpx_elim"),
         ("R_repair", "RandomFeasible"), ("R_gpx_elim", "RandomFeasible")]
tests = []
for a, b in pairs:
    if a in stats and b in stats:
        x, y = stats[a]["ks"], stats[b]["ks"]
        x, y = x[~np.isnan(x)], y[~np.isnan(y)]
        if len(x) and len(y):
            try:
                p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            except ValueError:
                p = float("nan")
            tests.append((a, b, float(np.mean(x)), float(np.mean(y)), p))
with open("results/mannwhitney.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["A", "B", "mean_A", "mean_B", "p_value"])
    for t in tests:
        w.writerow([t[0], t[1], round(t[2], 2), round(t[3], 2), f"{t[4]:.3g}"])

# ---------- графики ----------
def hist_of(name):
    return np.load(f"results/hist_{name}.npy")

def conv_plot(names, fname, title):
    plt.figure(figsize=(8, 4.8))
    top = ds_k
    for n in names:
        if n not in stats: continue
        H = hist_of(n)
        x = H[0, :, 0]
        y = H[:, :, 4].copy()
        # в начале штрафных прогонов допустимых решений ещё нет: столбец целиком NaN, точка не рисуется
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(y, axis=0); lo = np.nanmin(y, axis=0); hi = np.nanmax(y, axis=0)
        plt.plot(x, mean, label=n); plt.fill_between(x, lo, hi, alpha=0.15)
        top = max(top, np.nanmax(hi))
    plt.axhline(CHI, color="k", ls="--", lw=1, label=f"χ(G)={CHI}")
    plt.axhline(ds_k, color="gray", ls=":", lw=1, label=f"DSATUR={ds_k}")
    plt.xlabel("число вычислений фитнес-функции"); plt.ylabel("лучшее допустимое число цветов")
    plt.title(title); plt.ylim(CHI - 0.5, top + 0.5); plt.legend(fontsize=8); plt.grid(alpha=.3)   # шкала по данным
    plt.tight_layout(); plt.savefig(f"figures/{fname}", dpi=140); plt.close()

conv_plot(["P_lam0.5", "P_lam2", "P_lam10", "R_repair"], "conv_constraints.png",
          "Обработка ограничений: среднее и [min, max] по запускам")
conv_plot(["R_repair", "R_onepoint", "R_gpx", "R_mincon", "R_elim", "R_gpx_elim", "RandomFeasible"],
          "conv_operators.png", "Операторы вариации и случайный поиск")

# доля допустимых особей в популяции и штраф
plt.figure(figsize=(8, 4.5))
for n in ["P_lam0.5", "P_lam2", "P_lam10", "P_lam2_eps"]:
    if n in stats:
        H = hist_of(n); plt.plot(H[0, :, 0], H[:, :, 5].mean(0), label=n)
plt.xlabel("число вычислений фитнес-функции"); plt.ylabel("доля допустимых особей в популяции")
plt.title("Доля допустимых решений при штрафах"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/feasible_share.png", dpi=140); plt.close()

# F: min/mean/max в одном прогоне-среднем для итогового алгоритма
for n in ["R_gpx_elim"]:
    if n in stats:
        H = hist_of(n); x = H[0, :, 0]
        plt.figure(figsize=(8, 4.5))
        for col, lab in [(1, "min F"), (2, "mean F"), (3, "max F")]:
            plt.plot(x, H[:, :, col].mean(0), label=lab)
        plt.xlabel("вычисления фитнес-функции"); plt.ylabel("F (среднее по запускам)")
        plt.title(f"{n}: динамика min/mean/max фитнеса"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig("figures/fitness_stats.png", dpi=140); plt.close()

# boxplot
plt.figure(figsize=(10, 4.8))
data = [stats[n]["ks"][~np.isnan(stats[n]["ks"])] for n in order]
plt.boxplot(data, tick_labels=order); plt.axhline(CHI, color="k", ls="--", lw=1)
plt.axhline(ds_k, color="gray", ls=":", lw=1)
plt.xticks(rotation=35, ha="right", fontsize=8); plt.ylabel("лучшее допустимое число цветов")
plt.title("Распределение результатов серий запусков"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/boxplot.png", dpi=140); plt.close()

# рисунок графа с лучшей раскраской
best = json.load(open("results/best_solutions.json"))
MAIN = "R_gpx_elim"                         # итоговая конфигурация; при равном k показываем её решение
bn = min((n for n in best), key=lambda n: (best[n]["k"], n != MAIN, n))
col = np.array(best[bn]["colors"])
pr = Problem(g, g.max_deg + 1, info["clique_lb"])
print("best solution from", bn, metrics(pr, col.astype(np.int64)))
rng = np.random.default_rng(0)
pos = rng.random((g.n, 2))
for it in range(300):                       # простая силовая раскладка (Fruchterman–Reingold)
    d = pos[:, None, :] - pos[None, :, :]; dist = np.linalg.norm(d, axis=2) + 1e-9
    force = (d / dist[..., None] ** 2 * 0.05).sum(1)
    for u, v in g.edges:
        dd = pos[u] - pos[v]; ll = np.linalg.norm(dd) + 1e-9
        force[u] -= dd * ll * 0.5; force[v] += dd * ll * 0.5
    pos += np.clip(force, -0.05, 0.05) * (1 - it / 300)
plt.figure(figsize=(7.5, 7.5))
for u, v in g.edges:
    plt.plot(*zip(pos[u], pos[v]), color="gray", lw=0.25, alpha=0.5, zorder=1)
uc = sorted(set(col.tolist())); cmap = plt.get_cmap("tab10")
plt.scatter(pos[:, 0], pos[:, 1], c=[cmap(uc.index(c) % 10) for c in col], s=140, zorder=2, edgecolors="k")
for i in range(g.n):
    plt.text(pos[i, 0], pos[i, 1], str(i), fontsize=6, ha="center", va="center", zorder=3)
plt.axis("off"); plt.title(f"Лучшая найденная раскраска: {len(uc)} цветов ({bn})")
plt.tight_layout(); plt.savefig("figures/best_coloring.png", dpi=140); plt.close()

# таблица классов лучшей раскраски
with open("results/best_coloring_table.csv", "w") as fh:
    fh.write("color,size,vertices\n")
    for j, c in enumerate(uc):
        vs = np.flatnonzero(col == c)
        fh.write(f"{j},{len(vs)},\"{' '.join(map(str, vs))}\"\n")

# допустимые / недопустимые примеры (на этом же графе)
ex = {}
ex["feasible_1_DSATUR"] = ds.tolist()
ex["feasible_2_GA_best"] = [uc.index(c) for c in col]
r2 = np.random.default_rng(7)
c1 = np.array(ex["feasible_2_GA_best"]); c1b = c1.copy()
u, v = g.edges[0]; c1b[u] = c1b[v]                     # u получает цвет соседа v: конфликт с v и
ex["infeasible_1_neighbor_color"] = c1b.tolist()       # со всеми соседями u этого цвета, k не меняется
c2 = r2.integers(0, 6, g.n)                            # случайная 6-раскраска
ex["infeasible_2_random_6_colors"] = c2.tolist()
out = {}
for k, c in ex.items():
    c = np.array(c); kk, conf, _ = metrics(pr, c.astype(np.int64))
    out[k] = {"colors_used": kk, "conflicts": conf, "feasible": conf == 0, "vector": c.tolist()}
json.dump(out, open("results/feasibility_examples.json", "w"), indent=1)
for k, v in out.items():
    print(k, v["colors_used"], v["conflicts"], v["feasible"])

for n in order:
    s = stats[n]
    print(f"{n:16s} best={int(s['best'])} mean={s['mean']:.2f} med={s['median']} std={s['std']:.2f} "
          f"worst={int(s['worst'])} chi%={s['hit_opt']:.2f} feasRuns={s['feasible_runs']:.2f} "
          f"popFeas={s['final_pop_feasible']:.2f} t={s['time']:.1f}s evBest={s['evals_to_best']:.0f}")
print("DSATUR", ds_k)
for t in tests:
    print(t[0], "vs", t[1], f"{t[2]:.2f} / {t[3]:.2f} p={t[4]:.3g}")
