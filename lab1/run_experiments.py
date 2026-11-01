"""Серия независимых запусков: python run_experiments.py [--runs 30] [--seed0 1000] [--budget 100000]"""
import argparse, json, os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from ga import Config, run_ga, run_random, D

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=30)
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--budget", type=int, default=100_000)
ap.add_argument("--out", default="results")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

B = a.budget
CONFIGS = [
    Config(name="GA_base(sigma=0.01)", sigma0=0.01, budget=B),
    Config(name="GA_sigma=0.002", sigma0=0.002, budget=B),
    Config(name="GA_sigma=0.05", sigma0=0.05, budget=B),
    Config(name="GA_adaptive(sigma0=0.05)", sigma0=0.05, adaptive=True, budget=B),
    Config(name="GA_rank_selection(sigma=0.01)", sigma0=0.01, selection="rank", budget=B),
]
json.dump([c.__dict__ for c in CONFIGS], open(f"{a.out}/configs.json", "w"), indent=1, ensure_ascii=False)

rows, curves = [], {}
grid = np.linspace(1000, B, 100).astype(int)          # общая сетка по числу вычислений f
def on_grid(h, col):
    return np.interp(grid, h[:, 0], h[:, col])         # best-so-far по оценкам

for c in CONFIGS + ["random"]:
    name = "RandomSearch" if c == "random" else c.name
    cur = []
    for k in range(a.runs):
        seed = a.seed0 + k; t = time.time()
        r = run_random(B, seed) if c == "random" else run_ga(c, seed)
        h = r["hist"]
        cur.append(on_grid(h, 1) if c == "random" else np.vstack([on_grid(h, 1), on_grid(h, 2), on_grid(h, 3)]))
        rows.append(dict(config=name, run=k, seed=seed, best_f=r["best_f"], evals=r["evals"], time_s=time.time() - t,
                         **({} if c == "random" else {f"x{i+1}": v for i, v in enumerate(r["best_x"])})))
    curves[name] = np.array(cur)

df = pd.DataFrame(rows); df.to_csv(f"{a.out}/runs.csv", index=False)
def stats(x): return pd.Series(dict(best=x.min(), mean=x.mean(), median=x.median(), std=x.std(ddof=1), worst=x.max(),
                                    **{"success(f<1e-2)": (x < 1e-2).mean(), "success(f<1)": (x < 1).mean()}))
summ = df.groupby("config", sort=False)["best_f"].apply(stats).unstack()
summ["mean_time_s"] = df.groupby("config", sort=False)["time_s"].mean()
summ.to_csv(f"{a.out}/summary.csv"); print(summ.to_string())

# двусторонний тест Манна—Уитни: базовая конфигурация против остальных
base_f = df[df.config == CONFIGS[0].name].best_f
mw = pd.DataFrame([dict(config=n, p_value=mannwhitneyu(base_f, df[df.config == n].best_f).pvalue)
                   for n in df.config.unique() if n != CONFIGS[0].name])
mw.to_csv(f"{a.out}/mannwhitney.csv", index=False); print(mw.to_string(index=False))

# график 1: min/среднее/max по серии для базовой конфигурации (лучшее f, среднее и худшее по популяции — усреднено по запускам)
base = curves[CONFIGS[0].name]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(grid, base[:, 0].min(0), label="min по серии (best)")
ax[0].plot(grid, np.median(base[:, 0], 0), label="медиана по серии (best)")
ax[0].plot(grid, base[:, 0].max(0), label="max по серии (best)")
ax[0].fill_between(grid, np.percentile(base[:, 0], 25, 0), np.percentile(base[:, 0], 75, 0), alpha=.25, label="IQR")
ax[0].set(yscale="log", xlabel="число вычислений f", ylabel="лучшее f", title=f"{CONFIGS[0].name}: лучшее значение, {a.runs} запусков")
ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(grid, base[:, 0].mean(0), label="лучшая особь (среднее по запускам)")
ax[1].plot(grid, base[:, 1].mean(0), label="среднее по популяции")
ax[1].plot(grid, base[:, 2].mean(0), label="худшая особь")
ax[1].set(yscale="log", xlabel="число вычислений f", title="Лучшее / среднее / худшее по популяции")
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{a.out}/convergence_base.png", dpi=140); plt.close()

# график 2: сравнение конфигураций и random search (медиана и IQR)
plt.figure(figsize=(8.5, 5))
for name, cu in curves.items():
    y = cu if cu.ndim == 2 else cu[:, 0]
    plt.plot(grid, np.median(y, 0), label=name)
    plt.fill_between(grid, np.percentile(y, 25, 0), np.percentile(y, 75, 0), alpha=.12)
plt.yscale("log"); plt.xlabel("число вычислений f"); plt.ylabel("лучшее f (медиана, IQR)")
plt.title("Сравнение конфигураций и случайного поиска"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{a.out}/compare_configs.png", dpi=140); plt.close()

# boxplot финальных значений
plt.figure(figsize=(9, 4.8))
names = list(curves); plt.boxplot([df[df.config == n].best_f for n in names], tick_labels=[n.replace("_", "\n", 1) for n in names])
plt.yscale("log"); plt.ylabel("итоговое f"); plt.title("Распределение итоговых значений"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{a.out}/boxplot_final.png", dpi=140); plt.close()
