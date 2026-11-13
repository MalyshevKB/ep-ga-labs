"""Серия экспериментов ЛР №2 (раскраска графа). Запуск: python run_experiments.py --runs 20
Результаты: results/runs.csv, results/hist_<config>.npy, results/best_solutions.json, config.json"""
import argparse, csv, json, os, sys, time
from dataclasses import asdict
import numpy as np
from gc_ga.graph import load_graph, dsatur, max_clique
from gc_ga.ga import Config, Problem, run_ga, run_random_feasible

BASE = dict(pop_size=100, max_evals=50_000, elite=2, tournament=3, p_cross=0.9, p_mut=0.02)
CONFIGS = [
    # --- 1. способы обработки ограничений (операторы: uniform + random) ---
    Config(name="P_lam0.5",  constraint="penalty", lam=0.5, **BASE),
    Config(name="P_lam2",    constraint="penalty", lam=2.0, **BASE),
    Config(name="P_lam10",   constraint="penalty", lam=10.0, **BASE),
    Config(name="R_repair",  constraint="repair", **BASE),
    Config(name="P_lam2_eps", constraint="penalty", lam=2.0, eps=0.5, **BASE),
    # --- 2. операторы (ремонт; меняется один оператор) ---
    Config(name="R_onepoint", constraint="repair", crossover="onepoint", **BASE),
    Config(name="R_gpx",      constraint="repair", crossover="gpx", **BASE),
    Config(name="R_mincon",   constraint="repair", mutation="mincon", **BASE),
    Config(name="R_elim",     constraint="repair", mutation="elim", **BASE),
    # --- 3. итоговые комбинации ---
    Config(name="R_gpx_elim", constraint="repair", crossover="gpx", mutation="elim", **BASE),
    Config(name="P_eps_gpx_elim", constraint="penalty", lam=2.0, eps=0.5, crossover="gpx", mutation="elim", **BASE),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--graph", default="data/graph.json")
    a = ap.parse_args()
    g, meta = load_graph(a.graph)
    info = json.load(open("data/graph_info.json"))
    pr = Problem(g, K=g.max_deg + 1, lb=info["clique_lb"])
    seeds = [a.seed0 + i for i in range(a.runs)]
    if not a.only:   # частичный дозапуск (--only) не перезаписывает описание полной серии
        json.dump({"graph": meta | {"n": g.n, "m": g.m}, "K": pr.K, "seeds": seeds,
                   "configs": [asdict(c) for c in CONFIGS]}, open("config.json", "w"), indent=1)
    os.makedirs("results", exist_ok=True)
    rows_path = "results/runs.csv"
    done = set()
    if os.path.exists(rows_path):
        done = {(r["config"], int(r["seed"])) for r in csv.DictReader(open(rows_path))}
    new_file = not os.path.exists(rows_path)
    f = open(rows_path, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["config", "seed", "best_k", "evals_to_best", "final_best_conf",
                    "final_best_k", "final_feasible_share", "evals", "time_s"])
    bests = json.load(open("results/best_solutions.json")) if os.path.exists("results/best_solutions.json") else {}

    def do(name, fn):
        hists = []
        for s in seeds:
            if (name, s) in done:
                continue
            r = fn(s)
            w.writerow([name, s, r["best_k"], r["evals_to_best"], r.get("final_best_conf", 0),
                        r.get("final_best_k", r["best_k"]), round(r["final_feasible_share"], 4),
                        r["evals"], round(r["time"], 2)]); f.flush()
            hists.append(r["hist"])
            if r["best_c"] is not None and (name not in bests or r["best_k"] < bests[name]["k"]):
                bests[name] = {"k": int(r["best_k"]), "seed": s, "colors": [int(x) for x in r["best_c"]]}
        if hists:
            old = f"results/hist_{name}.npy"
            arr = np.stack(hists)
            if os.path.exists(old):
                arr = np.concatenate([np.load(old), arr])
            np.save(old, arr)
        json.dump(bests, open("results/best_solutions.json", "w"))
        print(name, "готово", time.strftime("%H:%M:%S"), flush=True)

    for cfg in CONFIGS:
        if a.only and cfg.name not in a.only:
            continue
        do(cfg.name, lambda s, cfg=cfg: run_ga(pr, cfg, s))
    if not a.only or "RandomFeasible" in a.only:
        do("RandomFeasible", lambda s: run_random_feasible(pr, BASE["max_evals"], s))
