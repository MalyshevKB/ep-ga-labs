# ЛР №2. ГА для дискретной задачи с ограничениями — раскраска графа (вариант 16)

**Выполнил:** Малышев Кирилл Борисович, группа Б24-527

МИФИ, «Эволюционное программирование и ГА», группа Б24-527, N=15, Y=26 → V = 1 + ((17·15+7·2+26) mod 20) = 16.

## Установка и запуск
Проверено на Python 3.14, numpy 2.5, matplotlib 3.11, scipy 1.18, Windows 11.

    pip install -r requirements.txt
    python make_graph.py                    # граф G(80;0.3), seed 2026 -> data/
    python run_experiments.py --runs 20     # все конфигурации, 20 запусков (seed 1000..1019)
    python analyze.py                       # таблицы, графики, примеры решений
    python make_flowchart.py                # блок-схема -> figures/flowchart.png

Один запуск ГА (seed задаётся параметром):

    python -c "from gc_ga.graph import *; from gc_ga.ga import *; \
    g,_=load_graph('data/graph.json'); pr=Problem(g,g.max_deg+1,5); \
    r=run_ga(pr,Config(name='x',constraint='repair',crossover='gpx',mutation='elim'),seed=1000); print(r['best_k'])"

Полная серия занимает около 45 минут на одном ядре; `run_experiments.py` возобновляется с места остановки:
уже посчитанные пары (конфигурация, seed) из `results/runs.csv` пропускаются. Поэтому в поставляемой папке, где все
запуски уже есть, он ничего не пересчитывает. Чтобы пересчитать серию с нуля, удалите `results/runs.csv`,
`results/best_solutions.json` и `results/hist_*.npy`, затем запустите `run_experiments.py` и `analyze.py`.

## Состав
- `gc_ga/graph.py` — граф, DSATUR, максимальная клика (нижняя оценка), точный перебор хроматического числа
- `gc_ga/ga.py` — ГА: представление, штраф/ремонт, кроссоверы (uniform, one-point, GPX), мутации (random, mincon, elim)
- `config.json` — параметры всех конфигураций и seed; `data/` — граф (json, csv), `data/graph_info.json`
- `results/` — `runs.csv` (все запуски), `summary.csv`, `mannwhitney.csv`, `best_solutions.json`, `best_coloring_table.csv`, `feasibility_examples.json`, `hist_*.npy`
- `figures/` — графики; `report.md` — отчёт
