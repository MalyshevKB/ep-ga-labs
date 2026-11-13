"""Минимальная отрисовка блок-схем средствами matplotlib (без внешних программ).

Узел: (x, y, текст, вид), вид — 'term' (начало/конец), 'proc' (действие), 'dec' (условие).
Ребро: (откуда, сторона, куда, сторона, подпись, шаги), сторона — 't', 'b', 'l', 'r';
шаги — ортогональный маршрут: ('x', значение) — горизонтально до x, ('y', значение) — вертикально до y.
Последний отрезок до точки входа достраивается автоматически."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

CHAR_W, LINE_H, FONT = 0.122, 0.30, 9


def _size(text, kind):
    lines = text.split("\n")
    w = max(2.4, max(len(s) for s in lines) * CHAR_W + 0.5)
    h = len(lines) * LINE_H + 0.32
    if kind == "dec":
        w, h = w * 1.35, h * 1.55
    return w, h


def _port(node, side):
    x, y, w, h = node
    return {"t": (x, y + h / 2), "b": (x, y - h / 2), "l": (x - w / 2, y), "r": (x + w / 2, y)}[side]


def draw(nodes, edges, path, scale=1.0):
    geo = {k: (x, y) + _size(t, kind) for k, (x, y, t, kind) in nodes.items()}
    xs = [x - w / 2 for x, y, w, h in geo.values()] + [x + w / 2 for x, y, w, h in geo.values()]
    ys = [y - h / 2 for x, y, w, h in geo.values()] + [y + h / 2 for x, y, w, h in geo.values()]
    for e in edges:                                    # маршруты рёбер тоже должны поместиться
        for ax_, v in e[5]:
            (xs if ax_ == "x" else ys).append(v)
    x0, x1, y0, y1 = min(xs) - 0.4, max(xs) + 0.4, min(ys) - 0.3, max(ys) + 0.3
    fig = plt.figure(figsize=((x1 - x0) * 0.62 * scale, (y1 - y0) * 0.62 * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.axis("off")

    for k, (x, y, text, kind) in nodes.items():
        _, _, w, h = geo[k]
        if kind == "dec":
            ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                                 closed=True, fc="#fff4d6", ec="#7a5c00", lw=1.2))
        else:
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                        boxstyle="round,pad=0,rounding_size=%.2f" % (h / 2 if kind == "term" else 0.08),
                                        fc="#e3f0ff" if kind == "term" else "#ffffff", ec="#1f3f66", lw=1.2))
        ax.text(x, y, text, ha="center", va="center", fontsize=FONT, linespacing=1.25)

    for src, s_side, dst, d_side, label, steps in edges:
        p = [_port(geo[src], s_side)]
        for ax_, v in steps:
            p.append((v, p[-1][1]) if ax_ == "x" else (p[-1][0], v))
        end = _port(geo[dst], d_side)
        cx, cy = p[-1]
        if abs(cx - end[0]) > 1e-9 and abs(cy - end[1]) > 1e-9:
            p.append((end[0], cy) if d_side in "tb" else (cx, end[1]))
        p.append(end)
        for a, b in zip(p[:-2], p[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#333", lw=1.1)
        ax.annotate("", xy=p[-1], xytext=p[-2], arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1,
                                                                shrinkA=0, shrinkB=0, mutation_scale=12))
        if label:
            (ax0, ay0), (ax1, ay1) = p[0], p[1]
            horiz = abs(ay1 - ay0) < 1e-9
            ax.text(ax0 + (0.12 if not horiz else (0.35 if ax1 > ax0 else -0.35)),
                    ay0 + (0.14 if horiz else -0.22), label, fontsize=FONT - 1, color="#7a0000",
                    ha="left" if not horiz else "center", va="center")
    fig.savefig(path, dpi=150)
    plt.close(fig)
