from __future__ import annotations

import csv
import html
import itertools
import math
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"


def fnum(value: float, digits: int = 4) -> str:
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def attrs(**items: object) -> str:
    parts = []
    for key, value in items.items():
        if value is None:
            continue
        if key.endswith("_"):
            key = key[:-1]
        key = key.replace("_", "-")
        parts.append(f'{key}="{esc(value)}"')
    return " ".join(parts)


class Svg:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.title = title
        self.elements: list[str] = []

    def add(self, element: str) -> None:
        self.elements.append(element)

    def line(self, x1, y1, x2, y2, **style) -> None:
        self.add(f"<line {attrs(x1=x1, y1=y1, x2=x2, y2=y2, **style)} />")

    def polyline(self, pts, **style) -> None:
        if len(pts) < 2:
            return
        points = " ".join(f"{fnum(x, 3)},{fnum(y, 3)}" for x, y in pts)
        self.add(f"<polyline {attrs(points=points, **style)} />")

    def polygon(self, pts, **style) -> None:
        points = " ".join(f"{fnum(x, 3)},{fnum(y, 3)}" for x, y in pts)
        self.add(f"<polygon {attrs(points=points, **style)} />")

    def circle(self, x, y, r, **style) -> None:
        self.add(f"<circle {attrs(cx=x, cy=y, r=r, **style)} />")

    def text(self, x, y, text, size=14, anchor="middle", rotate=None, **style) -> None:
        transform = None
        if rotate is not None:
            transform = f"rotate({rotate} {x} {y})"
        self.add(
            f"<text {attrs(x=x, y=y, font_size=size, text_anchor=anchor, transform=transform, **style)}>"
            f"{esc(text)}</text>"
        )

    def arrow(self, x1, y1, x2, y2, **style) -> None:
        defaults = dict(stroke="#111", stroke_width=2, fill="none", marker_end="url(#arrow)")
        defaults.update(style)
        self.line(x1, y1, x2, y2, **defaults)

    def write(self, path: Path) -> None:
        css = """
        text { font-family: "Helvetica Neue", Arial, sans-serif; fill: #111; }
        .axis { stroke: #111; stroke-width: 2.2; fill: none; }
        .thin { stroke: #111; stroke-width: 1.4; fill: none; }
        .dash { stroke: #111; stroke-width: 1.2; fill: none; stroke-dasharray: 8 7; }
        .frontier { stroke: #0a5c84; stroke-width: 4; fill: none; stroke-linecap: round; stroke-linejoin: round; }
        .frontier-fill { fill: #d8eef6; stroke: #0a5c84; stroke-width: 2; }
        .soft { fill: #e7edf1; stroke: none; opacity: 0.85; }
        .dot { fill: #73808a; opacity: 0.55; }
        .label { font-weight: 700; }
        """
        body = "\n".join(self.elements)
        path.write_text(
            f"""<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" role="img" aria-label="{esc(self.title)}">
<title>{esc(self.title)}</title>
<defs>
<marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
<path d="M0,0 L10,4 L0,8 Z" fill="#111"/>
</marker>
</defs>
<style>{css}</style>
<rect width="100%" height="100%" fill="#fff"/>
{body}
</svg>
""",
            encoding="utf-8",
        )


class Plot:
    def __init__(self, svg: Svg, xlim, ylim, margin=(70, 35, 50, 65)):
        self.svg = svg
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
        self.left, self.top, self.right, self.bottom = margin
        self.w = svg.width - self.left - self.right
        self.h = svg.height - self.top - self.bottom

    def map(self, p):
        x, y = p
        sx = self.left + (x - self.x0) / (self.x1 - self.x0) * self.w
        sy = self.top + (self.y1 - y) / (self.y1 - self.y0) * self.h
        return sx, sy

    def axes(self, xlabel, ylabel):
        x0, y0 = self.map((self.x0, self.y0))
        x1, _ = self.map((self.x1, self.y0))
        _, y1 = self.map((self.x0, self.y1))
        self.svg.line(x0, y0, x1, y0, class_="axis")
        self.svg.line(x0, y0, x0, y1, class_="axis")
        self.svg.text(x1 + 8, y0 + 20, xlabel, anchor="start", size=16, class_="label")
        self.svg.text(x0 - 20, y1 - 8, ylabel, anchor="end", size=16, class_="label")

    def polyline(self, pts, **style):
        self.svg.polyline([self.map(p) for p in pts], **style)

    def polygon(self, pts, **style):
        self.svg.polygon([self.map(p) for p in pts], **style)

    def circle(self, p, r, **style):
        x, y = self.map(p)
        self.svg.circle(x, y, r, **style)

    def text(self, p, text, **style):
        x, y = self.map(p)
        self.svg.text(x, y, text, **style)

    def line(self, a, b, **style):
        x1, y1 = self.map(a)
        x2, y2 = self.map(b)
        self.svg.line(x1, y1, x2, y2, **style)

    def arrow(self, a, b, **style):
        x1, y1 = self.map(a)
        x2, y2 = self.map(b)
        self.svg.arrow(x1, y1, x2, y2, **style)


def covariance_is_spd(sigma: np.ndarray) -> bool:
    return bool(np.all(np.linalg.eigvalsh(sigma) > 1e-10))


def solve_active_target(mu: np.ndarray, sigma: np.ndarray, target: float, active: tuple[int, ...]):
    idx = np.array(active)
    active_mu = mu[idx]
    if len(idx) == 1:
        if abs(float(active_mu[0]) - target) < 1e-9:
            w = np.zeros(len(mu))
            w[idx[0]] = 1.0
            return w
        return None
    if target < float(active_mu.min()) - 1e-9 or target > float(active_mu.max()) + 1e-9:
        return None

    s = sigma[np.ix_(idx, idx)]
    ones = np.ones((len(idx), 1))
    kkt = np.block(
        [
            [2 * s, ones, active_mu[:, None]],
            [ones.T, np.zeros((1, 2))],
            [active_mu[None, :], np.zeros((1, 2))],
        ]
    )
    rhs = np.r_[np.zeros(len(idx)), 1.0, target]
    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        return None
    w = np.zeros(len(mu))
    w[idx] = sol[: len(idx)]
    return w


def min_variance_for_target(mu: np.ndarray, sigma: np.ndarray, target: float):
    best = None
    best_v = math.inf
    best_active = None
    for size in range(1, len(mu) + 1):
        for active in itertools.combinations(range(len(mu)), size):
            w = solve_active_target(mu, sigma, target, active)
            if w is None:
                continue
            if abs(float(w.sum()) - 1.0) > 1e-7:
                continue
            if abs(float(mu @ w) - target) > 1e-7:
                continue
            if np.any(w < -1e-8):
                continue
            w = np.maximum(w, 0.0)
            w = w / w.sum()
            v = float(w @ sigma @ w)
            if v < best_v:
                best = w
                best_v = v
                best_active = active
    if best is None:
        return None
    return {"target": target, "variance": best_v, "weights": best, "active": best_active}


def frontier_points(mu: np.ndarray, sigma: np.ndarray, targets: np.ndarray):
    pts = []
    for target in targets:
        solved = min_variance_for_target(mu, sigma, float(target))
        if solved is not None:
            pts.append(solved)
    return pts


def efficient_points(mu: np.ndarray, sigma: np.ndarray, samples: int = 240):
    all_pts = frontier_points(mu, sigma, np.linspace(float(mu.min()), float(mu.max()), samples))
    gmv = min(all_pts, key=lambda item: item["variance"])
    efficient = [pt for pt in all_pts if pt["target"] >= gmv["target"] - 1e-9]
    if efficient[0]["target"] > gmv["target"] + 1e-9:
        efficient.insert(0, gmv)
    return efficient, gmv, all_pts


def unconstrained_target_weights(mu: np.ndarray, sigma: np.ndarray, target: float):
    n = len(mu)
    ones = np.ones((n, 1))
    kkt = np.block(
        [
            [2 * sigma, ones, mu[:, None]],
            [ones.T, np.zeros((1, 2))],
            [mu[None, :], np.zeros((1, 2))],
        ]
    )
    rhs = np.r_[np.zeros(n), 1.0, target]
    sol = np.linalg.solve(kkt, rhs)
    return sol[:n]


def simplex_grid(step: int):
    for i in range(step + 1):
        for j in range(step + 1 - i):
            k = step - i - j
            yield np.array([i, j, k], dtype=float) / step


def quadratic_in_x(sigma: np.ndarray):
    # For a three-security portfolio w = [x1, x2, 1 - x1 - x2].
    base = np.array([0.0, 0.0, 1.0])
    bmat = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    m = bmat.T @ sigma @ bmat
    center = -np.linalg.solve(m, bmat.T @ sigma @ base)
    w_center = base + bmat @ center
    v_center = float(w_center @ sigma @ w_center)
    return center, m, v_center


def ellipse_points(center, m, level, count=220):
    vals, vecs = np.linalg.eigh(m)
    pts = []
    for t in np.linspace(0, 2 * math.pi, count):
        local = np.array([math.cos(t), math.sin(t)])
        delta = vecs @ (np.sqrt(level / vals) * local)
        pts.append(tuple(center + delta))
    return pts


def line_rect_intersections(a, b, c, xlim, ylim):
    # a*x + b*y + c = 0 clipped to a rectangle.
    xs = []
    for x in xlim:
        if abs(b) > 1e-12:
            y = -(a * x + c) / b
            if ylim[0] - 1e-9 <= y <= ylim[1] + 1e-9:
                xs.append((x, y))
    for y in ylim:
        if abs(a) > 1e-12:
            x = -(b * y + c) / a
            if xlim[0] - 1e-9 <= x <= xlim[1] + 1e-9:
                xs.append((x, y))
    unique = []
    for p in xs:
        if not any(abs(p[0] - q[0]) < 1e-7 and abs(p[1] - q[1]) < 1e-7 for q in unique):
            unique.append(p)
    return unique[:2]


def triangle_labels(plot: Plot) -> None:
    plot.text((0, 1), "a", dx="-8", dy="-8", anchor="end", size=15, class_="label")
    plot.text((1, 0), "b", dx="8", dy="20", anchor="start", size=15, class_="label")
    plot.text((0, 0), "c", dx="-8", dy="18", anchor="end", size=15, class_="label")


def draw_three_security_geometry(path: Path, mu, sigma, title: str, case_note: str):
    efficient, gmv, all_pts = efficient_points(mu, sigma)
    center, m, v_center = quadratic_in_x(sigma)
    svg = Svg(740, 620, title)
    plot = Plot(svg, (-0.12, 1.15), (-0.12, 1.18), margin=(70, 55, 40, 60))
    plot.axes("X1", "X2")
    plot.polygon([(0, 0), (1, 0), (0, 1)], fill="#f7f9fa", stroke="#111", stroke_width=1.8)
    triangle_labels(plot)

    for level in [0.004, 0.009, 0.018, 0.035, 0.06]:
        pts = ellipse_points(center, m, level)
        plot.polyline(pts, class_="thin")

    a = float(mu[0] - mu[2])
    b = float(mu[1] - mu[2])
    for target in np.linspace(float(mu.min()) + 0.012, float(mu.max()) - 0.006, 6):
        pts = line_rect_intersections(a, b, float(mu[2] - target), (-0.12, 1.15), (-0.12, 1.18))
        if len(pts) == 2:
            plot.polyline(pts, class_="dash")

    critical = []
    for target in np.linspace(float(mu.min()), float(mu.max()), 140):
        w = unconstrained_target_weights(mu, sigma, float(target))
        critical.append((float(w[0]), float(w[1])))
    plot.polyline(critical, stroke="#111", stroke_width=2.2, fill="none")

    eff_xy = [(float(pt["weights"][0]), float(pt["weights"][1])) for pt in efficient]
    plot.polyline(eff_xy, class_="frontier")
    plot.circle((float(center[0]), float(center[1])), 3.8, fill="#111")
    plot.text((float(center[0]), float(center[1])), "X", dx="11", dy="-8", anchor="start", size=15, class_="label")
    plot.arrow((0.13, 1.02), (0.45, 1.14), stroke_width=2)
    plot.text((0.48, 1.15), "direction of increasing E", anchor="start", size=13)

    svg.text(465, 95, "isomean lines", size=13, anchor="start")
    svg.line(575, 91, 660, 91, class_="dash")
    svg.text(465, 125, "isovariance curves", size=13, anchor="start")
    svg.polyline([(590, 119), (610, 113), (630, 119), (630, 130), (610, 136), (590, 130), (590, 119)], class_="thin")
    svg.text(465, 155, "efficient portfolios", size=13, anchor="start")
    svg.line(590, 151, 660, 151, class_="frontier")
    svg.text(440, 603, case_note, size=12, anchor="middle")
    svg.write(path)


def figure_1(path: Path, mu, sigma):
    efficient, gmv, all_pts = efficient_points(mu, sigma)
    cloud = [(float(mu @ w), float(w @ sigma @ w)) for w in simplex_grid(42)]
    xs = [p[0] for p in cloud]
    ys = [p[1] for p in cloud]
    xlim = (min(xs) - 0.006, max(xs) + 0.006)
    ylim = (min(ys) - 0.006, max(ys) + 0.008)
    svg = Svg(680, 520, "Figure 1 recreation: attainable E,V combinations")
    plot = Plot(svg, xlim, ylim, margin=(70, 35, 35, 65))
    plot.axes("E", "V")
    for p in cloud:
        plot.circle(p, 1.25, fill="#73808a", opacity=0.42)
    eff = [(float(pt["target"]), float(pt["variance"])) for pt in efficient]
    plot.polyline(eff, class_="frontier")
    plot.text((np.mean(xs), max(ys) * 0.92), "attainable E,V combinations", size=14, class_="label")
    last = eff[len(eff) // 2]
    label = (xlim[1] - 0.004, last[1] + 0.017)
    plot.arrow((label[0] - 0.015, label[1] - 0.004), last, stroke_width=2)
    plot.text(label, "efficient E,V combinations", anchor="end", size=13, class_="label")
    svg.write(path)


def project_tetra(weights, vertices):
    p = np.zeros(2)
    for weight, vertex in zip(weights, vertices):
        p += weight * np.array(vertex, dtype=float)
    return tuple(p)


def figure_4(path: Path, mu, sigma):
    efficient, gmv, all_pts = efficient_points(mu, sigma, samples=260)
    svg = Svg(700, 560, "Figure 4 recreation: four-security efficient path")
    vertices = [(360, 70), (105, 430), (600, 420), (325, 515)]
    labels = ["X1", "X2", "X3", "X4"]
    for i, j in itertools.combinations(range(4), 2):
        svg.line(*vertices[i], *vertices[j], class_="thin")
    for vertex, label in zip(vertices, labels):
        svg.circle(vertex[0], vertex[1], 3.5, fill="#111")
        svg.text(vertex[0], vertex[1] - 12, label, size=16, class_="label")
    pts = [project_tetra(pt["weights"], vertices) for pt in efficient]
    svg.polyline(pts, class_="frontier")
    for idx in [20, 70, 130]:
        if idx < len(pts) - 2:
            svg.arrow(pts[idx][0], pts[idx][1], pts[idx + 2][0], pts[idx + 2][1], stroke="#0a5c84", stroke_width=3)
    svg.text(438, 105, "efficient portfolios", size=15, anchor="start", class_="label")
    svg.arrow(535, 102, 610, 102, stroke_width=2)
    svg.text(350, 540, "projected tetrahedron of feasible long-only portfolios", size=13)
    svg.write(path)


def project_3d(x1, x2, z):
    ox, oy = 210, 470
    ex = np.array([380, 0])
    ey = np.array([-145, -130])
    ez = np.array([0, -250])
    p = np.array([ox, oy]) + x1 * ex + x2 * ey + z * ez
    return tuple(p)


def figure_5(path: Path, mu, sigma):
    efficient, gmv, all_pts = efficient_points(mu, sigma)
    targets = np.array([pt["target"] for pt in efficient])
    variances = np.array([pt["variance"] for pt in efficient])
    e_norm = (targets - targets.min()) / (targets.max() - targets.min()) * 0.55 + 0.30
    v_norm = (variances - variances.min()) / (variances.max() - variances.min()) * 0.55 + 0.18
    svg = Svg(700, 560, "Figure 5 recreation: E and V over efficient portfolios")
    base = [project_3d(0, 0, 0), project_3d(1, 0, 0), project_3d(0, 1, 0)]
    svg.polygon(base, fill="#f7f9fa", stroke="#111", stroke_width=1.6)
    base_path = [project_3d(float(pt["weights"][0]), float(pt["weights"][1]), 0) for pt in efficient]
    svg.polyline(base_path, class_="frontier")
    for idx in np.linspace(0, len(efficient) - 1, 5, dtype=int):
        w = efficient[idx]["weights"]
        x, y = project_3d(float(w[0]), float(w[1]), 0)
        xe, ye = project_3d(float(w[0]), float(w[1]), float(e_norm[idx]))
        svg.line(x, y, xe, ye, stroke="#777", stroke_width=1.0, fill="none")
    e_path = [
        project_3d(float(pt["weights"][0]), float(pt["weights"][1]), float(z))
        for pt, z in zip(efficient, e_norm)
    ]
    v_path = [
        project_3d(float(pt["weights"][0]), float(pt["weights"][1]), float(z))
        for pt, z in zip(efficient, v_norm)
    ]
    svg.polyline(e_path, stroke="#111", stroke_width=2.2, fill="none")
    svg.polyline(v_path, stroke="#0a5c84", stroke_width=2.8, fill="none")
    svg.text(*base[0], "c", dx="-8", dy="22", anchor="end", size=14, class_="label")
    svg.text(*base[1], "b / X1", dx="8", dy="22", anchor="start", size=14, class_="label")
    svg.text(*base[2], "a / X2", dx="-8", dy="-8", anchor="end", size=14, class_="label")
    svg.text(495, 135, "E", size=16, class_="label")
    svg.text(505, 205, "V", size=16, fill="#0a5c84", class_="label")
    svg.text(385, 435, "set of efficient portfolios", size=13, anchor="start", class_="label")
    svg.write(path)


def figure_6(path: Path, mu, sigma):
    efficient, gmv, all_pts = efficient_points(mu, sigma)
    ev = [(float(pt["target"]), float(pt["variance"])) for pt in efficient]
    xlim = (min(x for x, _ in ev) - 0.004, max(x for x, _ in ev) + 0.004)
    ylim = (min(y for _, y in ev) - 0.004, max(y for _, y in ev) + 0.01)
    svg = Svg(650, 510, "Figure 6 recreation: efficient E,V combinations")
    plot = Plot(svg, xlim, ylim, margin=(70, 35, 40, 65))
    plot.axes("E", "V")
    plot.polyline(ev, class_="frontier")
    mid = ev[len(ev) // 2]
    plot.arrow((mid[0] + 0.012, mid[1] + 0.012), mid, stroke_width=2)
    plot.text((mid[0] + 0.014, mid[1] + 0.014), "efficient E,V combinations", anchor="start", size=13, class_="label")
    svg.write(path)


def figure_7(path: Path, mu, sigma):
    center, m, v_center = quadratic_in_x(sigma)
    level = 0.018
    candidates = [
        p
        for p in ellipse_points(center, m, level, 360)
        if p[0] >= 0.03 and p[1] >= 0.03 and p[0] + p[1] <= 0.97
    ]
    best = None
    for p1 in candidates:
        for p2 in candidates:
            d = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            if mid[0] >= 0 and mid[1] >= 0 and mid[0] + mid[1] <= 1:
                if best is None or d > best[0]:
                    best = (d, p1, p2, mid)
    _, p1, p2, mid = best
    svg = Svg(690, 560, "Figure 7 recreation: mixing equal-variance portfolios")
    plot = Plot(svg, (-0.1, 1.1), (-0.1, 1.1), margin=(70, 40, 45, 65))
    plot.axes("X1", "X2")
    plot.polygon([(0, 0), (1, 0), (0, 1)], fill="#f7f9fa", stroke="#111", stroke_width=1.6)
    triangle_labels(plot)
    for lev in [0.010, level]:
        plot.polyline(ellipse_points(center, m, lev), class_="thin")
    plot.line(p1, p2, stroke="#111", stroke_width=2.2)
    plot.circle(p1, 4, fill="#111")
    plot.circle(p2, 4, fill="#111")
    plot.circle(mid, 4, fill="#0a5c84")
    plot.circle((float(center[0]), float(center[1])), 3, fill="#111")
    plot.text(p1, "P'", dx="-10", dy="-8", anchor="end", size=14, class_="label")
    plot.text(p2, "P''", dx="9", dy="-7", anchor="start", size=14, class_="label")
    plot.text(mid, "P", dx="0", dy="-10", anchor="middle", size=14, class_="label")
    plot.text((float(center[0]), float(center[1])), "X", dx="11", dy="-5", anchor="start", size=14, class_="label")
    plot.arrow((0.82, 0.67), (0.61, 0.58), stroke_width=2)
    plot.arrow((0.82, 0.67), (0.58, 0.49), stroke_width=2)
    plot.text((0.86, 0.68), "isovariance lines", anchor="start", size=13, class_="label")
    svg.write(path)


def write_csv(path: Path, mu, sigma):
    efficient, gmv, _ = efficient_points(mu, sigma)
    targets = np.linspace(float(gmv["target"]), float(mu.max()), 121)
    rows = frontier_points(mu, sigma, targets)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target_return", "variance", "std_dev", "w1", "w2", "w3", "active_set"])
        for row in rows:
            w = row["weights"]
            active = " ".join(str(i + 1) for i in row["active"])
            writer.writerow(
                [
                    fnum(row["target"], 8),
                    fnum(row["variance"], 8),
                    fnum(math.sqrt(row["variance"]), 8),
                    fnum(float(w[0]), 8),
                    fnum(float(w[1]), 8),
                    fnum(float(w[2]), 8),
                    active,
                ]
            )


def matrix_table(matrix: np.ndarray, row_labels=None, col_labels=None) -> str:
    if row_labels is None:
        row_labels = [str(i + 1) for i in range(matrix.shape[0])]
    if col_labels is None:
        col_labels = [str(i + 1) for i in range(matrix.shape[1])]
    head = "".join(f"<th>{esc(c)}</th>" for c in [""] + list(col_labels))
    rows = [f"<tr>{head}</tr>"]
    for label, row in zip(row_labels, matrix):
        cells = "".join(f"<td>{fnum(float(x), 4)}</td>" for x in row)
        rows.append(f"<tr><th>{esc(label)}</th>{cells}</tr>")
    return "<table>" + "\n".join(rows) + "</table>"


def write_report(path: Path, figures: list[tuple[str, str]], mu_a, sigma_a, mu_b, sigma_b, mu4, sigma4):
    fig_html = "\n".join(
        f'<section class="figure"><h2>{esc(title)}</h2><img src="{esc(filename)}" alt="{esc(title)}"></section>'
        for title, filename in figures
    )
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recreating Markowitz Portfolio Selection Results</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5;
  color: #1d252c;
  margin: 0;
  background: #ffffff;
}}
main {{ max-width: 980px; margin: 0 auto; padding: 36px 22px 64px; }}
h1 {{ font-size: 30px; margin: 0 0 12px; }}
h2 {{ font-size: 20px; margin: 32px 0 12px; }}
p {{ max-width: 820px; }}
.note {{ background: #eef5f8; border-left: 4px solid #0a5c84; padding: 12px 16px; }}
.equations {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #f6f8fa; padding: 14px 16px; }}
.figure img {{ width: 100%; max-width: 760px; border: 1px solid #d8dee4; display: block; }}
table {{ border-collapse: collapse; margin: 10px 0 24px; font-size: 14px; }}
th, td {{ border: 1px solid #d8dee4; padding: 6px 10px; text-align: right; }}
th {{ background: #f6f8fa; }}
a {{ color: #0a5c84; }}
</style>
</head>
<body>
<main>
<h1>Recreating Markowitz Portfolio Selection Results</h1>
<p class="note">The 1952 paper contains schematic figures and formulas, not an empirical dataset. This recreation therefore uses explicit synthetic beliefs, then recomputes the expected-return and variance geometry that the paper describes.</p>

<h2>Core Computation</h2>
<div class="equations">
E(w) = sum_i w_i * mu_i<br>
V(w) = sum_i sum_j w_i * w_j * sigma_ij<br>
sum_i w_i = 1, w_i &gt;= 0
</div>
<p>For each target expected return, the script enumerates active long-only asset sets and solves the quadratic KKT system for the minimum-variance portfolio. The efficient frontier begins at the feasible minimum-variance portfolio and ends at the maximum-return asset.</p>

<h2>Three-Security Example A</h2>
<p>This case recreates the paper's interior minimum-variance geometry used in the three-security diagrams.</p>
<p>Means: {", ".join(fnum(float(x), 4) for x in mu_a)}</p>
{matrix_table(sigma_a, row_labels=["1", "2", "3"], col_labels=["1", "2", "3"])}

<h2>Three-Security Example B</h2>
<p>This case recreates the alternative geometry where the unconstrained variance center lies outside the feasible triangle.</p>
<p>Means: {", ".join(fnum(float(x), 4) for x in mu_b)}</p>
{matrix_table(sigma_b, row_labels=["1", "2", "3"], col_labels=["1", "2", "3"])}

<h2>Four-Security Example</h2>
<p>Means: {", ".join(fnum(float(x), 4) for x in mu4)}</p>
{matrix_table(sigma4, row_labels=["1", "2", "3", "4"], col_labels=["1", "2", "3", "4"])}

{fig_html}

<h2>Reproducibility</h2>
<p>Run <code>python3 recreate_markowitz_results.py</code> from this folder to regenerate every SVG, this report, and <code>efficient_frontier_case_a.csv</code>.</p>
</main>
</body>
</html>
"""
    path.write_text(report, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    mu_a = np.array([0.15, 0.12, 0.05])
    sigma_a = np.array(
        [
            [0.09, 0.01, 0.015],
            [0.01, 0.05, 0.01],
            [0.015, 0.01, 0.04],
        ],
        dtype=float,
    )
    mu_b = np.array([0.15, 0.12, 0.05])
    sigma_b = np.array(
        [
            [0.05, 0.02, 0.04],
            [0.02, 0.04, 0.06],
            [0.04, 0.06, 0.16],
        ],
        dtype=float,
    )
    mu4 = np.array([0.16, 0.12, 0.08, 0.04])
    sigma4 = np.array(
        [
            [0.10, 0.03, 0.02, 0.01],
            [0.03, 0.06, 0.015, 0.01],
            [0.02, 0.015, 0.04, 0.005],
            [0.01, 0.01, 0.005, 0.03],
        ],
        dtype=float,
    )
    for name, sigma in [("Example A", sigma_a), ("Example B", sigma_b), ("Example 4", sigma4)]:
        if not covariance_is_spd(sigma):
            raise ValueError(f"{name} covariance matrix is not positive definite.")

    figures = [
        ("Figure 1: attainable E,V combinations", "figure_1_attainable_ev.svg"),
        ("Figure 2: three-security efficient portfolios, interior center", "figure_2_three_security_inside.svg"),
        ("Figure 3: three-security efficient portfolios, outside center", "figure_3_three_security_outside.svg"),
        ("Figure 4: four-security efficient path", "figure_4_four_security.svg"),
        ("Figure 5: E and V over the efficient set", "figure_5_ev_over_portfolios.svg"),
        ("Figure 6: efficient E,V frontier", "figure_6_efficient_ev.svg"),
        ("Figure 7: diversification between equal-variance portfolios", "figure_7_equal_variance_mix.svg"),
    ]
    figure_1(OUT / figures[0][1], mu_a, sigma_a)
    draw_three_security_geometry(
        OUT / figures[1][1],
        mu_a,
        sigma_a,
        "Figure 2 recreation: three-security efficient portfolios, interior center",
        "minimum-variance point inside the attainable set",
    )
    draw_three_security_geometry(
        OUT / figures[2][1],
        mu_b,
        sigma_b,
        "Figure 3 recreation: three-security efficient portfolios, outside center",
        "variance center outside the attainable set",
    )
    figure_4(OUT / figures[3][1], mu4, sigma4)
    figure_5(OUT / figures[4][1], mu_a, sigma_a)
    figure_6(OUT / figures[5][1], mu_a, sigma_a)
    figure_7(OUT / figures[6][1], mu_a, sigma_a)
    write_csv(OUT / "efficient_frontier_case_a.csv", mu_a, sigma_a)
    write_report(OUT / "markowitz_recreation_report.html", figures, mu_a, sigma_a, mu_b, sigma_b, mu4, sigma4)
    write_report(DOCS / "index.html", figures, mu_a, sigma_a, mu_b, sigma_b, mu4, sigma4)
    (DOCS / ".nojekyll").touch()
    for _, filename in figures:
        shutil.copy2(OUT / filename, DOCS / filename)


if __name__ == "__main__":
    main()
