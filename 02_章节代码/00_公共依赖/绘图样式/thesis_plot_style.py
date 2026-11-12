"""Unified publication-grade figure style for the PhD dissertation (150 mm body width).

Specification: README.md in this folder (3.0, 2026-11-06). Acceptance: check_figures.py.

Benchmarked against IEEE TAC / Automatica visual standards, SciencePlots, and
Nicolas P. Rougier's *Scientific Visualization: Python + Matplotlib*:
- Per-glyph mixed font fallback: Times New Roman (Latin/numerals) + SimSun (CJK) + STIX (Math)
- Inward precision major/minor ticks with recessive hairline gridlines
- White stroke halos for in-canvas annotations and primary trajectories
- White surface rings for multi-robot markers and clean layered topology edges
- 3.0: fixed canvas tiers (LAYOUTS / new_figure) so that figures of the same
  composition have the same printed height, and save_figure() warns when a
  canvas or a text size would violate the specification.
"""
import re
import warnings
from pathlib import Path
import numpy as np
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text
import matplotlib.patheffects as pe

MM = 1 / 25.4

# ---------------------------------------------------------------------------
# 3.0 canvas tiers (printed size, 1:1). README.md section 2 is the normative text.
# ---------------------------------------------------------------------------
TEXT_WIDTH_MM = 150.0     # body width of cugbody.cls; inserted with width=\textwidth
NARROW_WIDTH_MM = 105.0   # narrow single panel; inserted at natural size (no width option)
MIN_FONT_PT = 7.8         # floor for any base text in the final PDF (sub/superscripts exempt)
HEIGHT_TOLERANCE_MM = 4.0 # transitional tolerance for legacy figures only
MAX_HEIGHT_MM = 128.0     # taller canvases need '% figure-exception:' in the figure environment
ABS_MAX_HEIGHT_MM = 190.0 # hard cap so that a 4-line caption still fits on the page

# code: (width_mm, height_mm, nrows, ncols, description)
# Heights were derived (2026-11-06) by measuring the data-axes box under apply_style():
# ordinary data panels land at width:height ~ 1.5 (accepted 1.4-1.6); stacked
# shared-x strips (P1W, V2, V3, right column of P3) at 2.2-4.7.
LAYOUTS = {
    'P1':  (105, 78, 1, 1, '单面板（窄）'),
    'P1W': (150, 78, 1, 1, '单面板（全宽，长时间轴）'),
    'P2':  (150, 58, 1, 2, '1×2 双面板（数据曲线）'),
    'P2E': (150, 66, 1, 2, '1×2 双面板（含等比例轨迹/几何面板）'),
    'P3':  (150, 88, None, None, '三面板：左 1 大图 + 右上下 2 图'),
    'V2':  (150, 100, 2, 1, '2×1 上下共享横轴'),
    'P4':  (150, 116, 2, 2, '2×2 四面板'),
    'V3':  (150, 128, 3, 1, '3×1 上下共享横轴'),
    'R3':  (150, 52, 1, 3, '1×3 快照/示意（无数据坐标轴）'),
    'R6':  (150, 100, 2, 3, '2×3 快照（无数据坐标轴）'),
}
# Conceptual / mechanism / flow diagrams (layout 'D') snap to one of these heights.
D_HEIGHTS = (52, 58, 66, 78, 88, 100, 116)
TIER_HEIGHTS = tuple(sorted({v[1] for v in LAYOUTS.values()} | set(D_HEIGHTS)))
# Accepted data-axes aspect (width / height) by panel kind; README section 2.4.
AXES_ASPECT = {'data': (1.4, 1.6), 'strip': (2.2, 4.7)}


class ThesisFigureWarning(UserWarning):
    """Raised (as a warning) when a figure would violate the 3.0 specification."""


# Perceptually calibrated semantic palette (Okabe-Ito & Paul Tol aligned)
COLORS = {
    'method': '#0072B2',      # Primary method (Deep Ocean Blue)
    'reference': '#3D4650',   # True value / centralized reference / threshold (Deep Slate)
    'comparison': '#D55E00',  # Baseline 1 / primary ablation (Vermilion Orange)
    'third': '#009E73',       # Baseline 2 (Bluish Green)
    'fourth': '#CC79A7',      # Baseline 3 (Reddish Purple)
    'fifth': '#E69F00',       # Baseline 4 / auxiliary (Amber Gold)
    'sixth': '#56B4E9',       # Baseline 5 / light variant (Sky Blue)
    'failure': '#B64242',     # Violation / deadlock / alert (Brick Red)
    'ink': '#1E242B',         # Primary text ink (never use saturated series hue for text)
    'spine': '#2B303A',       # Axes spine & tick ink
    'edge': '#94A3B8',        # Background communication graph edges (Muted Cool Slate)
    'grid': '#E5E9F0',        # Recessive solid hairline grid
    'light': '#F1F5F9',       # Subtle background fill / band wash
}

# Fixed 8-node categorical palette for multi-robot trajectories / states
NODE_PALETTE = [
    '#0072B2', '#D55E00', '#009E73', '#CC79A7',
    '#E69F00', '#56B4E9', '#332288', '#882255',
]

STYLES = ['-', '--', '-.', ':']

FIEDLER = LinearSegmentedColormap.from_list(
    'thesis_fiedler',
    [COLORS['method'], '#F8FAFC', COLORS['comparison']],
)


def apply_style():
    """Configure Matplotlib rcParams for 150 mm print-ready dissertation figures."""
    for family in ('Times New Roman', 'SimSun'):
        font_manager.findfont(family, fallback_to_default=False)

    mpl.rcParams.update({
        # Typography: Times New Roman for ASCII/Latin/digits, SimSun fallback for CJK
        'font.family': ['Times New Roman', 'SimSun'],
        'font.size': 9.5,
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
        'text.color': COLORS['ink'],
        'axes.labelcolor': COLORS['ink'],
        'xtick.color': COLORS['spine'],
        'ytick.color': COLORS['spine'],

        # Font sizes matched to 1:1 150 mm print canvas
        'axes.labelsize': 9.5,
        'axes.titlesize': 9.8,
        'xtick.labelsize': 8.8,
        'ytick.labelsize': 8.8,
        'legend.fontsize': 8.5,

        # Legend: subtle white backing shield so gridlines never cut through text
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': 'none',
        'legend.framealpha': 0.90,
        'legend.handlelength': 1.8,
        'legend.handletextpad': 0.45,
        'legend.columnspacing': 1.1,
        'legend.borderpad': 0.3,

        # Spines & inward precision ticks (IEEE / SciencePlots convention)
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': COLORS['spine'],
        'axes.linewidth': 0.65,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.2,
        'ytick.major.size': 3.2,
        'xtick.minor.size': 1.8,
        'ytick.minor.size': 1.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.45,
        'ytick.minor.width': 0.45,
        'xtick.major.pad': 3.5,
        'ytick.major.pad': 3.5,

        # Lines & markers: smooth rounded caps/joins
        'lines.linewidth': 1.6,
        'lines.markersize': 4.5,
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
        'lines.dash_capstyle': 'round',

        # Recessive hairline grid
        'grid.color': COLORS['grid'],
        'grid.linewidth': 0.45,
        'grid.linestyle': '-',
        'grid.alpha': 0.85,
        'axes.axisbelow': True,

        # Figure & export settings (Type 42 TrueType embedding for PDF)
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'axes.prop_cycle': mpl.cycler(
            color=[COLORS[k] for k in ('method', 'comparison', 'third', 'fourth', 'reference')]
        ),
    })


P2E_WIDTH_RATIOS = {'both': [1, 1], 'left': [44, 56], 'right': [56, 44]}


def new_figure(layout, height_mm=None, geometry='both', blank=False, **subplot_kw):
    """Create a canvas at a 3.0 tier size and return ``(fig, axes)``.

    ``layout`` is a key of LAYOUTS, or 'D' for a conceptual diagram whose
    ``height_mm`` must be one of D_HEIGHTS. 'P3' returns ``[ax_left, ax_top, ax_bottom]``
    (left panel about 46 % of the width); 'V2'/'V3' share the x axis by default.
    For 'P2E', ``geometry`` says which panel holds the equal-aspect trajectory/geometry
    ('left', 'right' or 'both'); with one geometry panel the data panel gets the wider
    column so that it keeps a ~1.4:1 axes box.
    The layout code is remembered and written into the PDF by save_figure().
    """
    import matplotlib.pyplot as plt
    apply_style()

    if layout == 'D':
        if height_mm not in D_HEIGHTS:
            raise ValueError(f"layout='D' 需要 height_mm 取 {D_HEIGHTS} 之一，收到 {height_mm!r}")
        width, height, nrows, ncols = TEXT_WIDTH_MM, height_mm, 1, 1
    elif layout in LAYOUTS:
        if height_mm is not None:
            raise ValueError(f'{layout} 的高度固定为 {LAYOUTS[layout][1]} mm；需要其他高度请用 layout="D"')
        width, height, nrows, ncols, _ = LAYOUTS[layout]
    else:
        raise ValueError(f'未知构图代号 {layout!r}；可选 {sorted(LAYOUTS)} 或 "D"')

    fig = plt.figure(figsize=(width * MM, height * MM), layout='constrained')
    if blank:  # caller builds its own GridSpec (e.g. render frames + data panels); layout code is still recorded
        fig._thesis_layout = layout
        return fig, None
    if layout == 'P3':
        gs = fig.add_gridspec(2, 2, width_ratios=[46, 54])
        axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    else:
        if layout in ('V2', 'V3'):
            subplot_kw.setdefault('sharex', True)
        if layout == 'P2E':
            if geometry not in P2E_WIDTH_RATIOS:
                raise ValueError(f"P2E 的 geometry 取 {sorted(P2E_WIDTH_RATIOS)} 之一，收到 {geometry!r}")
            subplot_kw.setdefault('gridspec_kw', {'width_ratios': P2E_WIDTH_RATIOS[geometry]})
        axes = fig.subplots(nrows, ncols, **subplot_kw)
    fig._thesis_layout = layout
    return fig, axes


def axes_aspects(fig):
    """Return [(axes, width_mm, height_mm, width/height)] for visible rectilinear axes after layout."""
    fig.canvas.draw()
    w_fig, h_fig = (float(v) * 25.4 for v in fig.get_size_inches())
    out = []
    for ax in fig.axes:
        if ax.name != 'rectilinear' or not ax.axison:
            continue
        p = ax.get_position()
        w, h = p.width * w_fig, p.height * h_fig
        out.append((ax, w, h, w / h if h else float('inf')))
    return out


def _check_figure_spec(fig, layout):
    """Warn about canvas sizes and text sizes that violate the 3.0 specification."""
    w_mm, h_mm = (float(v) * 25.4 for v in fig.get_size_inches())
    problems = []
    if min(abs(w_mm - TEXT_WIDTH_MM), abs(w_mm - NARROW_WIDTH_MM)) > 0.5:
        problems.append(f'画布宽 {w_mm:.1f} mm，只允许 150 mm（width=\\textwidth）或 105 mm（原尺寸插入）')
    if layout in LAYOUTS:
        nominal = LAYOUTS[layout][1]
        if abs(h_mm - nominal) > 0.5:
            problems.append(f'{layout} 的画布高应为 {nominal} mm，实际 {h_mm:.1f} mm')
    elif layout == 'D':
        if min(abs(h_mm - h) for h in D_HEIGHTS) > 0.5:
            problems.append(f'D 档画布高应取 {D_HEIGHTS} 之一，实际 {h_mm:.1f} mm')
    else:
        problems.append('未声明构图代号：请用 new_figure(layout) 建图，或 save_figure(..., layout=...)')
        if min(abs(h_mm - h) for h in TIER_HEIGHTS) > HEIGHT_TOLERANCE_MM:
            problems.append(f'画布高 {h_mm:.1f} mm 不在任何档位 {TIER_HEIGHTS} 的 ±{HEIGHT_TOLERANCE_MM:g} mm 内')
    small = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        size = float(artist.get_fontsize())
        if size < MIN_FONT_PT - 1e-6:
            small.append(f'{size:.1f} pt「{artist.get_text()[:24]}」')
    if small:
        problems.append(f'{len(small)} 处文字低于 {MIN_FONT_PT} pt：' + '；'.join(small[:5]))
    # Data-axes shape: equal-aspect (trajectory/geometry) axes, axis-off panels and insets are exempt.
    lo, hi = AXES_ASPECT['data'][0] - 0.05, AXES_ASPECT['strip'][1] + 0.1
    for ax, w, h, a in axes_aspects(fig):
        if ax.get_aspect() != 'auto' or w < 25:
            continue
        if a < lo or a > hi:
            kind = '偏方' if a < lo else '过扁'
            problems.append(f'坐标框 {w:.0f}×{h:.0f} mm，宽高比 {a:.2f}（{kind}；数据曲线宜 1.4–1.6，时间序列条 2.2–4.7）'
                            f'「{ax.get_title(loc="left")[:16]}」')
    for msg in problems:
        warnings.warn(msg, ThesisFigureWarning, stacklevel=3)
    return problems


def halo_stroke(lw=2.4, color='white', alpha=0.90):
    """Return a white stroke path-effect list that separates text/lines from background clutter."""
    return [pe.withStroke(linewidth=lw, foreground=color, alpha=alpha)]


def panel(ax, tag, title, pad=6.5):
    """Set a left-aligned subplot header: '(a) Title' with Times New Roman tag and SimSun title."""
    clean_tag = str(tag).strip('() ')
    ax.set_title(f'({clean_tag}) {title}', loc='left', pad=pad, color=COLORS['ink'])


def grid(ax, minor=False):
    """Enable recessive major gridlines and automatic inward minor ticks on continuous axes."""
    ax.grid(True, which='major', color=COLORS['grid'], linewidth=0.45, alpha=0.85)
    try:
        ax.minorticks_on()
        if isinstance(ax.xaxis.get_major_locator(), mpl.ticker.FixedLocator):
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        if isinstance(ax.yaxis.get_major_locator(), mpl.ticker.FixedLocator):
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    except Exception:
        pass
    if minor:
        ax.grid(True, which='minor', color=COLORS['grid'], linewidth=0.28, alpha=0.5)


def halo_text(ax, x, y, s, halo_lw=2.5, **kwargs):
    """Place an in-canvas text annotation with an automatic white halo ring."""
    kwargs.setdefault('color', COLORS['ink'])
    kwargs.setdefault('fontsize', 8.5)
    txt = ax.text(x, y, s, **kwargs)
    if 'path_effects' not in kwargs and 'bbox' not in kwargs:
        txt.set_path_effects(halo_stroke(lw=halo_lw))
    return txt


def halo_line(ax, x, y, halo_lw=3.0, **kwargs):
    """Plot a primary curve with a subtle white outer stroke so it stands out at crossings."""
    kwargs.setdefault('color', COLORS['method'])
    kwargs.setdefault('lw', 1.75)
    kwargs.setdefault('zorder', 4)
    lines = ax.plot(x, y, **kwargs)
    for ln in lines:
        ln.set_path_effects(halo_stroke(lw=halo_lw, color='white', alpha=0.85))
    return lines


def add_trajectory_arrows(ax, traj_xy, color, fractions=(0.48,), size=8.5, zorder=3.5):
    """Add clean tangential direction arrow(s) along a 2D robot trajectory (T, 2)."""
    pts = np.asarray(traj_xy, dtype=float)
    if pts.ndim != 2 or len(pts) < 4:
        return
    diffs = np.diff(pts, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    total_len = float(np.sum(seg_len))
    if total_len < 1e-3:
        return
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    for frac in fractions:
        target = float(frac) * total_len
        idx = int(np.searchsorted(cum_len, target))
        idx = max(1, min(idx, len(pts) - 1))
        # Find a nearby point with non-negligible displacement for stable tangent direction
        p0 = pts[max(0, idx - 1)]
        p1 = pts[idx]
        if np.linalg.norm(p1 - p0) < 1e-4 and idx + 1 < len(pts):
            p1 = pts[idx + 1]
        if np.linalg.norm(p1 - p0) < 1e-4:
            continue
        arrow = FancyArrowPatch(
            posA=tuple(p0),
            posB=tuple(p1),
            arrowstyle='-|>,head_length=0.48,head_width=0.30',
            color=color,
            lw=0.0,
            mutation_scale=size,
            zorder=zorder,
        )
        arrow.set_path_effects(halo_stroke(lw=1.6, color='white', alpha=0.85))
        ax.add_patch(arrow)


def draw_robot_nodes(
    ax,
    start_xy=None,
    end_xy=None,
    goal_xy=None,
    colors=None,
    ms_start=4.4,
    ms_end=5.2,
    ms_goal=4.8,
    label_nodes=False,
    label_offset=(0.0, 0.14),
):
    """Draw multi-robot start (open circle), end (filled circle with white ring), and goal markers."""
    n = 0
    for arr in (start_xy, end_xy, goal_xy):
        if arr is not None:
            n = len(arr)
            break
    if n == 0:
        return
    if colors is None:
        colors = [NODE_PALETTE[i % len(NODE_PALETTE)] for i in range(n)]
    elif isinstance(colors, str):
        colors = [colors] * n

    for i in range(n):
        c = colors[i]
        if goal_xy is not None:
            gx, gy = goal_xy[i]
            mk_g, = ax.plot(
                gx, gy, marker='x', ms=ms_goal, mew=1.25,
                color=c, alpha=0.85, zorder=2.5,
            )
            mk_g.set_path_effects(halo_stroke(lw=2.2, color='white', alpha=0.85))
        if start_xy is not None:
            sx, sy = start_xy[i]
            ax.plot(
                sx, sy, marker='o', ms=ms_start,
                mfc='white', mec=c, mew=1.15, zorder=4.0,
            )
        if end_xy is not None:
            ex, ey = end_xy[i]
            ax.plot(
                ex, ey, marker='o', ms=ms_end,
                mfc=c, mec='white', mew=1.1, zorder=5.0,
            )
        if label_nodes:
            ref = end_xy[i] if end_xy is not None else start_xy[i]
            halo_text(
                ax,
                ref[0] + label_offset[0],
                ref[1] + label_offset[1],
                str(i + 1),
                fontsize=8.2,
                ha='center',
                va='center',
                zorder=5.5,
            )


def shared_top_legend(fig, handles, labels, ncol=None, bbox_to_anchor=None, fontsize=8.8):
    """Place a clean, borderless horizontal legend bar centered above all subplots.

    3.0: by default the legend is placed with ``loc='outside upper center'``, so that
    constrained layout reserves room for it inside the fixed canvas. (The 2.0 default
    anchored it at y=1.02, i.e. outside the canvas, where an exact-size save clips it.)
    Pass ``bbox_to_anchor`` to keep the old behaviour.
    """
    if ncol is None:
        ncol = min(len(labels), 4)
    if bbox_to_anchor is None:
        loc, extra = 'outside upper center', {}
    else:
        loc, extra = 'lower center', {'bbox_to_anchor': bbox_to_anchor}
    return fig.legend(
        handles,
        labels,
        loc=loc,
        ncol=ncol,
        **extra,
        fontsize=fontsize,
        frameon=False,
        handlelength=1.9,
        handletextpad=0.45,
        columnspacing=1.3,
    )


def _has_cjk(s: str) -> bool:
    return any('一' <= ch <= '鿿' for ch in s)


def _normalize_thesis_label(txt: str) -> str:
    """Normalize legacy terminology, code-names, and unit formats to the 2.0 specification."""
    if not txt:
        return txt
    # 1. Normalize full-width parentheses
    txt = txt.replace('（', ' (').replace('）', ')')
    while '  (' in txt:
        txt = txt.replace('  (', ' (')

    # 2. Remove redundant '(无量纲)'
    txt = txt.replace(' (无量纲)', '').replace('(无量纲)', '')

    # 3. Replace internal literature code-names and '种子' literal translations
    replacements = [
        ('Tan-TAC 2025 离散流', 'Tan 等 (2025) 离散流'),
        ('Tan-TAC 2025 输入', 'Tan 等 (2025) 输入'),
        ('Tan-TAC 2025', 'Tan 等 (2025)'),
        ('Liu-Auto 2026 原式', 'Liu 等 (2026) 原式'),
        ('查询加速 Liu-2026', 'Liu 等 (2026) 谱加速'),
        ('同一种子下的估计误差', '同一初值下的估计误差'),
        ('同中心化种子', '同中心化初值'),
        ('本地结清与全局种子的累计次数', '本地结清与全网回退的累计次数'),
        ('(全局种子次数)', '(全网回退次数)'),
        ('全局种子', '全网回退'),
        ('位置 x [m]', r'水平位置 $x$ ($\mathrm{m}$)'),
        ('位置 y [m]', r'纵向位置 $y$ ($\mathrm{m}$)'),
        ('局部 x (m)', r'局部水平位置 $x$ ($\mathrm{m}$)'),
        ('局部 y (m)', r'局部纵向位置 $y$ ($\mathrm{m}$)'),
        ('x (m)', r'水平位置 $x$ ($\mathrm{m}$)'),
        ('y (m)', r'纵向位置 $y$ ($\mathrm{m}$)'),
        ('模拟时间 [s]', r'时间 $t$ ($\mathrm{s}$)'),
        ('物理时间 (s)', r'物理时间 $t$ ($\mathrm{s}$)'),
        ('时间 / s', r'时间 $t$ ($\mathrm{s}$)'),
        ('距离 / m', r'距离 ($\mathrm{m}$)'),
        ('构型样本 k', r'构型样本 $k$'),
        ('谱值序号 q', r'特征值序号 $q$'),
    ]
    for old, new in replacements:
        if old in txt:
            txt = txt.replace(old, new)

    # 4. Standardize trailing slash units on axis labels (e.g. '$p_x$ / m' -> '$p_x$ ($\mathrm{m}$)')
    txt = re.sub(r'\s*/\s*\(m/s\)$', lambda _: r' ($\mathrm{m/s}$)', txt)
    txt = re.sub(r'\s*\(1/s\)$', lambda _: r' ($\mathrm{s}^{-1}$)', txt)
    txt = re.sub(r'(?<![A-Za-z])\s+/\s*m$', lambda _: r' ($\mathrm{m}$)', txt)
    txt = re.sub(r'(?<![A-Za-z])\s+/\s*s$', lambda _: r' ($\mathrm{s}$)', txt)
    txt = re.sub(r'(?<![A-Za-z])\s+/\s*deg$', lambda _: r' ($\mathrm{deg}$)', txt)
    return txt.strip()


def _wrap_ascii_in_mathtext(seg: str) -> str:
    """Wrap ASCII words/numbers/units outside $...$ into $\\mathrm{...}$ so they render in STIX/Times."""
    def _repl(m):
        token = m.group(0)
        if not token.strip():
            return token
        lead = len(token) - len(token.lstrip(' '))
        trail = len(token) - len(token.rstrip(' '))
        core = token.strip(' ')
        if not core:
            return token
        inner = core.replace('%', r'\%').replace(' ', r'\ ')
        return (' ' * lead) + f'$\\mathrm{{{inner}}}$' + (' ' * trail)

    return re.sub(r'[A-Za-z0-9()/\-+.,:=%\[\]]+(?:[ \t]+[A-Za-z0-9()/\-+.,:=%\[\]]+)*', _repl, seg)


def save_figure(fig, stem, dpi=300, layout=None):
    """Finalize typography/halos and export both vector PDF and high-DPI PNG.

    The canvas is saved at its exact size (never ``bbox_inches='tight'``), so the
    figure is inserted 1:1. ``layout`` defaults to the code given to new_figure();
    it is written into the PDF keywords as ``thesis-layout=<code>`` for check_figures.py.
    """
    stem = Path(stem)
    layout = layout or getattr(fig, '_thesis_layout', None)
    stem.parent.mkdir(parents=True, exist_ok=True)

    # 1. Ensure inward ticks on all standard 2D axes (and suppress minor ticks on discrete FixedLocator axes)
    for ax in fig.axes:
        if ax.name == 'rectilinear':
            if isinstance(ax.xaxis.get_major_locator(), mpl.ticker.FixedLocator):
                ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
            if isinstance(ax.yaxis.get_major_locator(), mpl.ticker.FixedLocator):
                ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.tick_params(
                axis='both', which='major', direction='in',
                length=3.2, width=0.6, colors=COLORS['spine'],
            )
            ax.tick_params(
                axis='both', which='minor', direction='in',
                length=1.8, width=0.45, colors=COLORS['spine'],
            )
            for sp in ('bottom', 'left'):
                if sp in ax.spines:
                    ax.spines[sp].set_color(COLORS['spine'])
                    ax.spines[sp].set_linewidth(0.65)

    # 2. Protect in-axes annotations and legends from gridline/curve collisions
    for ax in fig.axes:
        if ax.name == 'rectilinear' and ax.axison:
            for txt_artist in ax.texts:
                c_val = str(txt_artist.get_color()).lower()
                if c_val in ('white', 'w', '#ffffff', '#fff'):
                    continue
                if not txt_artist.get_path_effects() and txt_artist.get_bbox_patch() is None:
                    txt_artist.set_path_effects(halo_stroke(lw=2.2, color='white', alpha=0.88))
        leg = ax.get_legend()
        if leg is not None:
            frame = leg.get_frame()
            if frame is not None and not leg.get_frame_on():
                leg.set_frame_on(True)
                frame.set_facecolor('white')
                frame.set_edgecolor('none')
                frame.set_alpha(0.90)

    # 3. Dual-font perfection & terminology normalization across all Text artists:
    for artist in fig.findobj(match=Text):
        txt = artist.get_text()
        if not txt:
            continue
        txt_norm = _normalize_thesis_label(txt)
        if txt_norm != txt:
            txt = txt_norm
            artist.set_text(txt)
        if _has_cjk(txt) and '$' in txt:
            new_lines = []
            for line in txt.split('\n'):
                parts = line.split('$')
                if len(parts) % 2 == 1:
                    for idx in range(0, len(parts), 2):
                        parts[idx] = _wrap_ascii_in_mathtext(parts[idx])
                    line = '$'.join(parts).replace('$$', '')
                new_lines.append(line)
            txt_new = '\n'.join(new_lines)
            if txt_new != txt:
                artist.set_text(txt_new)
            artist.set_fontfamily('SimSun')
        else:
            artist.set_fontfamily(['Times New Roman', 'SimSun'])

    # 4. Specification check (warnings only; the export itself is unchanged)
    _check_figure_spec(fig, layout)

    metadata = {'Keywords': f'thesis-layout={layout}'} if layout else None
    fig.savefig(stem.with_suffix('.pdf'), metadata=metadata)
    fig.savefig(stem.with_suffix('.png'), dpi=dpi)


def _robot_scene():
    """Synthetic six-robot scene shared by the previews (demonstration data, not results)."""
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + 0.25
    start_xy = np.column_stack([1.15 * np.cos(angles), 0.78 * np.sin(angles)])
    end_xy = np.column_stack([0.65 * np.cos(angles + 0.55), 0.48 * np.sin(angles + 0.55)])
    goal_xy = np.column_stack([0.55 * np.cos(angles + 0.65), 0.42 * np.sin(angles + 0.65)])
    s_grid = np.linspace(0, 1, 60)[:, None]
    trajs = []
    for i in range(6):
        ctrl = 0.5 * (start_xy[i] + end_xy[i]) + np.array([-0.18 * np.sin(angles[i]), 0.18 * np.cos(angles[i])])
        trajs.append((1 - s_grid) ** 2 * start_xy[i] + 2 * (1 - s_grid) * s_grid * ctrl + s_grid ** 2 * end_xy[i])
    return start_xy, end_xy, goal_xy, trajs


def _render_style_preview():
    """Generate style_preview (P4, four data panels) and style_preview_p3 (P3, trajectory + strips).

    All curves are synthetic demonstration data.
    """
    import matplotlib.pyplot as plt

    apply_style()
    here = Path(__file__).resolve().parent
    t = np.linspace(0, 4.0, 200)

    # ---- Preview 1: P4, four ordinary data panels --------------------------------------
    fig, axs = new_figure('P4')

    ax = axs[0, 0]
    halo_line(ax, t, np.exp(-0.55 * t), color=COLORS['method'], ls='-', label='本文方法')
    ax.plot(t, np.exp(-0.95 * t), color=COLORS['comparison'], ls='--', lw=1.35, label='对照方法 1')
    ax.plot(t, np.exp(-1.35 * t), color=COLORS['third'], ls='-.', lw=1.35, label='对照方法 2')
    ax.axhline(0.18, color=COLORS['reference'], ls=':', lw=1.15, label=r'阈值 $\lambda_{\mathrm{req}}$')
    panel(ax, 'a', '颜色与线型双重编码')
    ax.set(xlabel=r'算法时间 $t$ ($\mathrm{s}$)', ylabel=r'代数连通度 $\lambda_2$', xlim=(0, 4.0), ylim=(0, 1.08))
    ax.legend(loc='upper right')
    grid(ax)

    ax = axs[0, 1]
    ell = np.arange(0, 41)
    med = 0.8 * (0.72 ** ell) + 1e-6
    ax.fill_between(ell, 0.45 * med, 1.85 * med, color=COLORS['method'], alpha=0.14, lw=0, label='15%–85% 分位带')
    halo_line(ax, ell, med, color=COLORS['method'], ls='-', label='本文方法 (中位数)')
    ax.plot(ell, 0.8 * (0.93 ** ell), color=COLORS['comparison'], ls='--', lw=1.35, label='对照方法 1')
    halo_text(ax, 20, 2e-3, r'指数收敛率 $\rho^\ell$')
    ax.set_yscale('log')
    ax.set(xlim=(0, 40), ylim=(1e-6, 2.0), xlabel=r'同步通信轮数 $\ell$', ylabel=r'目标函数间隙 $J_k-J_k^\star$')
    ax.legend(loc='lower left')
    panel(ax, 'b', '对数坐标与分位带')
    grid(ax)

    ax = axs[1, 0]
    n_robots = np.array([8, 12, 16, 20, 25, 30])
    for key, mk, ls, rate, lab in (('method', 'o', '-', [100, 100, 100, 97, 94, 91], '本文方法'),
                                   ('comparison', 's', '--', [100, 92, 81, 70, 58, 47], '对照方法 1'),
                                   ('third', '^', '-.', [96, 85, 72, 61, 50, 38], '对照方法 2')):
        ax.plot(n_robots, rate, color=COLORS[key], ls=ls, lw=1.35, marker=mk, ms=4.6,
                mfc='white' if key != 'method' else COLORS[key], mec=COLORS[key], mew=1.1, label=lab)
    ax.set(xlabel=r'机器人数量 $N$', ylabel=r'达标率 ($\%$)', xlim=(6, 32), ylim=(30, 104))
    ax.legend(loc='lower left')
    panel(ax, 'c', '离散点用不同形状标识')
    grid(ax)

    ax = axs[1, 1]
    d_method = 0.42 + 0.55 * np.exp(-1.2 * t) + 0.03 * np.sin(3 * t) * np.exp(-0.4 * t)
    d_base = 0.30 + 0.70 * np.exp(-1.6 * t) + 0.12 * np.exp(-((t - 2.2) / 0.6) ** 2)
    ax.axvspan(0.95, 1.75, color=COLORS['failure'], alpha=0.10, lw=0, label='对照方法违约区间')
    halo_line(ax, t, d_method, color=COLORS['method'], ls='-', label='本文方法')
    ax.plot(t, d_base, color=COLORS['comparison'], ls='--', lw=1.35, label='对照方法 1')
    ax.axhline(0.40, color=COLORS['reference'], ls=':', lw=1.15, label=r'安全距离 $d_{\mathrm{safe}}$')
    ax.set(xlabel=r'物理时间 $t$ ($\mathrm{s}$)', ylabel=r'最小机间距离 ($\mathrm{m}$)', xlim=(0, 4.0), ylim=(0.2, 1.05))
    ax.legend(loc='upper right')
    panel(ax, 'd', '阈值与违约区间')
    grid(ax)

    save_figure(fig, here / 'style_preview')
    plt.close(fig)

    # ---- Preview 2: P3, equal-aspect trajectory (left) + two shared-time strips (right) --
    fig, (ax_l, ax_t, ax_b) = new_figure('P3')
    start_xy, end_xy, goal_xy, trajs = _robot_scene()
    for i in range(6):
        for j in range(i + 1, 6):
            d = float(np.linalg.norm(end_xy[i] - end_xy[j]))
            if d < 1.05:
                ax_l.plot(*zip(end_xy[i], end_xy[j]), color=COLORS['edge'], lw=0.8 + 0.7 * (1.05 - d), alpha=0.45, zorder=2)
    for i, traj in enumerate(trajs):
        ax_l.plot(traj[:, 0], traj[:, 1], color=NODE_PALETTE[i], lw=1.4, zorder=3)
        add_trajectory_arrows(ax_l, traj, color=NODE_PALETTE[i], fractions=(0.52,), size=8.0)
    draw_robot_nodes(ax_l, start_xy=start_xy, end_xy=end_xy, goal_xy=goal_xy, label_nodes=True, label_offset=(0.0, 0.13))
    ax_l.set_aspect('equal')
    # P3 left column is about 55 x 62 mm: an equal-aspect domain should be near-square (<= 1.1:1)
    ax_l.set(xlim=(-1.35, 1.35), ylim=(-1.25, 1.25), xlabel=r'水平位置 $x$ ($\mathrm{m}$)', ylabel=r'纵向位置 $y$ ($\mathrm{m}$)')
    panel(ax_l, 'a', '六机轨迹与终态拓扑')
    grid(ax_l)

    lam = 0.25 + 0.55 * np.exp(-0.9 * t)
    halo_line(ax_t, t, lam, color=COLORS['method'], ls='-', label='本文方法')
    ax_t.axhline(0.18, color=COLORS['reference'], ls=':', lw=1.15, label=r'阈值 $\lambda_{\mathrm{req}}$')
    ax_t.set(xlim=(0, 4.0), ylim=(0, 0.9), ylabel=r'$\lambda_2$')
    ax_t.legend(loc='upper right', ncol=2)
    panel(ax_t, 'b', '代数连通度')
    grid(ax_t)

    d_min = 0.42 + 0.55 * np.exp(-1.2 * t)
    halo_line(ax_b, t, d_min, color=COLORS['method'], ls='-')
    ax_b.axhline(0.40, color=COLORS['reference'], ls=':', lw=1.15)
    halo_text(ax_b, 3.0, 0.46, r'$d_{\mathrm{safe}}$')
    ax_b.set(xlim=(0, 4.0), ylim=(0.3, 1.05), xlabel=r'物理时间 $t$ ($\mathrm{s}$)', ylabel=r'$\min\,d_{ij}$ ($\mathrm{m}$)')
    panel(ax_b, 'c', '最小机间距离')
    grid(ax_b)

    save_figure(fig, here / 'style_preview_p3')
    plt.close(fig)


if __name__ == '__main__':
    _render_style_preview()

