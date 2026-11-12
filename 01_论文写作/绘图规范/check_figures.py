#!/usr/bin/env python3
"""图件规范 3.0 自动验收（README.md 第十节）。

静态检查（默认）：按 整篇论文/chapters.json 读取各章 content.tex 实际收录的 figure 环境，
检查位置说明符、插入缩放、画布尺寸与构图档位、最终字号与字体、内嵌位图分辨率、白边、
题注与标签，以及各章 packages.tex / settings.tex 的浮动设置。
页面检查（--pdf）：图与首次引用的页距、同页图数、编译日志中图表引起的 Overfull；
再加 --pages 时渲染正文页，检查页底空白与孤立浮动页。

用法（在任意目录运行均可）：
  python check_figures.py
  python check_figures.py --pdf ../整篇论文/build/main.pdf --pages --out qa/<日期>_<说明>
  python check_figures.py --strict          # 有 ERROR 时退出码为 1
依赖：Python 3.10+、Matplotlib（读取 thesis_plot_style 的档位常量）、TeX Live 的
pdfinfo / pdffonts / pdftotext / pdftoppm、poppler 的 pdfimages（可缺省）；
白边与 --pages 需要 Pillow。本脚本只读，不修改任何论文源文件和图件。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WRITING = HERE.parent  # 01_论文写作
sys.path.insert(0, str(HERE))
from thesis_plot_style import (  # noqa: E402  single source of the tier numbers
    ABS_MAX_HEIGHT_MM, D_HEIGHTS, HEIGHT_TOLERANCE_MM, LAYOUTS, MAX_HEIGHT_MM,
    MIN_FONT_PT, NARROW_WIDTH_MM, TEXT_WIDTH_MM, TIER_HEIGHTS,
)

SPEC_VERSION = '3.0'
PT_PER_MM = 72 / 25.4
NONTIER_LAYOUTS = {
    '1x3-data': '一排 3 个带数据坐标轴的子图',
    '2x3-data': '2×3 排布且含带数据坐标轴的子图',
    '3+2': '首排 3 个带坐标轴的子图',
}
BANNED_FONTS = re.compile(r'YaHei|SimHei|DejaVu|Arial|Helvetica|Calibri|Segoe|NotoSans|SourceHanSans|Heiti|Dengxian', re.I)
SCRIPT_FACTORS = (0.7, 0.49)  # mathtext sub/superscript and second-level scripts
CODES = {
    'F01': '位置说明符', 'F02': '插入缩放', 'F03': '画布宽度', 'F04': '画布高度与档位',
    'F05': '每排子图数', 'F06': '字号下限', 'F07': '字体', 'F08': '位图分辨率',
    'F09': '白边与偏心', 'F10': '题注', 'F11': '标签', 'F12': '未声明构图',
    'F13': 'TikZ', 'F14': '同章同档高度不一',
    'C01': 'placeins[section]', 'C02': '章内浮动参数',
    'P01': '图与首次引用的页距', 'P02': '页底空白', 'P03': '页顶空白/孤立浮动页',
    'P04': '图表溢出版心', 'P05': '同页图数',
}


# ----------------------------------------------------------------------------- utilities
def run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        return ''
    return r.stdout.decode('utf-8', 'replace')


class Report:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, level: str, code: str, where: str, message: str):
        self.items.append({'level': level, 'code': code, 'where': where, 'message': message})

    def count(self, level: str) -> int:
        return sum(1 for i in self.items if i['level'] == level)


def read_group(s: str, i: int, open_ch: str = '{', close_ch: str = '}'):
    """Return (content, index after the closing delimiter) for a balanced group at s[i]."""
    if i >= len(s) or s[i] != open_ch:
        return None, i
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == '\\':
            j += 2
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return None, i


def text_units(tex: str) -> float:
    """Approximate printed width in CJK-character units (ASCII counts as half)."""
    s = re.sub(r'\$([^$]*)\$', lambda m: 'x' * max(1, len(m.group(1)) // 3), tex)
    s = re.sub(r'\\[a-zA-Z]+\*?', '', s)
    s = re.sub(r'[{}~]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return sum(1.0 if ord(ch) > 0x2E80 else 0.5 for ch in s)


# ----------------------------------------------------------------------------- TeX sources
SPECIAL_COMMENTS = [
    (re.compile(r'%\s*layout\s*=\s*([A-Za-z0-9+\-]+)'), lambda m: '\\THESISLAYOUT{' + m.group(1) + '}'),
    (re.compile(r'%\s*figure-exception\s*:\s*(.*)$'), lambda m: '\\THESISEXCEPTION{' + m.group(1).strip() + '}'),
]


def strip_comments(line: str) -> str:
    for pat, rep in SPECIAL_COMMENTS:
        line = pat.sub(rep, line)
    return re.sub(r'(?<!\\)%.*', '', line)


def expand_tex(path: Path, base: Path, seen: set) -> list[tuple[Path, int, str]]:
    if not path.suffix:
        path = path.with_suffix('.tex')
    if not path.exists() or path in seen:
        return []
    seen.add(path)
    out = []
    for i, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = strip_comments(raw)
        m = re.search(r'\\(?:input|include)\{([^}]+)\}', line)
        if m:
            out.append((path, i, line[:m.start()]))
            out.extend(expand_tex(base / m.group(1), base, seen))
            out.append((path, i, line[m.end():]))
        else:
            out.append((path, i, line))
    return out


def load_chapters() -> list[dict]:
    cfg = json.loads((WRITING / '整篇论文' / 'chapters.json').read_text(encoding='utf-8'))
    return [{'number': c['number'], 'dir': WRITING / c['directory']} for c in cfg]


def parse_figures(ch: dict) -> list[dict]:
    chdir = ch['dir']
    lines = expand_tex(chdir / 'content.tex', chdir, set())
    buf, prov, offs, acc = [], [], [], 0
    for f, ln, t in lines:
        offs.append(acc)
        buf.append(t + '\n')
        prov.append((f, ln))
        acc += len(t) + 1
    text = ''.join(buf)

    def where(pos):
        lo, hi = 0, len(offs) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offs[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        f, ln = prov[lo]
        return f, ln

    figs = []
    for k, m in enumerate(re.finditer(r'\\begin\{(figure\*?)\}(\[[^\]]*\])?(.*?)\\end\{\1\}', text, re.S), 1):
        body = m.group(3)
        src, line = where(m.start())
        fig = {
            'chapter': ch['number'], 'number': f"{ch['number']}.{k}", 'source': src, 'line': line,
            'place': (m.group(2) or '')[1:-1], 'body': body,
            'label': (re.findall(r'\\label\{([^}]+)\}', body) or [''])[0],
            'layout_tex': (re.findall(r'\\THESISLAYOUT\{([^}]+)\}', body) or [None])[0],
            'exception': (re.findall(r'\\THESISEXCEPTION\{([^}]*)\}', body) or [None])[0],
            'tikz': 'tikzpicture' in body,
            'resize': re.findall(r'\\(resizebox|scalebox|adjustbox)', body),
            'graphics': [],
        }
        for g in re.finditer(r'\\includegraphics(\[[^\]]*\])?\{([^}]+)\}', body):
            fig['graphics'].append({'opts': (g.group(1) or '')[1:-1], 'path': g.group(2).strip()})
        cap = re.search(r'\\caption', body)
        fig['short'], fig['caption'] = None, None
        if cap:
            i = cap.end()
            while i < len(body) and body[i].isspace():
                i += 1
            if i < len(body) and body[i] == '[':
                fig['short'], i = read_group(body, i, '[', ']')
            while i < len(body) and body[i].isspace():
                i += 1
            fig['caption'], _ = read_group(body, i)
        figs.append(fig)
    return figs


# ----------------------------------------------------------------------------- PDF internals
def pdf_objects(data: bytes) -> dict:
    objs = {int(m.group(1)): m.group(3) for m in re.finditer(rb'(\d+)\s+(\d+)\s+obj\b(.*?)endobj', data, re.S)}
    extra = {}
    for body in objs.values():
        head, raw = stream_of(body)
        if head is None or raw is None or b'/ObjStm' not in head:
            continue
        n = int(re.search(rb'/N\s+(\d+)', head).group(1))
        first = int(re.search(rb'/First\s+(\d+)', head).group(1))
        nums = list(map(int, raw[:first].split()))
        pairs = [(nums[2 * i], nums[2 * i + 1]) for i in range(n)]
        for i, (on, off) in enumerate(pairs):
            end = pairs[i + 1][1] if i + 1 < n else len(raw) - first
            extra[on] = raw[first + off:first + end]
    objs.update(extra)
    return objs


def stream_of(body: bytes):
    m = re.search(rb'stream\r?\n', body)
    if not m:
        return None, None
    head, raw = body[:m.start()], body[m.end():]
    raw = raw[:raw.rfind(b'endstream')]
    if b'FlateDecode' in head:
        try:
            raw = zlib.decompressobj().decompress(raw)
        except zlib.error:
            return head, None
    return head, raw


TOKEN = re.compile(rb'\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>|\[|\]|/[^\s/\[\]()<>]+|[-+]?\d*\.?\d+|[A-Za-z\'"*]+[0-9]?|<<|>>')
NUMBER = re.compile(rb'[-+]?\d*\.?\d+')


def _mul(a, b):
    return [a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5]]


def _interpret(content, ctm, objs, xobjects, sizes, depth=0):
    """Minimal content-stream interpreter recording the device-space size of each text show."""
    if depth > 5:
        return
    stack, saved, arr = [], [], None
    tm = [1, 0, 0, 1, 0, 0]
    tlm, tf = tm[:], 0.0
    for t in TOKEN.findall(content):
        if t == b'[':
            arr = []
            continue
        if t == b']':
            stack.append(arr)
            arr = None
            continue
        if arr is not None:
            arr.append(t)
            continue
        if NUMBER.fullmatch(t) or t.startswith((b'/', b'(', b'<')):
            stack.append(t)
            continue
        try:
            if t == b'q':
                saved.append(ctm[:])
            elif t == b'Q':
                ctm = saved.pop() if saved else ctm
            elif t == b'cm':
                ctm = _mul([float(x) for x in stack[-6:]], ctm)
            elif t == b'BT':
                tm = [1, 0, 0, 1, 0, 0]
                tlm = tm[:]
            elif t == b'Tm':
                tm = [float(x) for x in stack[-6:]]
                tlm = tm[:]
            elif t in (b'Td', b'TD'):
                tlm = _mul([1, 0, 0, 1, float(stack[-2]), float(stack[-1])], tlm)
                tm = tlm[:]
            elif t == b'Tf':
                tf = float(stack[-1])
            elif t in (b'Tj', b'TJ', b"'", b'"'):
                full = _mul(tm, ctm)
                sizes.append(tf * math.hypot(full[2], full[3]))
            elif t == b'Do':
                ref = xobjects.get(stack[-1])
                head, raw = stream_of(objs.get(ref, b'')) if ref else (None, None)
                if head and raw and b'/Form' in head:
                    mm = re.search(rb'/Matrix\s*\[([^\]]*)\]', head)
                    fm = [float(x) for x in mm.group(1).split()] if mm else [1, 0, 0, 1, 0, 0]
                    sub = dict(xobjects)
                    sub.update({k: int(v) for k, v in re.findall(rb'(/[^\s/<>\[\]]+)\s+(\d+)\s+0\s+R', head)})
                    _interpret(raw, _mul(fm, ctm), objs, sub, sizes, depth + 1)
        except (ValueError, IndexError):
            pass
        stack = []


def pdf_text_sizes(path: Path) -> list[float]:
    objs = pdf_objects(path.read_bytes())
    xobjects = {}
    for body in objs.values():
        for blk in re.findall(rb'/XObject\s*<<(.*?)>>', body, re.S):
            xobjects.update({k: int(v) for k, v in re.findall(rb'(/[^\s/<>\[\]]+)\s+(\d+)\s+0\s+R', blk)})
    sizes = []
    for body in objs.values():
        if not re.search(rb'/Type\s*/Page\b', body):
            continue
        c = re.search(rb'/Contents\s+(\d+)\s+0\s+R', body)
        refs = [int(c.group(1))] if c else [
            int(x) for x in re.findall(rb'(\d+)\s+0\s+R', (re.search(rb'/Contents\s*\[([^\]]*)\]', body) or [b'', b''])[1])]
        for r in refs:
            _, raw = stream_of(objs.get(r, b''))
            if raw:
                _interpret(raw, [1, 0, 0, 1, 0, 0], objs, xobjects, sizes)
        break  # first page only
    return [s for s in sizes if s > 0.5]


def classify_sizes(sizes: list[float]) -> dict:
    """Split final text sizes into base text and math scripts; report base text below the floor."""
    c = Counter(round(s, 1) for s in sizes)
    base = [k for k in c if k >= MIN_FONT_PT - 0.05]
    small = {k: v for k, v in c.items() if k < MIN_FONT_PT - 0.05 and not any(
        abs(k - f * b) <= 0.25 for b in base for f in SCRIPT_FACTORS)}
    n = sum(c.values())
    return {'n': n, 'min': min(c) if c else None, 'median': round(statistics.median(sizes), 1) if sizes else None,
            'base_min': min(base) if base else None, 'small': dict(sorted(small.items())),
            'n_small': sum(small.values()), 'pct_small': round(100 * sum(small.values()) / n) if n else 0}


def pdf_info(path: Path) -> dict:
    out = run(['pdfinfo', str(path)])
    size = re.search(r'Page size:\s+([\d.]+) x ([\d.]+)', out)
    kw = re.search(r'Keywords:\s+(.*)', out)
    layout = None
    if kw:
        m = re.search(r'thesis-layout=([A-Za-z0-9+\-]+)', kw.group(1))
        layout = m.group(1) if m else None
    return {'w_mm': float(size.group(1)) / PT_PER_MM if size else None,
            'h_mm': float(size.group(2)) / PT_PER_MM if size else None, 'layout': layout}


def pdf_fonts(path: Path) -> list[str]:
    names = []
    for line in run(['pdffonts', str(path)]).splitlines()[2:]:
        if line.split():
            names.append(re.sub(r'^[A-Z]{6}\+', '', line.split()[0]))
    return sorted(set(names))


def pdf_images_ppi(path: Path) -> list[float]:
    if not shutil.which('pdfimages'):
        return []
    ppi = []
    for line in run(['pdfimages', '-list', str(path)]).splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 14 and parts[2] == 'image':
            try:
                ppi.append(min(float(parts[12]), float(parts[13])))
            except ValueError:
                pass
    return ppi


def ink_margins(path: Path, scale: float, tmp: Path) -> dict | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    stem = tmp / f'ink_{abs(hash(str(path)))}'
    run(['pdftoppm', '-r', f'{100 * scale:.3f}', '-gray', '-png', '-singlefile', str(path), str(stem)])
    png = Path(str(stem) + '.png')
    if not png.exists():
        return None
    with Image.open(png) as im:
        g = im.convert('L')
        bbox = g.point(lambda v: 255 if v < 235 else 0).getbbox()
        w, h = g.size
    if not bbox:
        return None
    px = 25.4 / 100
    l, t, r, b = bbox
    return {'left': round(l * px, 1), 'right': round((w - r) * px, 1), 'top': round(t * px, 1), 'bottom': round((h - b) * px, 1)}


# ----------------------------------------------------------------------------- figure checks
def nominal_height(layout: str | None, h: float):
    if layout in LAYOUTS:
        return LAYOUTS[layout][1]
    if layout == 'D':
        return min(D_HEIGHTS, key=lambda x: abs(x - h))
    return None


def check_figure(fig: dict, rep: Report, layout_map: dict, tmp: Path) -> dict:
    where = f"图{fig['number']} {fig['label']}".strip()
    src = f"{fig['source'].relative_to(WRITING).as_posix()}:{fig['line']}"
    row = {'number': fig['number'], 'chapter': fig['chapter'], 'label': fig['label'], 'source': src,
           'place': fig['place'], 'tikz': fig['tikz']}

    if fig['place'] != 'htbp':
        rep.add('ERROR', 'F01', where, f"位置说明符为 [{fig['place']}]，应为 [htbp]（{src}）")
    if fig['resize']:
        rep.add('ERROR', 'F02', where, f"figure 内使用 {', '.join(sorted(set(fig['resize'])))} 缩放（{src}）")
    if not fig['short']:
        rep.add('ERROR', 'F10', where, '缺少 \\caption[简明图题] 的方括号短题')
    else:
        u = text_units(fig['short'])
        if u > 30:
            rep.add('WARN', 'F10', where, f'短题约 {u:.0f} 字，宜 ≤ 30 字')
    if fig['caption']:
        lines = math.ceil(text_units(fig['caption']) / 40)
        row['caption_lines_est'] = lines
        if lines > 4:
            rep.add('WARN', 'F10', where, f'完整题注估计 {lines} 行，宜 ≤ 4 行，解释性内容移入正文')
    if fig['label'] and not re.match(rf"fig:ch{fig['chapter']}-", fig['label']):
        rep.add('WARN', 'F11', where, f"标签未用 fig:ch{fig['chapter']}- 前缀（存量标签可不改）")

    layout, layout_src = fig['layout_tex'], 'tex' if fig['layout_tex'] else None

    if fig['tikz']:
        if re.search(r'\\(scriptsize|tiny)\b', fig['body']):
            rep.add('ERROR', 'F13', where, 'TikZ 使用 \\scriptsize/\\tiny（7.5 bp 及以下），低于字号下限')
        n_abs = len(re.findall(r'\bat\s*\(\s*-?[\d.]+\s*,\s*-?[\d.]+\s*\)', fig['body']))
        if n_abs >= 4 and not re.search(r'=\s*[\d.]*\s*(mm|cm|pt)?\s*of\s', fig['body']):
            rep.add('WARN', 'F13', where, f'TikZ 有 {n_abs} 个节点用绝对坐标定位，框内文字变长时易重叠；宜改用 positioning 相对定位')
        if not layout:
            layout, layout_src = layout_map.get(fig['label']), 'map' if fig['label'] in layout_map else None
        row.update({'layout': layout, 'layout_src': layout_src})
        if not layout:
            rep.add('WARN', 'F12', where, '未声明构图代号（TikZ 图请在 figure 内写 % layout=D）')
        return row

    if len(fig['graphics']) != 1:
        rep.add('WARN', 'F05', where, f"一个 figure 内有 {len(fig['graphics'])} 个 \\includegraphics；规范要求子图在画布内拼好")
    if not fig['graphics']:
        return row
    g = fig['graphics'][0]
    # graphics paths are relative to the chapter directory (the one holding content.tex)
    chdir = next(p for p in [fig['source'].parent, *fig['source'].parents] if (p / 'content.tex').exists())
    cand = chdir / g['path']
    files = [cand] if cand.suffix else [cand.with_suffix(e) for e in ('.pdf', '.png', '.jpg', '.jpeg')]
    gfile = next((p for p in files if p.exists()), None)
    row['file'] = g['path']
    if not gfile:
        rep.add('ERROR', 'F03', where, f"找不到图件 {g['path']}")
        return row

    opts = g['opts']
    factor = None
    m = re.search(r'width\s*=\s*([\d.]*)\s*\\(textwidth|linewidth|columnwidth)', opts)
    if m:
        factor = float(m.group(1) or 1)
    m_abs = re.search(r'width\s*=\s*([\d.]+)\s*(mm|cm|pt|bp|in)', opts)
    if re.search(r'\b(height|scale|totalheight)\s*=', opts):
        rep.add('ERROR', 'F02', where, f'\\includegraphics 使用了 [{opts}]，只允许 width=\\textwidth 或不写宽度')
    if factor is not None and abs(factor - 1) > 1e-9:
        rep.add('ERROR', 'F02', where, f'按系数 {factor:g}\\textwidth 插入；应在出图时按档位重出，不在 LaTeX 中缩放')

    if gfile.suffix.lower() == '.pdf':
        info = pdf_info(gfile)
        nat_w, nat_h = info['w_mm'], info['h_mm']
    else:
        try:
            from PIL import Image
            with Image.open(gfile) as im:
                px_w, px_h = im.size
                dpi = (im.info.get('dpi') or (72, 72))[0] or 72
        except Exception:
            rep.add('WARN', 'F08', where, f'无法读取位图 {gfile.name}')
            return row
        nat_w, nat_h, info = px_w / dpi * 25.4, px_h / dpi * 25.4, {'layout': None}
    if factor is not None:
        out_w = factor * TEXT_WIDTH_MM
    elif m_abs:
        unit = {'mm': 1, 'cm': 10, 'pt': 25.4 / 72.27, 'bp': 25.4 / 72, 'in': 25.4}[m_abs.group(2)]
        out_w = float(m_abs.group(1)) * unit
    else:
        out_w = nat_w
    scale = out_w / nat_w
    out_h = nat_h * scale
    row.update({'canvas_mm': [round(nat_w, 1), round(nat_h, 1)], 'scale': round(scale, 3),
                'out_mm': [round(out_w, 1), round(out_h, 1)]})

    if abs(scale - 1) > 0.02 and not (factor is not None and abs(factor - 1) > 1e-9):
        rep.add('ERROR', 'F02', where, f'插入缩放比 {scale:.3f}（画布 {nat_w:.0f} mm 宽被压到 {out_w:.0f} mm），应为 1:1')
    if min(abs(nat_w - TEXT_WIDTH_MM), abs(nat_w - NARROW_WIDTH_MM)) > 1.0:
        level = 'ERROR' if abs(scale - 1) > 0.02 else 'WARN'
        rep.add(level, 'F03', where, f'画布宽 {nat_w:.1f} mm，应为 150 或 105 mm')

    if not layout:
        layout, layout_src = info.get('layout'), 'pdf' if info.get('layout') else None
    if not layout and fig['label'] in layout_map:
        layout, layout_src = layout_map[fig['label']], 'map'
    row.update({'layout': layout, 'layout_src': layout_src})
    if layout in NONTIER_LAYOUTS:
        rep.add('ERROR', 'F05', where, f'{NONTIER_LAYOUTS[layout]}；数据子图每排最多 2 个，改用 P3/V3 或拆图')
    elif layout and layout not in LAYOUTS and layout != 'D':
        rep.add('WARN', 'F12', where, f'未知构图代号 {layout}')
    else:
        nom = nominal_height(layout, out_h)
        if nom is None:
            rep.add('WARN', 'F12', where, '未声明构图代号（new_figure 自动写入；存量图可在 figure 内写 % layout=代号）')
            nearest = min(TIER_HEIGHTS, key=lambda x: abs(x - out_h))
            if abs(nearest - out_h) > HEIGHT_TOLERANCE_MM:
                rep.add('ERROR', 'F04', where, f'成品高 {out_h:.1f} mm 不在任何档位 {TIER_HEIGHTS} 的 ±{HEIGHT_TOLERANCE_MM:g} mm 内')
        else:
            row['nominal_h'] = nom
            d = out_h - nom
            if abs(d) > HEIGHT_TOLERANCE_MM:
                rep.add('ERROR', 'F04', where, f'{layout} 档标称高 {nom} mm，成品 {out_h:.1f} mm（差 {d:+.1f}）')
            elif abs(d) > 0.5:
                rep.add('WARN', 'F04', where, f'{layout} 档标称高 {nom} mm，成品 {out_h:.1f} mm，在存量容差内；修改时回到标称值')
    if out_h > ABS_MAX_HEIGHT_MM:
        rep.add('ERROR', 'F04', where, f'成品高 {out_h:.0f} mm 超过上限 {ABS_MAX_HEIGHT_MM:g} mm')
    elif out_h > MAX_HEIGHT_MM + HEIGHT_TOLERANCE_MM and not fig['exception']:
        rep.add('ERROR', 'F04', where, f'成品高 {out_h:.0f} mm 超过 {MAX_HEIGHT_MM:g} mm，需在 figure 内写 % figure-exception: 理由')

    if gfile.suffix.lower() == '.pdf':
        fs = classify_sizes([s * scale for s in pdf_text_sizes(gfile)])
        row['font'] = fs
        if fs['n_small']:
            sizes = ', '.join(f'{k:g}' for k in fs['small'])
            rep.add('ERROR', 'F06', where, f"{fs['n_small']}/{fs['n']} 处文字（{fs['pct_small']}%）低于 {MIN_FONT_PT} pt：{sizes} pt")
        fonts = pdf_fonts(gfile)
        row['fonts'] = fonts
        bad = [f for f in fonts if BANNED_FONTS.search(f)]
        if bad:
            rep.add('ERROR', 'F07', where, f"使用禁用字体：{', '.join(bad)}")
        bold = [f for f in fonts if 'Bold' in f and not f.startswith('STIX') and f not in bad]
        if bold:
            rep.add('WARN', 'F07', where, f"含粗体字形：{', '.join(bold)}（面板标题、坐标轴不加粗）")
        ppi = [p / scale for p in pdf_images_ppi(gfile)]
        if ppi:
            row['image_ppi_min'] = round(min(ppi))
            if min(ppi) < 300:
                rep.add('WARN', 'F08', where, f'内嵌位图成品分辨率最低 {min(ppi):.0f} ppi；渲染帧/照片须 ≥ 300，数据热图可说明后保留')
        mg = ink_margins(gfile, scale, tmp)
        if mg:
            row['margins_mm'] = mg
            if max(mg.values()) > 6 or abs(mg['left'] - mg['right']) > 4:
                rep.add('WARN', 'F09', where, f"画布白边 左{mg['left']} 右{mg['right']} 上{mg['top']} 下{mg['bottom']} mm；宜 ≤ 3 mm 且左右对称")
    else:
        eff = px_w / (out_w / 25.4)
        row['image_ppi_min'] = round(eff)
        if eff < 300:
            rep.add('ERROR', 'F08', where, f'位图成品分辨率 {eff:.0f} ppi，须 ≥ 300')
    return row


def check_chapter_settings(ch: dict, rep: Report):
    where = f"第{ch['number']}章"
    pk = ch['dir'] / 'packages.tex'
    if pk.exists():
        for i, line in enumerate(pk.read_text(encoding='utf-8').splitlines(), 1):
            line = strip_comments(line)
            if re.search(r'\\usepackage\[[^\]]*section[^\]]*\]\{placeins\}', line):
                rep.add('ERROR', 'C01', where, f'{pk.relative_to(WRITING).as_posix()}:{i} 加载 placeins[section]，会在每节前强制排完浮动体')
    st = ch['dir'] / 'settings.tex'
    if st.exists():
        for i, line in enumerate(st.read_text(encoding='utf-8').splitlines(), 1):
            line = strip_comments(line)
            if re.search(r'\\renewcommand\{?\\(top|bottom|text|floatpage|dbltop|dblfloatpage)fraction', line) or \
                    re.search(r'\\setcounter\{(top|bottom|total|dbltop)number\}', line):
                rep.add('ERROR', 'C02', where, f'{st.relative_to(WRITING).as_posix()}:{i} 在章内覆盖浮动参数：{line.strip()}')


# ----------------------------------------------------------------------------- page checks
def page_checks(pdf: Path, rows: list[dict], rep: Report, do_render: bool, tmp: Path) -> dict:
    aux, toc, log = pdf.with_suffix('.aux'), pdf.with_suffix('.toc'), pdf.with_suffix('.log')
    result = {'pdf': str(pdf), 'pdf_mtime': _dt.datetime.fromtimestamp(pdf.stat().st_mtime).isoformat(timespec='seconds')}
    labels = {}
    for m in re.finditer(r'\\newlabel\{(fig:[^}]*)\}\{\{([\d.]+)\}\{(\d+)\}', aux.read_text(encoding='utf-8', errors='replace')):
        labels.setdefault(m.group(2), (m.group(1), int(m.group(3))))
    starts, bib = [], None
    for m in re.finditer(r'\\contentsline \{chapter\}\{(.*?)\}\{(\d+)\}\{chapter', toc.read_text(encoding='utf-8', errors='replace')):
        if '参考文献' in m.group(1):
            bib = int(m.group(2))
        elif 'numberline' in m.group(1):
            starts.append(int(m.group(2)))
    pages = run(['pdftotext', '-enc', 'UTF-8', str(pdf), '-']).split('\f')
    cap_offsets = Counter()
    for num, (_, pg) in labels.items():
        pat = re.compile(r'(?m)^\s*图\s' + re.escape(num) + r'(?!\d)')
        for i, txt in enumerate(pages):
            if pat.search(txt):
                cap_offsets[i + 1 - pg] += 1
    if not cap_offsets:
        rep.add('WARN', 'P01', 'PDF', '无法从题注定位页码偏移，跳过页面检查')
        return result
    offset = cap_offsets.most_common(1)[0][0]
    body_end = (bib - 1) if bib else len(pages) - offset
    result.update({'page_offset': offset, 'chapter_starts': starts, 'body_end': body_end})

    by_page = Counter()
    dist = {}
    for num, (lab, fpage) in labels.items():
        by_page[fpage] += 1
        ref = re.compile(r'图\s?' + re.escape(num) + r'(?!\d)')
        cap = re.compile(r'(?m)^\s*图\s' + re.escape(num) + r'(?!\d)')
        refs = []
        for i, txt in enumerate(pages):
            pg = i + 1 - offset
            if pg < 1 or pg > body_end:
                continue
            capspans = {mm.start() for mm in cap.finditer(txt)}
            for mm in ref.finditer(txt):
                line_start = txt.rfind('\n', 0, mm.start()) + 1
                if any(line_start <= s <= mm.start() for s in capspans):
                    continue
                refs.append(pg)
        first = min(refs) if refs else None
        dist[num] = {'label': lab, 'page': fpage, 'first_ref': first}
        where = f'图{num} {lab}'
        if first is None:
            rep.add('ERROR', 'P01', where, '正文中找不到对该图的引用')
        elif fpage < first:
            rep.add('ERROR', 'P01', where, f'图在 p{fpage}，先于首次引用 p{first}（先见文字后见图）')
        elif fpage - first > 2:
            rep.add('ERROR', 'P01', where, f'图在 p{fpage}，距首次引用 p{first} 达 {fpage - first} 页')
        elif fpage - first == 2:
            rep.add('WARN', 'P01', where, f'图在 p{fpage}，距首次引用 p{first} 为 2 页')
    for pg, n in by_page.items():
        if n > 2:
            rep.add('WARN', 'P05', f'p{pg}', f'同页 {n} 张图')
    for r in rows:
        if r['number'] in dist:
            r.update({'page': dist[r['number']]['page'], 'first_ref': dist[r['number']]['first_ref']})
    result['figures'] = dist

    if log.exists():
        current = None
        envs_cache = {}
        for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
            for mm in re.finditer(r'(独立章节/[^/()\s]+/(?:sections|experiments)/[^()\s]+?\.tex)', line):
                current = mm.group(1)
            mo = re.search(r'Overfull \\hbox \(([\d.]+)pt too wide\) (in alignment|in paragraph) at lines (\d+)--(\d+)', line)
            if not (mo and current) or float(mo.group(1)) <= 1.0:
                continue
            path = WRITING / current
            if path not in envs_cache:
                envs_cache[path] = env_ranges(path)
            lo = int(mo.group(3))
            env = next((e for (a, b, e) in envs_cache[path] if a <= lo <= b), None)
            if env in ('table', 'figure'):
                rep.add('ERROR', 'P04', f'{current}:{lo}', f"{'表格' if env == 'table' else '图'}超出版心 {float(mo.group(1)) / PT_PER_MM:.1f} mm（{mo.group(2)}）")

    if do_render:
        result['blank_pages'] = blank_checks(pdf, offset, starts, body_end, rep, tmp)
    return result


def env_ranges(path: Path) -> list[tuple[int, int, str]]:
    out, stack = [], []
    if not path.exists():
        return out
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        for mm in re.finditer(r'\\(begin|end)\{(table|figure)\*?\}', strip_comments(line)):
            if mm.group(1) == 'begin':
                stack.append((i, mm.group(2)))
            elif stack:
                a, e = stack.pop()
                out.append((a, i, e))
    return out


def blank_checks(pdf: Path, offset: int, starts: list[int], body_end: int, rep: Report, tmp: Path) -> list[dict]:
    try:
        from PIL import Image
    except ImportError:
        rep.add('WARN', 'P02', 'PDF', '未安装 Pillow，跳过页底空白检查')
        return []
    first, last = 1 + offset, body_end + offset
    run(['pdftoppm', '-r', '40', '-gray', '-png', '-f', str(first), '-l', str(last), str(pdf), str(tmp / 'pg')])
    mm = 40 / 25.4
    y0, y1, x0, x1 = int(31 * mm), int(266 * mm), int(29 * mm), int(181 * mm)
    out = []
    for f in sorted(tmp.glob('pg-*.png')):
        pdfpage = int(f.stem.split('-')[1])
        pg = pdfpage - offset
        with Image.open(f) as im:
            g = im.convert('L')
            px = g.load()
            rows = [y for y in range(y0, y1) if any(px[x, y] < 200 for x in range(x0, x1, 2))]
        if not rows:
            continue
        top = (rows[0] - y0) / mm
        bottom = (y1 - 1 - rows[-1]) / mm
        chapter_end = (pg + 1) in starts or pg == body_end
        chapter_start = pg in starts
        out.append({'page': pg, 'top_blank_mm': round(top, 1), 'bottom_blank_mm': round(bottom, 1)})
        if bottom > 30 and not chapter_end:
            rep.add('ERROR', 'P02', f'p{pg}', f'页底空白 {bottom:.0f} mm（非章末页）')
        if top > 30 and not chapter_start:
            rep.add('ERROR', 'P03', f'p{pg}', f'页顶空白 {top:.0f} mm：疑似只排一张图的孤立浮动页')
    return out


# ----------------------------------------------------------------------------- reporting
def chapter_consistency(rows: list[dict], rep: Report):
    groups = defaultdict(list)
    for r in rows:
        if r.get('layout') in LAYOUTS and r.get('out_mm'):
            groups[(r['chapter'], r['layout'])].append(r)
    for (ch, lay), rs in groups.items():
        hs = [r['out_mm'][1] for r in rs]
        if len(rs) > 1 and max(hs) - min(hs) > 2:
            detail = '，'.join(f"图{r['number']} {r['out_mm'][1]:.0f}" for r in rs)
            rep.add('WARN', 'F14', f'第{ch}章 {lay}', f'同档高度相差 {max(hs) - min(hs):.0f} mm：{detail}')


def write_markdown(path: Path, rows, rep: Report, meta: dict, page_result):
    by_fig = defaultdict(list)
    for it in rep.items:
        m = re.match(r'图([\d.]+)', it['where'])
        if m:
            by_fig[m.group(1)].append(f"{it['code']}{'!' if it['level'] == 'ERROR' else '?'}")
    L = [f'# 图件规范 {SPEC_VERSION} 检查报告', '',
         f"- 生成时间：{meta['time']}", f"- 命令：`{meta['command']}`"]
    if page_result:
        L.append(f"- 检查的 PDF：`{page_result['pdf']}`（修改时间 {page_result['pdf_mtime']}，页码偏移 {page_result.get('page_offset')}）")
    L += [f"- 构图代号来源：tex = figure 内 `% layout=`；pdf = 图件 PDF 关键字；map = `--layout-map` 人工判读", '',
          '## 汇总', '', f"ERROR {rep.count('ERROR')} 条，WARN {rep.count('WARN')} 条。代码后 `!` 为 ERROR，`?` 为 WARN。", '',
          '| 代码 | 含义 | ERROR | WARN |', '|---|---|---:|---:|']
    cnt = Counter((i['code'], i['level']) for i in rep.items)
    for code, name in CODES.items():
        e, w = cnt[(code, 'ERROR')], cnt[(code, 'WARN')]
        if e or w:
            L.append(f'| {code} | {name} | {e} | {w} |')
    L += ['', '## 逐图', '',
          '| 图 | 标签 | 构图 | 画布 mm | 缩放 | 成品 mm | 字号 中位/最小正文 | <7.8pt | 问题 |',
          '|---|---|---|---|---:|---|---|---:|---|']
    for r in rows:
        f = r.get('font') or {}
        canvas = '×'.join(f'{v:.0f}' for v in r['canvas_mm']) if r.get('canvas_mm') else ('TikZ' if r.get('tikz') else '')
        out = '×'.join(f'{v:.0f}' for v in r['out_mm']) if r.get('out_mm') else ''
        lay = f"{r.get('layout') or '—'}" + (f"（{r['layout_src']}）" if r.get('layout_src') else '')
        fsz = f"{f.get('median')}/{f.get('base_min')}" if f else ''
        L.append(f"| {r['number']} | `{r['label']}` | {lay} | {canvas} | {r.get('scale', '')} | {out} | {fsz} | "
                 f"{str(f.get('pct_small')) + '%' if f else ''} | {' '.join(sorted(set(by_fig.get(r['number'], [])))) or '✓'} |")
    L += ['', '## 明细', '']
    for level in ('ERROR', 'WARN'):
        items = [i for i in rep.items if i['level'] == level]
        if not items:
            continue
        L += [f'### {level}（{len(items)}）', '']
        for code in CODES:
            sub = [i for i in items if i['code'] == code]
            if sub:
                L.append(f'**{code} {CODES[code]}**')
                L += [f"- {i['where']}：{i['message']}" for i in sub]
                L.append('')
    path.write_text('\n'.join(L) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='图件规范 3.0 自动验收（只读）')
    ap.add_argument('--pdf', type=Path, help='整篇论文 build/main.pdf；同目录需有 .aux/.toc/.log')
    ap.add_argument('--pages', action='store_true', help='渲染正文页，检查页底空白与孤立浮动页（需要 Pillow）')
    ap.add_argument('--layout-map', type=Path, help='JSON：{标签: 构图代号}，用于未声明代号的存量图')
    ap.add_argument('--out', type=Path, help='输出目录，写入 report.json 与 report.md（目录须为空或不存在）')
    ap.add_argument('--strict', action='store_true', help='有 ERROR 时退出码为 1')
    args = ap.parse_args()

    layout_map = json.loads(args.layout_map.read_text(encoding='utf-8')) if args.layout_map else {}
    layout_map = {k: v for k, v in layout_map.items() if not k.startswith('_')}
    rep, rows = Report(), []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for ch in load_chapters():
            check_chapter_settings(ch, rep)
            for fig in parse_figures(ch):
                rows.append(check_figure(fig, rep, layout_map, tmp))
        chapter_consistency(rows, rep)
        page_result = None
        if args.pdf:
            page_result = page_checks(args.pdf.resolve(), rows, rep, args.pages, tmp)

    meta = {'time': _dt.datetime.now().isoformat(timespec='seconds'), 'spec': SPEC_VERSION,
            'command': 'python check_figures.py ' + ' '.join(sys.argv[1:])}
    print(f"图件规范 {SPEC_VERSION}：{len(rows)} 张图，ERROR {rep.count('ERROR')}，WARN {rep.count('WARN')}")
    for level in ('ERROR', 'WARN'):
        for it in rep.items:
            if it['level'] == level:
                print(f"  [{level[0]}] {it['code']} {it['where']}：{it['message']}")
    if args.out:
        if args.out.exists() and any(args.out.iterdir()):
            sys.exit(f'输出目录 {args.out} 非空；请另建目录，避免覆盖已有记录')
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'report.json').write_text(json.dumps(
            {'meta': meta, 'figures': rows, 'findings': rep.items, 'pages': page_result},
            ensure_ascii=False, indent=1, default=str), encoding='utf-8')
        write_markdown(args.out / 'report.md', rows, rep, meta, page_result)
        print(f'报告已写入 {args.out}')
    if args.strict and rep.count('ERROR'):
        sys.exit(1)


if __name__ == '__main__':
    main()
