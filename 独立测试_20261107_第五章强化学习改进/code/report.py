"""Tables for README §7+ from saved rows only (no re-computation of any run).

    python -B -X utf8 report.py phaseA ../results/phaseA
    python -B -X utf8 report.py train ../results            (learning curves of results/train_*)
    python -B -X utf8 report.py phaseC ../results/phaseC [--ref ../results/phaseA]
"""
import paths  # noqa: F401
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest


def load(d):
    rows = []
    for f in sorted((Path(d) / 'rows').glob('*.json')):
        r = json.loads(f.read_text(encoding='utf8'))
        j = r['job']
        r['_key'] = (j['set'], j['scene'], j['label'], bool(j['mirror']))
        rows.append(r)
    return rows


def idx(rows):
    return {r['_key']: r for r in rows}


def arrived(r):
    return r is not None and r.get('status') == 'arrived'


def fmt(x, nd=2):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{nd}f}'


def pval(a, b):
    return binomtest(a, a + b).pvalue if a + b else 1.0


def paired(I, pairs):
    """pairs: list of (key_x, key_y). Returns discordant counts and paired time stats (x - y)."""
    ox = oy = both = 0
    dts = []
    missing = 0
    for kx, ky in pairs:
        rx, ry = I.get(kx), I.get(ky)
        if rx is None or ry is None:
            missing += 1
            continue
        ax, ay = arrived(rx), arrived(ry)
        ox += ax and not ay
        oy += ay and not ax
        if ax and ay:
            both += 1
            dts.append(rx['arrival'] - ry['arrival'])
    dts = np.array(dts)
    faster, slower = int((dts < -1e-9).sum()), int((dts > 1e-9).sum())
    return dict(only_x=ox, only_y=oy, p_arrive=pval(ox, oy), both=both,
                dt_median=float(np.median(dts)) if len(dts) else None, dt_mean=float(dts.mean()) if len(dts) else None,
                x_faster=faster, x_slower=slower, p_time=pval(faster, slower), missing=missing,
                abs_dt_median=float(np.median(np.abs(dts))) if len(dts) else None)


def group_summary(rows, sets, label, mirror=False):
    rs = [r for r in rows if r['_key'][0] in sets and r['_key'][2] == label and r['_key'][3] == mirror]
    arr = [r for r in rs if arrived(r)]
    st = {}
    for r in rs:
        st[r.get('status')] = st.get(r.get('status'), 0) + 1
    g = lambda k: [r[k] for r in rs if r.get(k) is not None]
    return dict(n=len(rs), arrived=len(arr), t_med=float(np.median([r['arrival'] for r in arr])) if arr else None,
                status=st, audit_failed=int(sum(r.get('audit_failed') or 0 for r in rs)),
                job_errors=int(sum(r.get('status') == 'job_error' for r in rs)),
                cuts=int(sum(r.get('step_cuts') or 0 for r in rs)),
                min_dist=min(g('audit_min_distance'), default=None),
                min_lam_margin=min((r['min_lambda2_lower'] - r['lambda_required'] for r in rs
                                    if r.get('min_lambda2_lower') is not None), default=None),
                interv=float(np.mean(g('tr_interv_mean'))) if g('tr_interv_mean') else None,
                sat_w=float(np.mean(g('tr_sat_w'))) if g('tr_sat_w') else None,
                sat_r=float(np.mean(g('tr_sat_r'))) if g('tr_sat_r') else None,
                sat_psi=float(np.mean(g('tr_sat_psi'))) if g('tr_sat_psi') else None,
                absw=float(np.mean(g('tr_absw_mean'))) if g('tr_absw_mean') else None,
                lat_nom=float(np.mean(g('tr_lat_nom_moving'))) if g('tr_lat_nom_moving') else None,
                lat_u=float(np.mean(g('tr_lat_u_moving'))) if g('tr_lat_u_moving') else None,
                acc_mean=float(np.mean(g('acc_mean'))) if g('acc_mean') else None,
                acc_p99=float(np.mean(g('acc_p99'))) if g('acc_p99') else None,
                acc_over=float(np.mean(g('acc_frac_over'))) if g('acc_frac_over') else None,
                frozen=float(np.median(g('frozen_robot_frac'))) if g('frozen_robot_frac') else None)


def scenes_of(rows, st):
    return sorted({r['_key'][1] for r in rows if r['_key'][0] == st})


def table(lines, head, body):
    lines.append('| ' + ' | '.join(head) + ' |')
    lines.append('|' + '---|' * len(head))
    for b in body:
        lines.append('| ' + ' | '.join(str(x) for x in b) + ' |')
    lines.append('')


def summary_rows(rows, labels, sets_list):
    body = []
    for sets, sname in sets_list:
        for lab, mir in labels:
            s = group_summary(rows, sets, lab, mir)
            if not s['n']:
                continue
            body.append([sname, lab + (' [镜像]' if mir else ''), f"{s['arrived']}/{s['n']}", fmt(s['t_med']),
                         ', '.join(f'{k} {v}' for k, v in sorted(s['status'].items()) if k != 'arrived') or '—',
                         s['audit_failed'], s['cuts'], fmt(s['min_dist'], 6), fmt(s['min_lam_margin'], 3)])
    return body


def phaseA(d):
    rows = load(d)
    I = idx(rows)
    L = [f'# 阶段 A 报告（自动生成，`code/report.py phaseA`，{len(rows)} 行）', '']
    errs = [r for r in rows if r.get('status') == 'job_error']
    L.append(f"作业错误 {len(errs)}；审计失败段合计 {sum(r.get('audit_failed') or 0 for r in rows)}。")
    L.append('')
    head = ['集合', '组', '到达', '到达时间中位数 s', '其余状态', '审计失败段', '切步', '最小间距', 'min(λ₂下界−λ_req)']
    labels = [('myopic', False), ('thesis', False), ('thesis_o2', False),
              ('thesis', True), ('thesis_o2', True)]
    table(L, head, summary_rows(rows, labels, [({'SYM'}, 'SYM'), ({'E1n'}, 'E1n'), ({'E2n'}, 'E2n'),
                                                ({'E1n', 'E2n'}, 'E1n+E2n')]))
    L.append('## SYM 逐场景'); L.append('')
    body = []
    for s in scenes_of(rows, 'SYM'):
        for lab in ('myopic', 'thesis', 'thesis_o2'):
            r = I.get(('SYM', s, lab, False))
            if r:
                body.append([s, lab, r.get('status'), fmt(r.get('arrival')), fmt(r.get('final_max_err'), 4),
                             fmt(r.get('tr_absw_mean'), 4)])
    table(L, ['场景', '组', '状态', '到达 s', '终态最大误差 m', 'mean|ω|'], body)
    L.append('## 配对比较（x − y；不一致对精确二项检验；时间只用双方都到达的配对）'); L.append('')
    body = []
    for sets, sname in (({'E1n'}, 'E1n'), ({'E2n'}, 'E2n'), ({'E1n', 'E2n'}, '合计')):
        sc = [(st, s) for st in sorted(sets) for s in scenes_of(rows, st)]
        for (lx, mx), (ly, my), nm in (
                (('thesis_o2', False), ('thesis', False), 'thesis_o2 − thesis（H6b）'),
                (('thesis', False), ('thesis', True), 'thesis 原 − 镜像（H6a）'),
                (('thesis_o2', False), ('thesis_o2', True), 'thesis_o2 原 − 镜像（等变核对）')):
            P = paired(I, [((st, s, lx, mx), (st, s, ly, my)) for st, s in sc])
            if P['missing'] == len(sc):
                continue
            body.append([sname, nm, f"{P['only_x']}:{P['only_y']}", fmt(P['p_arrive'], 3), P['both'],
                         fmt(P['dt_median']), fmt(P['dt_mean']), f"{P['x_faster']}/{P['x_slower']}",
                         fmt(P['p_time'], 3), fmt(P['abs_dt_median'])])
    table(L, ['集合', '比较', '仅 x : 仅 y', 'p', '都到达', 'Δt 中位数', 'Δt 均值', 'x 快/慢', 'p(符号)',
              '|Δt| 中位数'], body)
    L.append('## 学习组的诊断量（逐拍平均后对运行取平均）'); L.append('')
    body = []
    for sets, sname in (({'E1n'}, 'E1n'), ({'E2n'}, 'E2n')):
        for lab, mir in (('thesis', False), ('thesis_o2', False)):
            s = group_summary(rows, sets, lab, mir)
            if s['n']:
                body.append([sname, lab, fmt(s['interv'], 4), fmt(s['sat_w'], 3), fmt(s['sat_r'], 3),
                             fmt(s['sat_psi'], 3), fmt(s['absw'], 3), fmt(s['lat_nom'], 3), fmt(s['lat_u'], 3),
                             fmt(s['acc_mean'], 3), fmt(s['acc_p99'], 3), fmt(s['acc_over'], 4)])
    table(L, ['集合', '组', '平均干预 m/s', '|ω|≥1 比例', '|ρ|≥1 比例', '|ψ|≥1 比例', 'mean|ω|',
              '移动者侧向标称', '移动者侧向执行', '指令跳变均值 m/s²', 'p99', '>0.8 比例'], body)
    L.append('## 逐场景（E1n、E2n）'); L.append('')
    body = []
    for st in ('E1n', 'E2n'):
        for s in scenes_of(rows, st):
            cells = [st, s]
            for lab, mir in (('thesis', False), ('thesis', True),
                             ('thesis_o2', False), ('thesis_o2', True)):
                r = I.get((st, s, lab, mir))
                cells.append('—' if r is None else (fmt(r['arrival']) if arrived(r) else
                                                     f"{r.get('status')}({fmt(r.get('final_max_err'), 3)})"))
            body.append(cells)
    table(L, ['集合', '场景', 'thesis', 'thesis 镜像', 'thesis_o2', 'thesis_o2 镜像'], body)
    (Path(d).parent / (Path(d).name + '_report.md')).write_text('\n'.join(L), encoding='utf8')
    print('\n'.join(L))


def train_curves(d):
    L = ['# 训练曲线汇总（自动生成）', '']
    body = []
    for t in sorted(Path(d).glob('train_*')):
        f = t / 'train_log.jsonl'
        if not f.exists():
            continue
        rows = [json.loads(x) for x in f.read_text(encoding='utf8').splitlines() if x.strip()]
        proto = json.loads((t / 'protocol.json').read_text(encoding='utf8'))
        vals = [(r['iter'], r['env_steps'], r['val']) for r in rows if 'val' in r]
        first1 = next(((i, s) for i, s, v in vals if v['success'] >= 1.0 - 1e-9), (None, None))
        last = rows[-1]
        best = json.loads((t / 'done.json').read_text())['best'] if (t / 'done.json').exists() else None
        bv = next((v for i, s, v in vals if best and i == best[1]), None)
        gp = [r['grad_probe']['ratio_at_lambda'] for r in rows if r.get('grad_probe')]
        body.append([t.name, proto['critic'], proto['proj'], proto['seed'], last['iter'], last['env_steps'],
                     ' '.join(f"{v['success']:.2f}" for i, s, v in vals), first1[0], first1[1],
                     best[1] if best else '—', fmt(bv['mean_time']) if bv else '—',
                     fmt(float(np.mean([r['interv_mean'] for r in rows[-16:] if 'interv_mean' in r])), 4),
                     fmt(float(np.median(gp)), 3) if gp else '—', 'done' if (t / 'done.json').exists() else 'running'])
    table(L, ['运行', 'critic', 'λ', '种子', '轮', '环境步', '验证成功率序列', '首次 1.0 轮', '首次 1.0 步',
              '选中轮', '选中验证平均时间', '末 16 轮训练干预', 'λ·梯度比中位数', '状态'], body)
    (Path(d) / 'train_curves.md').write_text('\n'.join(L), encoding='utf8')
    print('\n'.join(L))


def arm_of(label):
    """phaseC labels: c{team|local}_p{0|1}_s{1|2}."""
    c, p, s = label.split('_')
    return c[1:], p[1:], s[1:]


def phaseC(d, ref=None):
    dirs = d.split(',')                         # phase C was run in parts (README §8); rows are merged
    rows = [r for x in dirs for r in load(x)]
    assert len({r['_key'] for r in rows}) == len(rows), 'duplicate rows across parts'
    I = idx(rows)
    labels = sorted({r['_key'][2] for r in rows})
    L = [f'# 阶段 C 报告（自动生成，{len(rows)} 行）', '']
    L.append(f"作业错误 {sum(r.get('status') == 'job_error' for r in rows)}；审计失败段合计 "
             f"{sum(r.get('audit_failed') or 0 for r in rows)}。")
    L.append('')
    head = ['集合', '检查点', '到达', '到达时间中位数 s', '其余状态', '审计失败段', '切步', '最小间距', 'min(λ₂下界−λ_req)']
    sets_list = [({'E1n'}, 'E1n'), ({'E2n'}, 'E2n'), ({'E1n', 'E2n'}, '合计')]
    body = summary_rows(rows, [(lab, False) for lab in labels], sets_list)
    if ref:
        rr = load(ref)
        for r in rr:
                I[r['_key']] = r
    table(L, head, body)
    L.append('## 诊断量'); L.append('')
    body = []
    for sets, sname in sets_list[:2]:
        for lab in labels + (['thesis'] if ref else []):
            s = group_summary(rows if lab != 'thesis' else load(ref), sets, lab)
            body.append([sname, lab, fmt(s['interv'], 4), fmt(s['sat_w'], 3), fmt(s['sat_psi'], 3),
                         fmt(s['lat_nom'], 3), fmt(s['lat_u'], 3), fmt(s['acc_mean'], 3), fmt(s['acc_p99'], 3),
                         fmt(s['acc_over'], 4)])
    table(L, ['集合', '检查点', '平均干预', '|ω|≥1', '|ψ|≥1', '移动者侧向标称', '移动者侧向执行', '跳变均值',
              'p99', '>0.8'], body)
    sc = [(st, s) for st in ('E1n', 'E2n') for s in scenes_of(rows, st)]
    L.append('## 析因配对（x − y，同种子、另一因素同水平、同场景）'); L.append('')
    comps = []
    for c in ('team', 'local'):
        for s in ('1', '2'):
            comps.append(('升级二主效应 proj 开 − 关', f'c{c}_p1_s{s}', f'c{c}_p0_s{s}'))
    for p in ('0', '1'):
        for s in ('1', '2'):
            comps.append(('升级一主效应 local − team', f'clocal_p{p}_s{s}', f'cteam_p{p}_s{s}'))
    for s in ('1', '2'):
        comps.append(('组合 − 基线', f'clocal_p1_s{s}', f'cteam_p0_s{s}'))
    for c in ('team', 'local'):
        for p in ('0', '1'):
            comps.append(('种子噪声 s2 − s1', f'c{c}_p{p}_s2', f'c{c}_p{p}_s1'))
    for s in ('1', '2'):
        comps.append(('基线 − 论文检查点', f'cteam_p0_s{s}', 'thesis'))
    body, groups = [], {}
    for nm, x, y in comps:
        groups.setdefault(nm, []).append((x, y))
    for nm, prs in groups.items():
        allp = [((st, s, x, False), (st, s, y, False)) for x, y in prs for st, s in sc]
        P = paired(I, allp)
        per = []
        for x, y in prs:
            q = paired(I, [((st, s, x, False), (st, s, y, False)) for st, s in sc])
            per.append(f"{x} vs {y}: {q['only_x']}:{q['only_y']}, Δt中位 {fmt(q['dt_median'])}")
        body.append([nm, len(prs), f"{P['only_x']}:{P['only_y']}", fmt(P['p_arrive'], 4), P['both'],
                     fmt(P['dt_median']), fmt(P['dt_mean']), f"{P['x_faster']}/{P['x_slower']}", fmt(P['p_time'], 4),
                     '<br>'.join(per)])
    table(L, ['比较', '配对组数', '仅 x : 仅 y', 'p', '都到达', 'Δt 中位数', 'Δt 均值', 'x 快/慢', 'p(符号)', '分组'], body)
    L.append('注：析因配对中同一场景出现在多对里，配对不独立；p 值只作描述，结论以“是否超出种子噪声尺度”为准（README §2）。')
    L.append('')
    L.append('## 逐场景到达时间'); L.append('')
    cols = labels + (['thesis'] if ref else [])
    body = []
    for st, s in sc:
        cells = [st, s]
        for lab in cols:
            r = I.get((st, s, lab, False))
            cells.append('—' if r is None else (fmt(r['arrival']) if arrived(r) else
                                                 f"{r.get('status')}({fmt(r.get('final_max_err'), 3)})"))
        body.append(cells)
    table(L, ['集合', '场景'] + cols, body)
    (Path(dirs[0]).parent / 'phaseC_report.md').write_text('\n'.join(L), encoding='utf8')
    print('\n'.join(L))


if __name__ == '__main__':
    mode, d = sys.argv[1], sys.argv[2]
    if mode == 'phaseA':
        phaseA(d)
    elif mode == 'train':
        train_curves(d)
    elif mode == 'phaseC':
        ref = sys.argv[sys.argv.index('--ref') + 1] if '--ref' in sys.argv else None
        phaseC(d, ref)
