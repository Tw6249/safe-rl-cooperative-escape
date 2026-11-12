"""Side-by-side videos and keyframe strips of fresh-seed runs (results/run01), written to results/video02.

Replaces the captions of render_videos.py (video01), which stated unverified mechanisms ("邻居不让路", "被唤醒") and
non-learned baseline is shown. Reads saved npz/rows only; nothing is simulated, every panel is drawn the same way."""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'results' / 'run01'
OUT = HERE.parent / 'results' / 'video02'
sys.path.insert(0, str(HERE.parents[1] / '20261102_安全强化学习' / 'code'))
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a']
NAMES = {'rl_full_env_B8': 'rl_full + 包络（μ 门控）',
         'rl_geom_env_B8': 'rl_geom + 包络（修正量门控）', 'rl_full_int_env_B8': 'rl_full_int + 包络（修正量门控）',
         'heur_B8': '固定环流 heur（无学习）', 'myopic_B8': '短视 myopic（无学习）'}
MOVED = 0.025          # a robot whose goal is its start counts as displaced beyond this distance (m)


def ffmpeg_path():
    p = shutil.which('ffmpeg')
    if p:
        return p
    return subprocess.run(['powershell', '-NoProfile', '-Command', '(Get-Command ffmpeg).Source'],
                          capture_output=True, text=True).stdout.strip()


def outcome(r):
    if r['status'] == 'arrived':
        return f"到达 {r['arrival']:.2f} s"
    if r['status'] == 'stalled':
        return f"{r['duration']:.2f} s 判为停滞（速度为零）"
    return f"{r['duration']:.0f} s 未到达（误差 {r['final_max_err']:.2f} m）"


def load_panel(scene, tag):
    row = json.loads((RUN / 'rows' / f'{scene}_{tag}.json').read_text(encoding='utf8'))
    z = np.load(RUN / 'runs' / f'{scene}_{tag}.npz')
    return dict(tag=tag, name=NAMES[tag], row=row, st=z['states'], t=np.array(z['times']), goal=z['goal'])


def pos_at(d, t):
    tt = min(t, d['t'][-1])
    n = d['st'].shape[1]
    return np.stack([np.interp(tt, d['t'], d['st'][:, i, 0]) for i in range(n)]), \
        np.stack([np.interp(tt, d['t'], d['st'][:, i, 1]) for i in range(n)])


def err_color(e):
    t = float(np.clip(e / 2.0, 0, 1))
    a, b = np.array(matplotlib.colors.to_rgb('#d5d4cd')), np.array(matplotlib.colors.to_rgb(SLOTS[0]))
    return tuple(a + t * (b - a))


def robot_style(i, xi, yi, goal, movers, slot_mode, e):
    if slot_mode:
        if i in movers:
            k = movers.index(i)
            return SLOTS[k] + '55', SLOTS[k], 1.6, 4
        moved = np.hypot(xi - goal[i, 0], yi - goal[i, 1]) > MOVED
        return ('#cde8dc', SLOTS[2], 1.4, 3) if moved else ('#e5e4dd', '#b5b4aa', 0.8, 3)
    return err_color(e), (INK2 if e > 0.02 else '#b5b4aa'), 0.8, 3


def setup(movers_n):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return movers_n <= 3


def legend_text(slot_mode):
    if slot_mode:
        return ('蓝/橙：互换的两台（× 为各自目标）；浅绿：目标在原地、当前偏离目标 > 2.5 cm（让开或被挤开，原因不区分）；灰：其余。'
                '圆半径 0.5 m，相切即安全距离 1 m。运行结束后保持最后一帧。')
    return '圆的颜色随目标误差由深蓝（远）变为浅灰（到位）；× 为目标。圆半径 0.5 m，相切即安全距离 1 m。运行结束后保持最后一帧。'


def render_mp4(panels, suptitle, out_mp4, tmax, fps=25, speed=1.0):
    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path()
    goal, p0 = panels[0]['goal'], panels[0]['st'][0]
    n = len(p0)
    movers = [i for i in range(n) if np.linalg.norm(p0[i] - goal[i]) > 1e-6]
    slot_mode = setup(len(movers))
    lim = np.abs(np.concatenate([d['st'].reshape(-1, 2) for d in panels])).max() + 0.75
    t_grid = np.arange(0, tmax + 1.5, speed / fps)
    k = len(panels)
    fig, axes = plt.subplots(1, k, figsize=(3.4 * k, 4.8), facecolor=SURF)
    arts = []
    for ax, d in zip(np.atleast_1d(axes), panels):
        ax.set_facecolor(SURF)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_title(f"{d['name']}\n{outcome(d['row'])}", color=INK, fontsize=9.6, loc='left', pad=6)
        if slot_mode:
            for q, i in enumerate(movers):
                ax.plot(goal[i, 0], goal[i, 1], marker='x', markersize=8, markeredgewidth=2.0, color=SLOTS[q], zorder=5)
        else:
            ax.scatter(goal[movers, 0], goal[movers, 1], marker='x', s=28, linewidths=1.3, color=INK2, zorder=2)
        circles = [plt.Circle((p0[i, 0], p0[i, 1]), 0.5, alpha=0.9) for i in range(n)]
        for c in circles:
            ax.add_patch(c)
        tails = [ax.plot([], [], color=SLOTS[q] if slot_mode else SLOTS[0], linewidth=1.8 if slot_mode else 1.0,
                         alpha=0.85 if slot_mode else 0.4, zorder=4 if slot_mode else 2)[0] for q, _ in enumerate(movers)]
        badge = ax.text(0.97, 0.97, '', transform=ax.transAxes, ha='right', va='top', fontsize=9, color=INK,
                        fontweight='bold')
        info = ax.text(0.03, 0.03, '', transform=ax.transAxes, ha='left', va='bottom', fontsize=8.2, color=INK2)
        arts.append((d, circles, tails, badge, info))
    fig.suptitle(suptitle, x=0.012, y=0.975, ha='left', color=INK, fontsize=11, fontweight='bold')
    clock = fig.text(0.012, 0.015, '', color=INK2, fontsize=9)
    fig.text(0.988, 0.015, legend_text(slot_mode), color=INK2, fontsize=8.2, ha='right')
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))

    def draw(fi):
        t = min(float(t_grid[fi]), tmax)
        clock.set_text(f'物理时间 t = {t:5.2f} s（{speed:g} 倍速）')
        for d, circles, tails, badge, info in arts:
            x, y = pos_at(d, t)
            e = np.hypot(x - goal[:, 0], y - goal[:, 1])
            for i, c in enumerate(circles):
                c.center = (x[i], y[i])
                fc, ec, lw, zo = robot_style(i, x[i], y[i], goal, movers, slot_mode, e[i])
                c.set_facecolor(fc)
                c.set_edgecolor(ec)
                c.set_linewidth(lw)
                c.set_zorder(zo)
            idx = int(np.searchsorted(d['t'], min(t, d['t'][-1]), side='right'))
            for line, i in zip(tails, movers):
                line.set_data(d['st'][:max(idx, 1), i, 0], d['st'][:max(idx, 1), i, 1])
            r = d['row']
            ended = t >= (r['arrival'] if r['status'] == 'arrived' else r['duration'])
            badge.set_text(outcome(r) if ended else '')
            info.set_text(f'当前最大误差 {e.max():.2f} m')
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(t_grid), interval=1000 / fps, blit=False)
    w = animation.FFMpegWriter(fps=fps, codec='libx264',
                               extra_args=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', '-crf', '21'])
    anim.save(out_mp4, writer=w, dpi=100, savefig_kwargs=dict(facecolor=SURF))
    plt.close(fig)
    print('saved', out_mp4, flush=True)


def render_strip(panels, times, suptitle, out_png):
    goal, p0 = panels[0]['goal'], panels[0]['st'][0]
    n = len(p0)
    movers = [i for i in range(n) if np.linalg.norm(p0[i] - goal[i]) > 1e-6]
    slot_mode = setup(len(movers))
    lim = np.abs(np.concatenate([d['st'].reshape(-1, 2) for d in panels])).max() + 0.65
    nr, nc = len(panels), len(times)
    fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc + 2.6, 2.45 * nr + 0.9), facecolor=SURF)
    for ri, d in enumerate(panels):
        for ci, t in enumerate(times):
            ax = axes[ri, ci]
            ax.set_facecolor(SURF)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            x, y = pos_at(d, t)
            e = np.hypot(x - goal[:, 0], y - goal[:, 1])
            idx = int(np.searchsorted(d['t'], min(t, d['t'][-1]), side='right'))
            if slot_mode:
                for q, i in enumerate(movers):
                    ax.plot(goal[i, 0], goal[i, 1], marker='x', markersize=7, markeredgewidth=1.8, color=SLOTS[q], zorder=5)
                    ax.plot(d['st'][:max(idx, 1), i, 0], d['st'][:max(idx, 1), i, 1], color=SLOTS[q], linewidth=1.5,
                            alpha=0.85, zorder=4)
            for i in range(n):
                fc, ec, lw, zo = robot_style(i, x[i], y[i], goal, movers, slot_mode, e[i])
                ax.add_patch(plt.Circle((x[i], y[i]), 0.5, facecolor=fc, edgecolor=ec, linewidth=lw, alpha=0.9, zorder=zo))
            if ri == 0:
                ax.set_title(f't = {t:.1f} s', color=INK, fontsize=10, fontweight='bold', pad=6)
            if ci == 0:
                ax.text(-0.06, 0.5, f"{d['name']}\n{outcome(d['row'])}", transform=ax.transAxes, ha='right',
                        va='center', fontsize=9, color=INK)
            r = d['row']
            ended = t >= (r['arrival'] if r['status'] == 'arrived' else r['duration'])
            ax.text(0.04, 0.04, ('已结束 · ' if ended else '') + f'最大误差 {e.max():.2f} m', transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=7.8, color=INK2)
    fig.suptitle(suptitle, x=0.012, y=0.985, ha='left', color=INK, fontsize=11, fontweight='bold')
    fig.text(0.012, 0.008, legend_text(slot_mode), fontsize=8.3, color=INK2)
    fig.tight_layout(rect=(0.2, 0.025, 1, 0.95))
    fig.savefig(out_png, dpi=140, facecolor=SURF)
    plt.close(fig)
    print('saved', out_png, flush=True)


if __name__ == '__main__':
    import sysperf
    sysperf.configure()
    OUT.mkdir(parents=True, exist_ok=False)
    s1 = 'H20_local1_s20261105'
    p1 = [load_panel(s1, t) for t in order1]
    title1 = '新种子 20261105 · H20_local1（N=20：中间一对互换，其余 18 台目标在原地）'
    render_strip(p1, [0.0, 1.0, 2.5, 5.0, 8.0, 13.5], title1 + ' 关键帧', OUT / 'fresh_H20_local1_strip.png')
    render_mp4(p1, title1, OUT / 'fresh_H20_local1_swap.mp4', tmax=15.0)
    s2 = 'H20_perm1_s20261105'
    p2 = [load_panel(s2, t) for t in order2]
    title2 = '新种子 20261105 · H20_perm1（N=20：随机全排列，18 台要移动）'
    render_strip(p2, [0.0, 4.0, 9.0, 15.0, 21.5], title2 + ' 关键帧', OUT / 'fresh_H20_perm1_strip.png')
    render_mp4(p2, title2, OUT / 'fresh_H20_perm1_allmove.mp4', tmax=23.0, speed=1.5)
