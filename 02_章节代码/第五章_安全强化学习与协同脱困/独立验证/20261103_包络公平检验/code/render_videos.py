"""Render side-by-side MP4 videos and keyframe filmstrips for fresh-seed (20261105) tasks from saved trajectories."""
import json, shutil, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'results' / 'run01'
OUT = HERE.parent / 'results' / 'video01'
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a']
STATUS = {'arrived': '到达', 'stalled': '卡死', 'certification_stop': '证书停止', 'horizon_reached': '60 s 未到达'}


def ffmpeg_path():
    p = shutil.which('ffmpeg')
    if p:
        return p
    return subprocess.run(['powershell', '-NoProfile', '-Command', '(Get-Command ffmpeg).Source'],
                          capture_output=True, text=True).stdout.strip()


def err_color(e):
    t = float(np.clip(e / 2.0, 0, 1))
    a, b = np.array(matplotlib.colors.to_rgb('#d5d4cd')), np.array(matplotlib.colors.to_rgb(SLOTS[0]))
    return tuple(a + t * (b - a))


def load_panel(scene, tag, title):
    row = json.loads((RUN / 'rows' / f'{scene}_{tag}.json').read_text(encoding='utf8'))
    z = np.load(RUN / 'runs' / f'{scene}_{tag}.npz')
    return dict(tag=tag, title=title, row=row, st=z['states'], t=np.array(z['times']), goal=z['goal'])


def pos_at(d, t):
    tt = min(t, d['t'][-1])
    n = d['st'].shape[1]
    return (np.stack([np.interp(tt, d['t'], d['st'][:, i, 0]) for i in range(n)]),
            np.stack([np.interp(tt, d['t'], d['st'][:, i, 1]) for i in range(n)]))


def render_mp4(scene, panels, suptitle, out_mp4, tmax=18.0, fps=25, speed=1.0):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path()
    goal = panels[0]['goal']
    p0 = panels[0]['st'][0]
    n = len(p0)
    movers = [i for i in range(n) if np.linalg.norm(p0[i] - goal[i]) > 1e-6]
    slot_mode = len(movers) <= 3
    lim = np.abs(np.concatenate([d['st'].reshape(-1, 2) for d in panels])).max() + 0.75
    t_grid = np.arange(0, tmax + 1.5, speed / fps)

    k = len(panels)
    fig, axes = plt.subplots(1, k, figsize=(3.4 * k, 4.8), facecolor=SURF)
    axes = np.atleast_1d(axes)
    arts = []
    for ax, d in zip(axes, panels):
        ax.set_facecolor(SURF)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_title(d['title'], color=INK, fontsize=9.8, loc='left', pad=6)
        if slot_mode:
            for rank, i in enumerate(movers):
                ax.plot(goal[i, 0], goal[i, 1], marker='x', markersize=8, markeredgewidth=2.0,
                        color=SLOTS[rank], zorder=5)
        else:
            ax.scatter(goal[movers, 0], goal[movers, 1], marker='x', s=28, linewidths=1.3, color=INK2, zorder=2)
        circles = [plt.Circle((p0[i, 0], p0[i, 1]), 0.5, facecolor='#e5e4dd', edgecolor='#b5b4aa',
                              linewidth=1.0, alpha=0.88, zorder=3) for i in range(n)]
        for rank, i in enumerate(movers):
            if slot_mode:
                circles[i].set_facecolor(SLOTS[rank] + '55')
                circles[i].set_edgecolor(SLOTS[rank])
                circles[i].set_linewidth(1.8)
                circles[i].set_zorder(4)
        for c in circles:
            ax.add_patch(c)
        tails = [ax.plot([], [], color=SLOTS[rank] if slot_mode else SLOTS[0],
                         linewidth=1.8 if slot_mode else 1.1, alpha=0.85 if slot_mode else 0.45,
                         zorder=4 if slot_mode else 2)[0] for rank, _ in enumerate(movers)]
        badge = ax.text(0.97, 0.96, '', transform=ax.transAxes, ha='right', va='top', fontsize=8.8,
                        color=INK, fontweight='bold')
        info = ax.text(0.03, 0.03, '', transform=ax.transAxes, ha='left', va='bottom', fontsize=8.2, color=INK2)
        arts.append((d, circles, tails, badge, info))

    fig.suptitle(suptitle, x=0.015, y=0.97, ha='left', color=INK, fontsize=11, fontweight='bold')
    clock = fig.text(0.015, 0.015, '', color=INK2, fontsize=9)
    note = ('彩色圆：需互换位置的机器人（× 为目标）；灰色圆：目标在原地的邻居（需主动让路再回位）。圆半径 0.5 m，相切即安全距离 1 m。'
            if slot_mode else '圆颜色随目标误差由深蓝（远）退为浅灰（到位）；× 为各自目标。圆半径 0.5 m，相切即安全距离 1 m。')
    fig.text(0.985, 0.015, note, color=INK2, fontsize=8.2, ha='right')
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))

    def draw(fi):
        t = min(float(t_grid[fi]), tmax)
        clock.set_text(f'物理时间 t = {t:5.2f} s')
        for d, circles, tails, badge, info in arts:
            x, y = pos_at(d, t)
            e = np.linalg.norm(np.stack([x, y], axis=1) - goal, axis=1)
            for i, c in enumerate(circles):
                c.center = (x[i], y[i])
                if not slot_mode:
                    c.set_facecolor(err_color(e[i]))
                    c.set_edgecolor(INK2 if e[i] > 0.02 else '#b5b4aa')
                elif i not in movers:
                    # highlight stationary neighbours when they temporarily move > 2 cm to yield
                    disp = np.linalg.norm([x[i] - goal[i, 0], y[i] - goal[i, 1]])
                    c.set_facecolor('#cde8dc' if disp > 0.025 else '#e5e4dd')
                    c.set_edgecolor('#1baf7a' if disp > 0.025 else '#b5b4aa')
                    c.set_linewidth(1.5 if disp > 0.025 else 1.0)
            idx = int(np.searchsorted(d['t'], min(t, d['t'][-1]), side='right'))
            for line, i in zip(tails, movers):
                line.set_data(d['st'][:max(idx, 1), i, 0], d['st'][:max(idx, 1), i, 1])
            r = d['row']
            if r['status'] == 'arrived' and r['arrival'] is not None and t >= r['arrival']:
                badge.set_text(f"已到达 {r['arrival']:.2f} s")
                badge.set_color('#1baf7a')
            elif r['status'] != 'arrived' and t >= tmax - 0.5:
                badge.set_text(f"60 s 未到 (终差 {r['final_max_err']*100:.0f} cm)")
                badge.set_color('#eb6834')
            else:
                badge.set_text('')
            info.set_text(f'当前最大误差 {e.max():.2f} m')
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(t_grid), interval=1000 / fps, blit=False)
    w = animation.FFMpegWriter(fps=fps, codec='libx264',
                               extra_args=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', '-crf', '21'])
    anim.save(out_mp4, writer=w, dpi=100, savefig_kwargs=dict(facecolor=SURF))
    plt.close(fig)
    print('saved', out_mp4)


def render_filmstrip(scene, panels, times, suptitle, out_png):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    goal = panels[0]['goal']
    p0 = panels[0]['st'][0]
    n = len(p0)
    movers = [i for i in range(n) if np.linalg.norm(p0[i] - goal[i]) > 1e-6]
    slot_mode = len(movers) <= 3
    lim = np.abs(np.concatenate([d['st'].reshape(-1, 2) for d in panels])).max() + 0.65

    nr, nc = len(panels), len(times)
    fig, axes = plt.subplots(nr, nc, figsize=(2.75 * nc + 1.2, 2.55 * nr + 0.8), facecolor=SURF)
    for ri, d in enumerate(panels):
        for ci, t in enumerate(times):
            ax = axes[ri, ci]
            ax.set_facecolor(SURF)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            x, y = pos_at(d, t)
            e = np.linalg.norm(np.stack([x, y], axis=1) - goal, axis=1)
            idx = int(np.searchsorted(d['t'], min(t, d['t'][-1]), side='right'))
            if slot_mode:
                for rank, i in enumerate(movers):
                    ax.plot(goal[i, 0], goal[i, 1], marker='x', markersize=7, markeredgewidth=1.8,
                            color=SLOTS[rank], zorder=5)
                    ax.plot(d['st'][:max(idx, 1), i, 0], d['st'][:max(idx, 1), i, 1], color=SLOTS[rank],
                            linewidth=1.5, alpha=0.85, zorder=4)
            for i in range(n):
                if slot_mode:
                    if i in movers:
                        rank = movers.index(i)
                        fc, ec, lw, zo = SLOTS[rank] + '55', SLOTS[rank], 1.6, 4
                    else:
                        disp = np.linalg.norm([x[i] - goal[i, 0], y[i] - goal[i, 1]])
                        fc = '#cde8dc' if disp > 0.025 else '#e5e4dd'
                        ec = '#1baf7a' if disp > 0.025 else '#b5b4aa'
                        lw, zo = (1.4 if disp > 0.025 else 0.8), 3
                else:
                    fc, ec, lw, zo = err_color(e[i]), (INK2 if e[i] > 0.02 else '#b5b4aa'), 0.8, 3
                ax.add_patch(plt.Circle((x[i], y[i]), 0.5, facecolor=fc, edgecolor=ec, linewidth=lw, alpha=0.9, zorder=zo))
            if ri == 0:
                ax.set_title(f't = {t:.1f} s', color=INK, fontsize=10, fontweight='bold', pad=6)
            if ci == 0:
                ax.text(-0.08, 0.5, d['title'], transform=ax.transAxes, ha='right', va='center',
                        fontsize=9.5, color=INK, fontweight='bold')
            r = d['row']
            done = r['status'] == 'arrived' and r['arrival'] is not None and t >= r['arrival']
            txt = f"已到达 ({r['arrival']:.1f} s)" if done else f"最大误差 {e.max():.2f} m"
            ax.text(0.04, 0.04, txt, transform=ax.transAxes, ha='left', va='bottom', fontsize=7.8,
                    color=('#1baf7a' if done else INK2), fontweight=('bold' if done else 'normal'))
    fig.suptitle(suptitle, x=0.02, y=0.98, ha='left', color=INK, fontsize=11, fontweight='bold')
    note = ('蓝/橙圆：互换机器人（× 为目标）；浅绿圆：正在主动让路的原地邻居（位移 > 2.5 cm）；灰圆：静止邻居。'
            if slot_mode else '深蓝圆：未到位机器人；浅灰圆：已到位机器人。')
    fig.text(0.02, 0.01, note, fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0.14, 0.03, 1, 0.94))
    fig.savefig(out_png, dpi=145, facecolor=SURF)
    plt.close(fig)
    print('saved', out_png)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    sc1 = 'H20_local1_s20261105'
    p1 = [
        load_panel(sc1, 'myopic_B8', '短视标称 myopic\n(卡死于 1.00 m)'),
        load_panel(sc1, 'rl_full_int_env_B8', '修正量门控 rl_full_int+包络\n(邻居不让路，冻结于 1.00 m)'),
        load_panel(sc1, 'rl_geom_env_B8', '纯几何 rl_geom+包络\n(让路不协调，半途卡死于 1.02 m)'),
        load_panel(sc1, 'rl_full_env_B8', '本文方法 rl_full+包络\n(协同让道互换，13.45 s 到达)'),
    ]
    render_mp4(sc1, p1, '新种子任务 H20_local1 (N=20 内部单对互换)：各方法闭环轨迹对比',
               OUT / 'fresh_H20_local1_swap.mp4', tmax=15.0, fps=25, speed=1.0)
    render_filmstrip(sc1, p1, [0.0, 2.5, 5.0, 8.0, 13.5],
                     '新种子任务 H20_local1 (N=20 内部单对互换) 关键帧对比：绿色圆为被唤醒主动挪窝让路的原地邻居',
                     OUT / 'fresh_H20_local1_filmstrip.png')

    sc2 = 'H20_perm1_s20261105'
    p2 = [
        load_panel(sc2, 'myopic_B8', '短视标称 myopic\n(顶牛死锁于 1.00 m)'),
        load_panel(sc2, 'heur_B8', '固定环流 heur\n(外圈乱转，停于 3.64 m)'),
        load_panel(sc2, 'rl_full_env_B8', '本文方法 rl_full+包络\n(全员协同穿插，21.30 s 到达)'),
    ]
    render_mp4(sc2, p2, '新种子任务 H20_perm1 (N=20 全排列大洗牌)：20 机密集穿插与终点收敛',
               OUT / 'fresh_H20_perm1_allmove.mp4', tmax=23.0, fps=25, speed=1.5)
    render_filmstrip(sc2, p2, [0.0, 4.0, 9.0, 15.0, 21.5],
                     '新种子任务 H20_perm1 (N=20 全排列大洗牌) 关键帧对比：深蓝=远，浅灰=已到位',
                     OUT / 'fresh_H20_perm1_filmstrip.png')
