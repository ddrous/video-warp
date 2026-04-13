#%% Fork Visualization — v5 (final improvements)
#
# Changes vs v4:
# - Titles lowered, prefixed with (a), (b), (c).
# - t=5 label in pure red.
# - NaiveWM colour changed to yellowish-orange.
# - Corrupted frames use a custom reddish colormap instead of grey.
# - Alien injection uses the same reddish colormap, no red border, shown for all methods.
# - Reduced white space between time axis and first method column.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap

# ══════════════════════════════════════════════════════════════════════════════
# 0.  USER-TUNABLE KNOBS
# ══════════════════════════════════════════════════════════════════════════════

CORNER_RADIUS       = 5      # FancyBboxPatch rounding (pixels in data-coords)

BORDER_LW_CLEAN     = 1.6    # linewidth – unperturbed & corrupted frames
BORDER_LW_INJECT    = 1.6    # linewidth – alien injection (same as clean)

BORDER_COL_CLEAN    = '#AAAAAA'   # neutral grey for all normal frames
BORDER_COL_INJECT   = '#CC0000'   # alien injection also neutral grey (no red border)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':     'sans-serif',
    # 'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
})

# Method accent colours – used for text labels and arrows (arrows now grey)
LABEL_COLS = {
    'NaiveWM': '#E9C46A',   # yellowish-orange (was pinkish)
    'LAPO':    '#06D6A0',
    'Ours':    '#118AB2',
}

# Per-method image colourmap: black → method hue → near‑white
CMAPS = {
    'NaiveWM': LinearSegmentedColormap.from_list('nwm', ['#030303', "#CC9200", '#FFDDA0']),
    'LAPO':    LinearSegmentedColormap.from_list('lpo', ['#030303', '#009960', '#BDFAE0']),
    'Ours':    LinearSegmentedColormap.from_list('vwp', ['#030303', '#006FA8', '#B8E8FF']),
}

# Reddish colormap for corrupted frames and alien injection
RED_CMAP = LinearSegmentedColormap.from_list('red_cmap', ['#1A0000', '#CC0000', '#FF8888'])

# BG_COLOR = '#F2F2F2'
BG_COLOR = "#FCFBFB"

# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
METHOD_SPECS = [
    {'name': 'NaiveWM', 'file': 'arrays_naive_wm.npz', 'col_offset': 1, 'label': '(a) Standard WM'},
    {'name': 'LAPO',    'file': 'arrays_lapo.npz',     'col_offset': 5, 'label': '(b) LAPO'},
    {'name': 'Ours',    'file': 'arrays_vwarp.npz',    'col_offset': 9, 'label': '(c) Ours'},
]

data_cache = {}
for m in METHOD_SPECS:
    try:
        d = np.load(m['file'])
        data_cache[m['name']] = {
            'clean':   d['clean_pred_video'],
            'corrupt': d['corrupt_pred_video'],
            'ref':     d['corrupt_frame_ref'],
            'true_video': d['ref_video'],
        }
        print(f'✓  {m["file"]}')
    except Exception as e:
        print(f'⚠  {m["file"]} ({e}) — synthetic placeholder')
        rng  = np.random.RandomState(0)
        base = rng.rand(10, 64, 64, 3).astype(np.float32)
        data_cache[m['name']] = {
            'clean':   base,
            'corrupt': np.clip(
                base * 0.35 + rng.rand(10, 64, 64, 3).astype(np.float32) * 0.55,
                0, 1),
            'ref': rng.rand(64, 64, 3).astype(np.float32),
        }

# t_values      = [2, 3, 4, 5, 6, 7]
# frame_indices = [2, 3, 4, 5, 6, 7]

# t_values      = [1, 2, 3, 4, 5]
t_values      = [1, 5, 6, 7, 9]
frame_indices = [0, 4, 5, 6, 8]
# frame_indices = [0, 2, 5, 8, 10]

# FORK_IDX      = 3          # row index where the timeline splits (t = 5)
FORK_IDX      = 2          # row index where the timeline splits (t = 5)
N_ROWS        = len(t_values)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  LAYOUT – more compact, larger vertical gap, narrower time column
# ══════════════════════════════════════════════════════════════════════════════
# fig = plt.figure(figsize=(12.5, 8.2), facecolor=BG_COLOR)
fig = plt.figure(figsize=(12.5, 5.2), facecolor=BG_COLOR)

width_ratios = [0.3,          # time column (reduced from 0.55)
                1, 1, 1,       # method 1: left, center, right
                0.05,          # small gap
                1, 1, 1,       # method 2
                0.05,          # small gap
                1, 1, 1,       # method 3
                0.10,          # gap before injection column for each method?
                0.90]          # injection column – we will reuse this for all methods? No, we need per-method injection.
# Actually, to give each method its own injection frame, we need an extra column per method.
# Simpler: add an injection column right after each method's corrupted column.
# Let's restructure: columns: time, then for each method: [left, center, right, inj_gap, inj]
# That means 1 + 5*3 = 16 columns. We'll adjust offsets accordingly.

# Redefine width_ratios and column offsets per method:
method_cols = 5  # left, center, right, inj_gap, inj
total_cols = 1 + len(METHOD_SPECS) * method_cols
width_ratios = [0.80]  # time column
for i in range(len(METHOD_SPECS)):
    width_ratios += [1, 1, 1, 0.05, 0.80]  # left, center, right, gap, injection
gs = gridspec.GridSpec(
    N_ROWS, total_cols,
    width_ratios=width_ratios,
    wspace=0.02,
    hspace=0.20,
    left=0.02, right=0.98, top=0.91, bottom=0.03,  # reduced left margin
)

# Compute column offsets for each method
col_offset = 1  # start after time column
for m in METHOD_SPECS:
    m['col_offset'] = col_offset
    col_offset += method_cols

# ══════════════════════════════════════════════════════════════════════════════
# 4.  FRAME HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _to_lum(img):
    """Strip batch dims → HxW luminance float in [0, 1]."""
    if   img.ndim == 5: img = img[0, 0]
    elif img.ndim == 4: img = img[0]
    img = np.clip(img.astype(float), 0.0, 1.0)
    if img.ndim == 3:
        if img.shape[2] == 1:
            return img[..., 0]
        elif img.shape[2] == 3:
            return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        else:
            raise ValueError(f"Unsupported channels: {img.shape[2]}")
    return img

def draw_frame(ax, raw, cmap, border_col, border_lw, dashed=False):
    lum = _to_lum(raw)
    im  = ax.imshow(lum, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.axis('off')
    h, w = lum.shape
    box = patches.FancyBboxPatch(
        (-0.5, -0.5), w, h,
        boxstyle=f'round,pad=0,rounding_size={CORNER_RADIUS}',
        linewidth=border_lw, edgecolor=border_col, facecolor='none',
        linestyle=(0, (5, 3)) if dashed else '-',
        transform=ax.transData, clip_on=False, zorder=5,
    )
    ax.add_patch(box)
    im.set_clip_path(box)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN DRAWING LOOP
# ══════════════════════════════════════════════════════════════════════════════
axes_dict = {}

for m in METHOD_SPECS:
    name   = m['name']
    c_off  = m['col_offset']
    lcolor = LABEL_COLS[name]
    cmap   = CMAPS[name]
    vid_c  = data_cache[name]['clean']
    vid_x  = data_cache[name]['corrupt']
    ref    = data_cache[name]['ref']   # each method's own injection reference
    gt_vid = data_cache[name]['true_video']  # ground truth video for this method

    # ── method name above the column group (lowered y) ───────────────────────
    ax_hdr = fig.add_subplot(gs[0, c_off:c_off + 3])  # over left,center,right
    ax_hdr.axis('off')
    ax_hdr.text(0.5, 1.2, m['label'],   # was 1.48, now 0.9
                ha='center', va='bottom',
                # fontsize=20, fontweight='bold' if name == 'Ours' else 'normal'
                fontsize=24, fontweight='normal'
                , color="black",
                # , color=lcolor,
                transform=ax_hdr.transAxes, clip_on=False)

    # ── frames ────────────────────────────────────────────────────────────────
    for r, (t, fi) in enumerate(zip(t_values, frame_indices)):
        if r < FORK_IDX:
            if fi == 0:
                ## We use the true video's first frame
                ax = fig.add_subplot(gs[r, c_off + 1])
                draw_frame(ax, gt_vid[0], cmap, BORDER_COL_CLEAN, BORDER_LW_CLEAN)
                axes_dict[(name, 'shared', r)] = ax
            else:
                # pre‑fork: single centred frame
                ax = fig.add_subplot(gs[r, c_off + 1])
                draw_frame(ax, vid_c[fi], cmap, BORDER_COL_CLEAN, BORDER_LW_CLEAN)
                axes_dict[(name, 'shared', r)] = ax
        else:
            # post‑fork: unperturbed (centre column)
            ax_c = fig.add_subplot(gs[r, c_off + 1])
            draw_frame(ax_c, vid_c[fi], cmap, BORDER_COL_CLEAN, BORDER_LW_CLEAN)
            axes_dict[(name, 'clean', r)] = ax_c

            # post‑fork: corrupted (right column) – REDDISH colormap
            ax_x = fig.add_subplot(gs[r, c_off + 2])
            draw_frame(ax_x, vid_x[fi], RED_CMAP, BORDER_COL_CLEAN, BORDER_LW_CLEAN)
            axes_dict[(name, 'corr', r)] = ax_x

            # sub‑label only on first post‑fork row (danger icon)
            if r == FORK_IDX:
                # ax_x.text(0.5, 1.29, '⚠️ Corrupted',
                ax_x.text(0.5, 1.29, '⚠️',
                          ha='center', va='bottom', fontsize=20,
                          color='#FF0000', fontweight='bold',
                          transform=ax_x.transAxes, clip_on=False)

    # ── L‑shaped fork arrows (grey) ──────────────────────────────────────────
    ax_top = axes_dict[(name, 'shared', FORK_IDX - 1)]
    ax_corr = axes_dict[(name, 'corr', FORK_IDX)]
    fig.add_artist(ConnectionPatch(
        xyA=(0.5, 0.0), xyB=(0.5, 1.0),
        coordsA='axes fraction', coordsB='axes fraction',
        axesA=ax_top, axesB=ax_corr,
        arrowstyle='-|>', mutation_scale=11, lw=2.0,
        color='#999999',
        connectionstyle='angle,angleA=0,angleB=-90,rad=0',
        zorder=10,
    ))

    # ── Alien injection frame for THIS method (rightmost column of its group) ──
    ax_inj = fig.add_subplot(gs[FORK_IDX, c_off + 4])   # injection column
    # draw_frame(ax_inj, ref, RED_CMAP, BORDER_COL_CLEAN, BORDER_LW_CLEAN)
    draw_frame(ax_inj, ref, "grey", BORDER_COL_INJECT, BORDER_LW_INJECT, dashed=False)

    ax_inj.set_title('Alien\nFrame', fontsize=14,
                     color='#FF0000', fontweight=None
                     , pad=5)

    # Arrow from injection frame to the corrupted frame of the same method
    ax_target = axes_dict[(name, 'corr', FORK_IDX)]
    fig.add_artist(ConnectionPatch(
        xyA=(0.0, 0.5), xyB=(1.0, 0.5),
        coordsA='axes fraction', coordsB='axes fraction',
        axesA=ax_inj, axesB=ax_target,
        arrowstyle='-|>', mutation_scale=12, lw=2.0,
        color='#FF0000', linestyle='--',
        connectionstyle='arc3,rad=0',
        zorder=10,
    ))

# ══════════════════════════════════════════════════════════════════════════════
# 6.  TIME AXIS
# ══════════════════════════════════════════════════════════════════════════════
TX = 0.020   # figure x‑fraction for the spine (moved left)
fig.add_artist(plt.Line2D(
    [TX, TX], [0.87, 0.07],
    transform=fig.transFigure, color='#999999', lw=1.6, zorder=1,
))
fig.add_artist(patches.FancyArrowPatch(
    (TX, 0.07), (TX, 0.04),
    transform=fig.transFigure,
    color='#999999', arrowstyle='-|>', mutation_scale=13, lw=1.6,
))
# fig.text(TX, 0.89, '$t$', ha='center', va='bottom',
#          fontsize=12, color='#999999', style='italic')

for r, t in enumerate(t_values):
    ax_t = fig.add_subplot(gs[r, 0])
    ax_t.axis('off')
    is_fork = (r == FORK_IDX)
    ax_t.text(0.90, 0.50, f'$t={t}$',
              ha='right', va='center',
              fontsize=18 if is_fork else 18,
              fontweight='bold' if is_fork else 'normal',
            #   color='#FF0000' if is_fork else '#999999',   # red at fork
              color='#999999',
              transform=ax_t.transAxes)

# ══════════════════════════════════════════════════════════════════════════════
# 7.  SAVE
# ══════════════════════════════════════════════════════════════════════════════
for ext in ('pdf', 'png'):
    plt.savefig(
        f'fork_comparison.{ext}',
        dpi=300, bbox_inches='tight',
        facecolor=BG_COLOR, transparent=False,
    )
    print(f'Saved fork_comparison.{ext}')

plt.show()