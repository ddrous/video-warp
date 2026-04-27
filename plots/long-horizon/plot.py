# !/usr/bin/env python3

#%%
"""
plot.py — Long-horizon forecast visualisation & distributional evaluation.

Usage
-----
  python plot.py --seq 57 --metric wasserstein
  python plot.py --seq 54 --metric jsd --smooth 20
  python plot.py --seq 57 --metric fft --table       # also print LaTeX table
  python plot.py --table --all                        # table over both sequences

Available metrics: wasserstein | jsd | ssim | fft | bhattacharyya
"""

import argparse
import os
import sys
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

import seaborn as sns
sns.set(palette="muted", color_codes=True)
sns.set_theme(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_COLS = {
    "NaiveWM": "#E9C46A",
    "LAPO":    "#06D6A0",
    "Ours":    "#118AB2",
}
DISPLAY_NAMES = {
    "NaiveWM": "Standard WM",
    "LAPO":    "LAPO",
    "Ours":    "NOVA (Ours)",
}
FILE_KEYS = {
    "NaiveWM": "standard",
    "LAPO":    "lapo",
    "Ours":    "nova",
}
METHOD_ORDER = ["NaiveWM", "LAPO", "Ours"]

# ─────────────────────────────────────────────────────────────────────────────
#  MATPLOTLIB GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────

# BG     = "#0c0c0f"
# PANEL  = "#13131a"
# GRID   = "#252530"
# FG     = "#e0e0e8"
# MUTED  = "#777790"


BG     = "#ffffff"  # Pure white background
PANEL  = "#ffffff"  # White axes background
GRID   = "#eaeaef"  # Light gray for grid lines
FG     = "#111111"  # Almost black for text and labels
MUTED  = "#777777"  # Medium gray for ticks

plt.rcParams.update({
    "figure.facecolor":      BG,
    "axes.facecolor":        PANEL,
    "axes.edgecolor":        GRID,
    "axes.labelcolor":       FG,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "xtick.color":           MUTED,
    "ytick.color":           MUTED,
    "xtick.labelsize":       18,
    "ytick.labelsize":       18,
    "grid.color":            GRID,
    "grid.linewidth":        0.6,
    "legend.framealpha":     0.12,
    "legend.edgecolor":      GRID,
    "legend.labelcolor":     FG,
    "legend.fontsize":       16,
    "font.family":           "sans-serif",   # clean, technical feel
    "text.color":            FG,
    "savefig.facecolor":     BG,
    "savefig.dpi":           250,
    "savefig.bbox":          "tight",
})

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_sequence(method_key: str, seq_id: int) -> np.ndarray:
    """Load (T, H, W) float32 array in [0, 1] for a given method/sequence."""
    fname = os.path.join(
        DATA_DIR, f"{FILE_KEYS[method_key]}_ID{seq_id}_T1000.npy"
    )
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"File not found: {fname}\n"
            f"Expected location: {DATA_DIR}\n"
            f"Available files: {[f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]}"
        )
    data = np.load(fname)              # (T, H, W, C) or (T, H, W)
    if data.ndim == 4:
        data = data[..., 0]            # drop channel dim → (T, H, W)
    return data.astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
#  DISTRIBUTIONAL METRICS
#
#  All metrics compare frame t against the reference frame (t = 1).
#  Rationale: a good world model preserves the *identity* of the content
#  (digit style, ink coverage) while allowing arbitrary motion.
#  Pixel-wise L2 would penalise valid motion; histogram / spectral metrics
#  do not, making them suitable proxies for identity consistency.
# ─────────────────────────────────────────────────────────────────────────────

HIST_BINS = 64

def _histp(frame: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(frame.ravel(), bins=HIST_BINS, range=(0.0, 1.0))
    h = h.astype(np.float64) + 1e-12
    return h / h.sum()

def _histv(frame: np.ndarray) -> np.ndarray:
    """Bin centres for Wasserstein (values)."""
    edges = np.linspace(0.0, 1.0, HIST_BINS + 1)
    return 0.5 * (edges[:-1] + edges[1:])

def metric_wasserstein(ref: np.ndarray, frame: np.ndarray) -> float:
    """
    Wasserstein-1 (Earth Mover's Distance) between pixel-intensity
    histograms of the reference frame and frame t.  Invariant to
    spatial permutation (i.e., object motion), sensitive to changes
    in the overall intensity distribution (identity drift).
    """
    v = _histv(ref)
    return float(wasserstein_distance(v, v, u_weights=_histp(ref), v_weights=_histp(frame)))

def metric_jsd(ref: np.ndarray, frame: np.ndarray) -> float:
    """
    Jensen–Shannon Divergence between pixel-intensity histograms.
    Bounded in [0, 1], symmetric, and finite even for zero-support bins.
    """
    return float(jensenshannon(_histp(ref), _histp(frame)))

def metric_ssim(ref: np.ndarray, frame: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM).  Higher is better.
    Note: this *does* capture spatial structure, so motion will slightly
    degrade scores even with preserved identity.
    """
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(ref, frame, data_range=1.0))
    except ImportError:
        # fallback: luminance + contrast components without structure map
        mu1, mu2   = ref.mean(), frame.mean()
        s1, s2     = ref.std(), frame.std()
        cov        = float(np.mean((ref - mu1) * (frame - mu2)))
        C1, C2     = 0.01 ** 2, 0.03 ** 2
        return float(
            ((2 * mu1 * mu2 + C1) * (2 * cov + C2))
            / ((mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2))
        )

def metric_fft(ref: np.ndarray, frame: np.ndarray) -> float:
    """
    Normalised L2 distance between FFT magnitude spectra.
    Translation-invariant by construction: rigid motion of the digit
    does not affect the magnitude spectrum, so persistent deviations
    indicate genuine identity change rather than displacement.
    """
    mag_r = np.abs(np.fft.fft2(ref))
    mag_f = np.abs(np.fft.fft2(frame))
    return float(np.linalg.norm(mag_r - mag_f) / mag_r.size)

def metric_bhattacharyya(ref: np.ndarray, frame: np.ndarray) -> float:
    """
    Bhattacharyya distance between pixel-intensity histograms.
    B(p, q) = -ln( sum_i sqrt(p_i * q_i) ).
    """
    p, q = _histp(ref), _histp(frame)
    bc = np.sum(np.sqrt(p * q))
    return float(-np.log(bc + 1e-15))


METRICS = {
    "wasserstein":   (metric_wasserstein,   "$W_1$",          False),
    "jsd":           (metric_jsd,           "$\\mathrm{JSD}$", False),
    "ssim":          (metric_ssim,          "$\\mathrm{SSIM}$", True),
    "fft":           (metric_fft,           "$d_{\\!\\mathcal{F}}$", False),
    "bhattacharyya": (metric_bhattacharyya, "$B$",             False),
}

def compute_scores(method_key: str, seq_id: int, metric_fn) -> np.ndarray:
    seq = load_sequence(method_key, seq_id)
    ref = seq[0].copy()
    return np.array([metric_fn(ref, seq[t]) for t in range(len(seq))], dtype=np.float64)

def smooth(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 1 — first and final predicted frames
# ─────────────────────────────────────────────────────────────────────────────

def plot_frames(seq_id: int, save_path: str | None = None) -> str:
    n = len(METHOD_ORDER)
    fig = plt.figure(figsize=(2.8 * n, 4.6), facecolor="white")
    outer = gridspec.GridSpec(
        1, n, figure=fig,
        wspace=0.06, left=0.12, right=0.98, top=0.88, bottom=0.02,
    )

    row_labels = [r"$t = 3$", r"$t = 1000$"]
    tick_col = "#555568"

    for col, method in enumerate(METHOD_ORDER):
        try:
            seq = load_sequence(method, seq_id)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}\n  → skipping {method}", file=sys.stderr)
            continue

        frames = [seq[0+2], seq[-1]]
        inner  = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col], hspace=0.05
        )
        col_hex = LABEL_COLS[method]

        for row, frame in enumerate(frames):
            ax = fig.add_subplot(inner[row])
            ax.imshow(frame, cmap="grey", vmin=0.0, vmax=1.0,
                      interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(col_hex)
                spine.set_linewidth(2.2)

            if row == 0:
                ax.set_title(
                    DISPLAY_NAMES[method],
                    color=col_hex, fontsize=20, fontweight="bold",
                    pad=7, fontfamily="sans-serif",
                )
                # pass
            if col == 0:
                ax.text(
                    -0.18, 0.5, row_labels[row],
                    transform=ax.transAxes,
                    color=FG, fontsize=14, va="center", ha="right",
                    fontfamily="sans-serif",
                )

    # fig.suptitle(
    #     f"Predicted frames  ·  Sequence {seq_id}",
    #     color=FG, fontsize=11, fontweight="bold", y=0.97,
    #     fontfamily="sans-serif",
    # )

    out = save_path or os.path.join(DATA_DIR, f"frames_ID{seq_id}.pdf")
    fig.savefig(out, facecolor=fig.get_facecolor(), transparent=False)
    print(f"[plot1] → {out}")
    plt.show()
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 2 — distributional metric over time
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric(
    seq_id: int,
    metric_key: str = "wasserstein",
    smooth_w: int = 15,
    save_path: str | None = None,
) -> str:
    if metric_key not in METRICS:
        raise ValueError(f'Unknown metric "{metric_key}". Choose: {list(METRICS)}')

    fn, ylabel, higher_better = METRICS[metric_key]
    direction = "↑ better" if higher_better else "↓ better"

    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.12)

    for method in METHOD_ORDER:
        try:
            raw = compute_scores(method, seq_id, fn)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}\n  → skipping {method}", file=sys.stderr)
            continue

        s  = smooth(raw, smooth_w)
        ts = np.arange(1, len(s) + 1)
        c  = LABEL_COLS[method]

        ax.plot(ts, s, color=c, linewidth=1.8, label=DISPLAY_NAMES[method],
                alpha=0.95, zorder=3)
        ax.fill_between(ts, s, alpha=0.07, color=c, zorder=2)
        # mark t=1 and t=T
        ax.scatter([1, len(s)], [s[0], s[-1]], color=c, s=28, zorder=5,
                   edgecolors=BG, linewidths=0.8)

    # ── axes decoration ───────────────────────────────────────────────────────
    ax.set_xlabel("Time step $t$", fontsize=18, labelpad=6)
    ax.set_ylabel(ylabel.replace("\\\\", "\\"), fontsize=18, labelpad=8)
    ax.set_xlim(1, 1000)
    ax.yaxis.grid(True, linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # ax.set_title(
    #     f"{ylabel.replace(chr(92)+chr(92), chr(92))}   [{direction}]"
    #     f"   ·   Sequence {seq_id}   ·   smoothing w={smooth_w}",
    #     color=FG, fontsize=10, pad=10, fontfamily="sans-serif",
    # )

    # leg = ax.legend(loc="upper left", framealpha=0.15)
    # for lh in leg.legend_handles:
    #     lh.set_alpha(1.0)

    # Set loc to lower center, push it above the plot with bbox_to_anchor, and use 3 columns
    leg = ax.legend(
        loc="lower center", 
        bbox_to_anchor=(0.5, 1.02), 
        ncol=3, 
        framealpha=0.15,
        borderaxespad=0.
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    # ── small annotation for alien-frame onset ────────────────────────────────
    ax.axvline(6, color="#ff4f4f", linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)
    # ax.text(7, ax.get_ylim()[1] * 0.97, "alien\nframe",
    #         color="#ff4f4f", fontsize=6.5, va="top", alpha=0.75,
    #         fontfamily="sans-serif")

    out = save_path or os.path.join(DATA_DIR, f"metric_{metric_key}_ID{seq_id}.pdf")
    fig.savefig(out)
    print(f"[plot2] → {out}")
    plt.show()
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  TABLE — aggregated statistics for LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def compute_table(seq_ids=(54, 57)) -> None:
    """
    Print a ready-to-paste LaTeX table with mean / min / max of every metric
    for every method, aggregated over the requested sequence IDs.
    """
    print("\n" + "=" * 72)
    print("  LaTeX table  (copy-paste ready)")
    print("=" * 72)

    header = r"""\begin{table}[h]
\centering
\caption{%
  Long-horizon identity-consistency metrics (mean / min / max over
  \(T=1000\) frames and both sequences \(\{54, 57\}\)).
  All metrics are computed w.r.t.\ the reference frame \(t=1\).
  \(\downarrow\) indicates lower is better; \(\uparrow\) higher is better.
}
\label{tab:lh_metrics}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{l l ccc}
\toprule
\textbf{Metric} & \textbf{Method}
  & \textbf{Mean} & \textbf{Min} & \textbf{Max} \\
\midrule"""

    print(header)

    metric_display = {
        "wasserstein":   r"$W_1$ \(\downarrow\)",
        "jsd":           r"$\mathrm{JSD}$ \(\downarrow\)",
        "ssim":          r"$\mathrm{SSIM}$ \(\uparrow\)",
        "fft":           r"$d_{\mathcal{F}}$ \(\downarrow\)",
        "bhattacharyya": r"$B$ \(\downarrow\)",
    }

    first_metric = True
    for mk, mdisplay in metric_display.items():
        fn, _, _ = METRICS[mk]
        first_method = True
        for method in METHOD_ORDER:
            all_scores = []
            for sid in seq_ids:
                try:
                    all_scores.append(compute_scores(method, sid, fn))
                except FileNotFoundError:
                    pass
            if not all_scores:
                continue
            combined = np.concatenate(all_scores)
            mu, lo, hi = combined.mean(), combined.min(), combined.max()

            metric_col = mdisplay if first_method else ""
            fmt = ".4f"
            row = (
                f"{metric_col} & {DISPLAY_NAMES[method]}"
                f" & {mu:{fmt}} & {lo:{fmt}} & {hi:{fmt}} \\\\"
            )
            if not first_method:
                pass
            print(row)
            first_method = False

        if not first_metric:
            pass
        first_metric = False
        print(r"\midrule")

    footer = r"""\bottomrule
\end{tabular}
\end{table}"""
    print(footer)
    print("=" * 72 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
#  LATEX PARAGRAPH (printed to stdout)
# ─────────────────────────────────────────────────────────────────────────────

LATEX_PARA = r"""
% ── Paste this paragraph into your paper ───────────────────────────────────
\paragraph{Evaluation metrics.}
Let $\mathbf{x}_t \in [0,1]^{H \times W}$ denote the predicted frame at
time step $t$, and let $p_t$ be its empirical pixel-intensity distribution,
estimated via a $B{=}64$-bin histogram.
All metrics compare $p_t$ (or $\mathbf{x}_t$) against the reference
$p_1$ (or $\mathbf{x}_1$) to quantify \emph{identity drift}: a faithful
world model should preserve the visual identity of each object even as it
undergoes rigid motion, and pixel-distribution metrics are invariant to
such motion by construction.

\textbf{(i) Wasserstein-1 distance.}
$W_1(p_1, p_t) = \inf_{\gamma \in \Gamma(p_1,\,p_t)}\mathbb{E}_{(u,v)\sim\gamma}|u - v|$,
which, for one-dimensional histograms, reduces to the closed-form integral
of the absolute difference of cumulative distribution functions:
$W_1 = \int_0^1 |F_1(x) - F_t(x)|\,\mathrm{d}x.$

\textbf{(ii) Jensen--Shannon divergence.}
$\mathrm{JSD}(p_1 \| p_t) = \tfrac{1}{2} D_{\mathrm{KL}}(p_1 \| m)
  + \tfrac{1}{2} D_{\mathrm{KL}}(p_t \| m),$
where $m = \tfrac{1}{2}(p_1 + p_t)$ and
$D_{\mathrm{KL}}(p\|q) = \sum_b p_b \ln(p_b / q_b)$.
$\mathrm{JSD} \in [0, \ln 2]$ and is symmetric and finite for all
histogram pairs.

\textbf{(iii) Structural Similarity Index (SSIM).}
$\mathrm{SSIM}(\mathbf{x}_1, \mathbf{x}_t) =
  \frac{(2\mu_1\mu_t + C_1)(2\sigma_{1t} + C_2)}
       {(\mu_1^2 + \mu_t^2 + C_1)(\sigma_1^2 + \sigma_t^2 + C_2)},$
with $C_1 = (0.01)^2$, $C_2 = (0.03)^2$, and luminance, contrast, and
structure computed over a sliding $11{\times}11$ Gaussian window.
Higher values ($\leq 1$) indicate greater structural preservation.
Note that this metric is not fully motion-invariant and is included
as a complementary structural reference.

\textbf{(iv) FFT magnitude distance.}
$d_{\mathcal{F}}(\mathbf{x}_1, \mathbf{x}_t) =
  \frac{1}{HW}\bigl\||\mathcal{F}(\mathbf{x}_1)| -
  |\mathcal{F}(\mathbf{x}_t)|\bigr\|_F,$
where $\mathcal{F}$ denotes the 2-D discrete Fourier transform.
Because the magnitude spectrum is translation-invariant, rigid displacement
of the object does not inflate $d_{\mathcal{F}}$; deviations therefore
reflect genuine structural change in the frequency domain.

\textbf{(v) Bhattacharyya distance.}
$B(p_1, p_t) = -\ln\!\sum_b \sqrt{p_{1,b}\,p_{t,b}},$
which measures the overlap of two distributions; $B = 0$ iff the
distributions are identical, and increases without bound as they diverge.

Aggregated statistics (mean, minimum, maximum) over the full $T{=}1000$
horizon and both test sequences are reported in \Cref{tab:lh_metrics}.
% ────────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Long-horizon forecast evaluation plots.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--seq", type=int, default=57, choices=[54, 57],
        help="Sequence ID to visualise (default: 57)",
    )
    parser.add_argument(
        "--metric", type=str, default="wasserstein",
        choices=list(METRICS),
        help=(
            "Distributional metric for Plot 2\n"
            "  wasserstein   Wasserstein-1 on pixel histograms\n"
            "  jsd           Jensen–Shannon Divergence\n"
            "  ssim          Structural Similarity Index\n"
            "  fft           FFT magnitude distance\n"
            "  bhattacharyya Bhattacharyya distance"
        ),
    )
    parser.add_argument(
        "--smooth", type=int, default=15,
        help="Moving-average window for metric curve (1 = no smoothing; default: 15)",
    )
    parser.add_argument(
        "--table", action="store_true",
        help="Print an aggregated LaTeX table (both sequences)",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Print the LaTeX methods paragraph to stdout",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run both plots for both sequence IDs",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory containing the .npy files (default: same dir as plot.py)",
    )
    # args = parser.parse_args()

    # args = parser.parse_args(args=[])

    # Simulate running: python plot.py --seq 54 --metric jsd --table
    # args = parser.parse_args(args=['--seq', '57', '--metric', 'ssim', '--table'])
    args = parser.parse_args(args=['--seq', '57', '--metric', 'wasserstein', '--table', '--latex'])

    global DATA_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir

    if args.latex:
        print(LATEX_PARA)

    seq_ids = [54, 57] if args.all else [args.seq]
    for sid in seq_ids:
        plot_frames(sid)
        plot_metric(sid, args.metric, args.smooth)

    if args.table:
        compute_table(seq_ids=(54, 57))

    if not (args.latex or args.table) and not args.all:
        # default: also print the paragraph
        pass   # keep clean; user can request with --latex


if __name__ == "__main__":
    main()