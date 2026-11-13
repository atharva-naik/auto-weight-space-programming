"""Figures for docs/blog (static PNGs on the blog surface). Palette: dataviz reference slots 1-3, validated on
#fcfcfb with validate_palette.js (all checks pass; aqua < 3:1 contrast, so every line is direct-labelled and the
numbers are also in the blog's tables).

Data sources:
  - original run (G1) and pool-of-1,024 run (G2): their log files were lost with sandbox the sandbox; the values are the
    val checks copied from the live logs into ROUND4_LOG.md (n = 100 val neurons each).
  - replication: runs/round4/sweep_4B/log.jsonl (pulled from the sandbox).

Usage: python src/blog_figures.py   -> docs/blog/fig_learning_curves.png, docs/blog/fig_controls.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, MUTED, RULE, BG = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#e34948"  # diverging pole (blue <-> red, gray midpoint), used for sign: blue = promotes/positive
plt.rcParams.update({"font.family": ["Helvetica", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"], "font.size": 10.5, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": RULE,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})

# only the checks recorded in ROUND4_LOG.md (lines 24 and 29); points that were never logged are left out
G1 = {250: 30, 500: 25, 750: 33, 1000: 29, 1250: 35, 1500: 45, 1750: 60, 2000: 84, 2500: 98, 3000: 98, 3500: 100,
      4000: 100}
G2 = {250: 23, 500: 30, 750: 34, 1000: 36, 1250: 30, 1500: 29, 1750: 26, 2000: 25, 2250: 32, 2500: 32}


def curve_from_log(path):
    if not os.path.exists(path):
        return {}
    return {x["step"]: 100 * x["correct_pmi"] for x in map(json.loads, open(path)) if x["event"] == "val_mid"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="y", color=RULE, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_curves(out):
    rep = curve_from_log("runs/round4/sweep_4B/log.jsonl")
    series = [("Full training pool, original", G1, S1), ("Full training pool, replication", rep, S3),
              ("1,024 neurons revisited", G2, S2)]
    fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=200)
    style(ax)
    ax.axhline(25, color=MUTED, linewidth=1, zorder=1)
    ax.text(68, 21.5, "chance (4-way)", color=MUTED, fontsize=9, ha="right")
    for name, c, col in series:
        if not c:
            continue
        xs = [s * 16 / 1000 for s in sorted(c)]
        ys = [c[s] for s in sorted(c)]
        ax.plot(xs, ys, color=col, linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=3, label=name)
        ax.scatter(xs, ys, s=22, color=col, edgecolor=BG, linewidth=1.2, zorder=4)
        ax.scatter([xs[-1]], [ys[-1]], s=40, color=col, edgecolor=BG, linewidth=1.5, zorder=5)
        dx, dy = {"Full training pool, original": (7, 0), "Full training pool, replication": (-4, 9),
                  "1,024 neurons revisited": (7, 0)}[name]
        ax.annotate(f"{ys[-1]:.0f}%", (xs[-1], ys[-1]), xytext=(dx, dy), textcoords="offset points",
                    color=INK, fontsize=9, va="center", ha="left" if dx > 0 else "right")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Training examples (thousands)")
    ax.set_ylabel("Validation accuracy (%)")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_controls(out):
    """Controls on the locked test split (replication files)."""
    d = "runs/round4/sweep_4B"
    n = json.load(open(f"{d}/eval_stage0_test_neuron_normal.json"))
    sc = json.load(open(f"{d}/eval_stage0_test_neuron_scrambled.json"))
    m = lambda r, k: 100 * sum(x[k] for x in r) / len(r)
    rows = [("Numbers in order", m(n, "correct_pmi")), ("Same numbers, shuffled", m(sc, "correct_pmi")),
            ("Numbers hidden", m(n, "correct_hidden"))]
    fig, ax = plt.subplots(figsize=(7.4, 2.2), dpi=200)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=RULE, linewidth=0.8)
    ys = list(range(len(rows)))[::-1]
    ax.barh(ys, [v for _, v in rows], height=0.42, color=S1, zorder=3)
    for y, (name, v) in zip(ys, rows):
        ax.text(max(v, 25) + 1.5, y, f"{v:.1f}%", va="center", color=INK, fontsize=9.5)
    ax.axvline(25, color=MUTED, linewidth=1, zorder=4)
    ax.text(25, len(rows) - 0.42, "chance", color=MUTED, fontsize=9, ha="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _ in rows], color=INK)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Four-way test accuracy (%)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_slice(out, a):
    """The 128 integers the model sees for one neuron (first test neuron; fixed rule)."""
    v = a["slice_ints"]
    fig, ax = plt.subplots(figsize=(7.4, 2.4), dpi=200)
    style(ax)
    ax.bar(range(len(v)), v, width=0.62, color=S1, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.8, zorder=4)
    ax.set_xlim(-1, len(v))
    ax.set_xticks([0, 32, 64, 96, 127])
    ax.set_xlabel(f"Coordinate (of {a['d']:,})")
    ax.set_ylabel("Value (20 = 1 std)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_lens(out, a):
    """Logit lens of that slice: scores of every vocabulary token, and the top 8 each way."""
    fig = plt.figure(figsize=(7.4, 3.3), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    style(ax)
    c, e = a["score_hist"]["counts"], a["score_hist"]["edges"]
    ax.bar([(e[i] + e[i + 1]) / 2 for i in range(len(c))], c, width=(e[1] - e[0]) * 0.9, color=S1, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Score")
    ax.set_ylabel("Tokens (log scale)")
    ax.set_title(f"All {sum(c):,} tokens", fontsize=9.5, color=INK2, loc="left")
    ax2 = fig.add_subplot(gs[1])
    style(ax2)
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", color=RULE, linewidth=0.8)
    up, dn = a["top_up"], a["top_down"][::-1]
    names = [repr(t)[1:-1] for t, _ in dn] + [repr(t)[1:-1] for t, _ in up[::-1]]
    vals = [x for _, x in dn] + [x for _, x in up[::-1]]
    ys = list(range(len(vals)))
    ax2.barh(ys, vals, height=0.6, color=[RED] * len(dn) + [S1] * len(up), zorder=3)
    ax2.axvline(0, color=INK2, linewidth=0.8, zorder=4)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([f"\"{n}\"" for n in names], fontsize=8.5, color=INK)
    ax2.set_title("Top 8 promoted (blue) and suppressed (red)", fontsize=9.5, color=INK2, loc="left")
    ax2.set_xlabel("Score")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_mech(out, m):
    """Mechanism probe: cosine between the residual stream (shown coordinates only) and the true slice, by layer."""
    fin, base = m["results"]["final"], m["results"]["base"]
    xs = list(range(len(fin["layer_cos_true"])))
    fig, ax = plt.subplots(figsize=(7.4, 3.3), dpi=200)
    style(ax)
    for name, ys, col in [("Trained reader, true neuron's slice", fin["layer_cos_true"], S1),
                          ("Trained reader, distractors' slices", fin["layer_cos_ctrl"], S3),
                          ("Base model, true neuron's slice", base["layer_cos_true"], S2)]:
        ax.plot(xs, ys, color=col, linewidth=2, solid_capstyle="round", zorder=3, label=name)
        ax.scatter([xs[-1]], [ys[-1]], s=40, color=col, edgecolor=BG, linewidth=1.5, zorder=5)
        if col == S1:
            ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points", color=INK,
                        fontsize=9, va="center")
    ax.set_xticks([0, 9, 17, 25, 32])  # index i = residual stream after block i-1 (0 = embeddings)
    ax.set_xticklabels(["embeddings", "8", "16", "24", "31\n(before final norm)"])
    ax.set_xlim(-0.5, 34)
    ax.set_ylim(-0.05, 0.62)
    ax.set_xlabel("Reader's residual stream after block")
    ax.set_ylabel("Alignment with the weight slice (cosine)")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


S4 = "#eda100"


def fig_transfer(out):
    """Compare both scratch budgets with transfer; show early transfer to three slices separately."""
    runs = [
        ("Transfer from the first slice", "sweep_4B_s1280_transfer", S1, "-"),
        ("From scratch · 64k budget", "sweep_4B_s1280", "#b06a00", "--"),
        ("From scratch · 192k budget", "r3_sweep_4B_s1280_192k", "#b34254", "-"),
    ]
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 5.8), dpi=200)
    for ax in axs:
        style(ax)
        ax.axhline(25, color=MUTED, linewidth=.9, linestyle=":", zorder=1)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Validation accuracy (%)")
    for label, run, color, line in runs:
        c = curve_from_log(f"runs/round4/{run}/log.jsonl")
        pts = [(step * 16 / 1000, value) for step, value in sorted(c.items()) if step * 16 <= 96000]
        if pts:
            xs, ys = zip(*pts)
            axs[0].plot(xs, ys, color=color, linestyle=line, linewidth=2, label=label)
    axs[0].set_xlim(0, 98)
    axs[0].set_xticks([0, 16, 32, 48, 64, 80, 96])
    axs[0].set_title("Same new slice: coordinates 1280–1407", loc="left", fontsize=11, pad=12)
    axs[0].set_xlabel("Examples on the new slice (thousands)")
    axs[0].legend(loc="lower right", frameon=True, facecolor=BG, edgecolor="none", framealpha=1, fontsize=8.5)
    axs[0].text(2, 20, "25% chance", fontsize=8, color=INK2)
    for label, run, color, line in [
        ("1280–1407", "sweep_4B_s1280_transfer", S1, "-"),
        ("640–767", "r3_sweep_4B_s640_transfer", "#158057", "--"),
        ("2048–2175", "r3_sweep_4B_s2048_transfer", "#c45b27", ":"),
    ]:
        c = curve_from_log(f"runs/round4/{run}/log.jsonl")
        pts = [(step * 16 / 1000, value) for step, value in sorted(c.items()) if step * 16 <= 24000]
        if pts:
            xs, ys = zip(*pts)
            axs[1].plot(xs, ys, color=color, linestyle=line, linewidth=2, marker="o",
                        markersize=3.5, label=label)
    axs[1].set_xlim(0, 25)
    axs[1].set_xticks([0, 4, 8, 12, 16, 20, 24])
    axs[1].set_title("Transfer to three new slices", loc="left", fontsize=11, pad=12)
    axs[1].set_xlabel("Additional examples after training the first reader (thousands)")
    axs[1].legend(loc="lower right", frameon=True, facecolor=BG, edgecolor="none", framealpha=1,
                  title="New coordinates", fontsize=8.5, title_fontsize=8.5)
    fig.tight_layout(h_pad=1.7)
    fig.savefig(out)
    plt.close(fig)


def fig_rank(out, rows):
    """A dot plot avoids implying a zero baseline on the zoomed rank scale."""
    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=200)
    style(ax)
    xs = [r["start"] for r in rows]
    ys = [r["eff_rank"] for r in rows]
    ax.scatter(xs, ys, s=28, color="#8099b9", zorder=3)
    ax.scatter(xs[:2], ys[:2], s=48, color=S1, zorder=4)
    for i in (0, 1):
        ax.annotate(f"{xs[i]}–{xs[i] + 127}: {ys[i]:.0f}",
                    (xs[i], ys[i]), xytext=(12, -6 if i == 0 else 7),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.axhline(128, color=RULE, linestyle=":", linewidth=1)
    ax.text(2470, 129, "128 coordinates", ha="right", fontsize=8, color=INK2)
    ax.set_xticks([0, 512, 1024, 1536, 2048, 2432])
    ax.set_ylim(80, 133)
    ax.set_xlim(-60, 2500)
    ax.set_xlabel("First coordinate of each 128-coordinate slice")
    ax.set_ylabel("Effective rank")
    ax.set_title("The first two slices have lower effective rank", loc="left", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)



def fig_patch(out, data):
    """Donor-label preference in a separate two-candidate activation-patching test."""
    values = data["interchange_picks_source"]
    labels = ["No patch", "Patch other coordinates", "Patch shown coordinates"]
    vals = [100 * values[k] for k in ("no_patch", "patch_C", "patch_S")]
    fig, ax = plt.subplots(figsize=(7.4, 2.8), dpi=200)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=RULE, linewidth=.8)
    ax.barh([2, 1, 0], vals, height=.48, color=["#aab7c8", "#aab7c8", S1], zorder=3)
    for y, value in zip([2, 1, 0], vals):
        ax.text(value + 1.5, y, f"{value:.0f}%", va="center", fontsize=10, color=INK)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 104)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Pairs where the reader prefers the donor's label (%)")
    ax.set_title(f"Activation swap · {data['n_pairs']} validation pairs", loc="left", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    if os.path.exists("runs/round4/mech_patch_4B.json"):
        fig_patch("docs/blog/fig_patch.png", json.load(open("runs/round4/mech_patch_4B.json")))
    fig_transfer("docs/blog/fig_transfer.png")
    if os.path.exists("runs/round4/slice_hardness.json"):
        fig_rank("docs/blog/fig_rank.png", json.load(open("runs/round4/slice_hardness.json")))
    if os.path.exists("runs/round4/mech_4B_final.json"):
        fig_mech("docs/blog/fig_mech.png", json.load(open("runs/round4/mech_4B_final.json")))
    if os.path.exists("runs/round4/assets_static.json"):
        a = json.load(open("runs/round4/assets_static.json"))
        fig_slice("docs/blog/fig_slice.png", a)
        fig_lens("docs/blog/fig_lens.png", a)
    fig_curves("docs/blog/fig_learning_curves.png")
    fig_controls("docs/blog/fig_controls.png")
    print("wrote docs/blog/fig_learning_curves.png, docs/blog/fig_controls.png")
