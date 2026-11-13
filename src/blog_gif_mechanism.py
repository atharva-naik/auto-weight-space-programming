"""GIF: a possible mechanism for the reader (speculation), drawn on top of the measured results.

Measured (Findings 2.2 and 2.3): the per-layer cosine between coordinates 0-127 of the reader's residual stream at
the answer position and the shown slice (runs/round4/mech_4B_final.json, 200 test items), and the swap test
(runs/round4/mech_patch_4B.json, 100 pairs). Guessed: that the model then reads the rebuilt slice out with its own
final norm and unembedding, the way the logit lens does.

The residual-stream strip is an illustration, not a real activation: for each layer it is a vector built to have
the measured average cosine with the example slice (fixed random noise elsewhere).

Usage: python3 -I src/blog_gif_mechanism.py  -> docs/blog/gif_mechanism.gif (+ _static.png)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blog_gifs import (BLUE, DIV, INK, INK2, MUTED, RED, RULE, FancyArrowPatch, Rectangle, canvas, caption,  # noqa: E402
                       save, strip, to_img, tok)

N = 7


def illustrate(v, c, noise):
    """A vector with cosine c to v (noise is a fixed unit vector orthogonal to v)."""
    vh = v / np.linalg.norm(v)
    return c * vh + np.sqrt(max(0.0, 1 - c * c)) * noise


def layer_bars(ax, ct, cc, upto, mark=None):
    L = len(ct)
    x0, x1, y0, hmax = 10, 90, 4, 13
    bw = (x1 - x0) / L
    for i in range(L):
        x = x0 + i * bw
        ax.add_patch(Rectangle((x, y0), bw * 0.8, hmax, fill=False, edgecolor=RULE, linewidth=0.7, zorder=1))
        if i <= upto:
            ax.add_patch(Rectangle((x, y0), bw * 0.8, max(ct[i], 0) / 0.6 * hmax, color=BLUE, zorder=2))
            ax.add_patch(Rectangle((x, y0), bw * 0.8, max(cc[i], 0) / 0.6 * hmax, color=RED, zorder=3))
    if mark is not None:
        x = x0 + mark * bw + bw * 0.4
        ax.add_patch(FancyArrowPatch((x, y0 + hmax + 3.2), (x, y0 + hmax + 0.6), arrowstyle="-|>", mutation_scale=9,
                                     color=INK))
    ax.text(x0, y0 + hmax + 1.2, "measured cosine with the shown slice, after each block (blue: true slice; red: distractors)",
            fontsize=8.5, color=INK2)
    ax.text(x0, y0 - 2.6, "embeddings", fontsize=8, color=MUTED)
    ax.text(x0 + 17 * bw, y0 - 2.6, "block 16", fontsize=8, color=MUTED)
    ax.text(x1, y0 - 2.6, "block 31 (before final norm)", fontsize=8, color=MUTED, ha="right")


def speculation_tag(ax):
    ax.text(97, 48.5, "SPECULATION", fontsize=9, fontweight="bold", color="#6b4fa0", ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="#efe9fb", ec="#c9b8ee", lw=0.8))


def main():
    a = json.load(open("runs/round4/assets_static.json"))
    m = json.load(open("runs/round4/mech_4B_final.json"))["results"]["final"]
    p = json.load(open("runs/round4/mech_patch_4B.json"))["interchange_picks_source"]
    ct, cc = np.array(m["layer_cos_true"]), np.array(m["layer_cos_ctrl"])
    v = np.array(a["slice_values"], float)
    ints = a["slice_ints"]
    rng = np.random.default_rng(0)
    n = rng.standard_normal(len(v))
    vh = v / np.linalg.norm(v)
    n -= (n @ vh) * vh
    n /= np.linalg.norm(n)
    frames, dur = [], []

    # 1: the input
    fig, ax = canvas()
    caption(ax, f"How the reader might work  (1/{N})",
            "Our guess, built on Findings 2.2 and 2.3. The blue bars and the swap numbers are measured;\n"
            "the story connecting them is not. The reader gets one neuron's 128 numbers as text.")
    speculation_tag(ax)
    ax.text(4, 36, "prompt (shortened)", fontsize=9.5, fontweight="bold", color=INK2)
    ax.text(4, 33.5, "[0] " + " ".join(str(x) for x in ints[:22]) + " …", fontsize=9, family="monospace",
            va="top", color=INK)
    ax.text(4, 29.5, "Through the logit lens, which tokens does this part of the vector promote most?",
            fontsize=9, family="monospace", va="top", color=INK)
    strip(ax, v, 22, 15, 74, 5, label="the same numbers")
    ax.text(22, 12.5, f"coordinates 0–127 of neuron L{a['layer']} N{a['neuron']}  (blue: positive, red: negative)",
            fontsize=8.5, color=MUTED)
    frames.append(to_img(fig)); dur.append(5000)

    # 2-4: layer by layer
    steps = [(8, "Through block 15, coordinates 0–127 of the residual stream (at the answer position) have\n"
                 "nothing to do with the shown numbers."),
             (21, "From block 16 on, they match the shown slice weakly (cosine 0.06–0.21). The true slice is\n"
                  "the closest of the four on only 34–59% of items, so this middle stretch is a faint signal."),
             (32, "The jump comes in the last block (31): cosine 0.54, and the true slice is the closest of the\n"
                  "four on 200 of 200 test items. Even here it is a partial copy, not an exact one.")]
    for k, (i, msg) in enumerate(steps):
        fig, ax = canvas()
        caption(ax, f"How the reader might work  ({k + 2}/{N})", msg)
        speculation_tag(ax)
        strip(ax, v, 22, 34, 74, 4, label="shown slice")
        r = illustrate(v, ct[i], n)
        strip(ax, r, 22, 27, 74, 4, label="residual stream, 0–127")
        ax.text(96, 25, f"cosine {ct[i]:.2f}   (illustration)", fontsize=8.5, color=MUTED, ha="right", va="top")
        layer_bars(ax, ct, cc, i, mark=i)
        frames.append(to_img(fig)); dur.append(3800 if k < 2 else 5200)

    # 5: the read-out (the guessed part)
    fig, ax = canvas()
    caption(ax, f"How the reader might work  (5/{N})",
            "The guess: once the slice sits in coordinates 0–127, the model's own final norm and unembedding\n"
            "turn it into token scores, which is what the logit lens computes. The model would be reusing its\n"
            "own output layer.")
    speculation_tag(ax)
    strip(ax, illustrate(v, ct[-1], n), 4, 30, 26, 5)
    ax.text(4, 27.5, "residual stream, 0–127", fontsize=8.5, color=MUTED)
    ax.add_patch(FancyArrowPatch((31, 32.5), (38, 32.5), arrowstyle="-|>", mutation_scale=12, color=INK2))
    ax.add_patch(Rectangle((39, 28.5), 18, 8, fill=False, edgecolor=INK2, linewidth=1.0))
    ax.text(48, 32.5, "final norm\n× unembedding", fontsize=9, ha="center", va="center", color=INK)
    ax.add_patch(FancyArrowPatch((58, 32.5), (64, 32.5), arrowstyle="-|>", mutation_scale=12, color=INK2))
    top = a["top_up"][:5]
    mx = top[0][1]
    for j, (t, s) in enumerate(top):
        y = 37 - j * 2.6
        ax.add_patch(Rectangle((66, y - 0.9), 14 * s / mx, 1.8, color=BLUE))
        ax.text(81, y, tok(t), fontsize=8.8, va="center", color=INK)
    ax.text(66, 23.5, "highest token scores", fontsize=8.5, color=MUTED)
    ax.text(4, 17, "answer:  PROMOTES: [" + ", ".join(tok(t) for t, _ in a["top_up"][:3]) + ", …",
            fontsize=9.5, color=INK)
    ax.text(4, 10, "A partial copy would explain why picking among four lists is easy (Finding 1.1) but naming\n"
                   "the exact top token is rare (Finding 6.1). We have not tested this read-out step directly.",
            fontsize=9, color=INK2, va="top")
    frames.append(to_img(fig)); dur.append(7000)

    # 6: the swap (measured)
    fig, ax = canvas()
    caption(ax, f"How the reader might work  (6/{N})",
            "What we did test: run the reader on example A, but overwrite coordinates 0–127 at the answer\n"
            "positions of blocks 17–31 with the values from example B's run.")
    speculation_tag(ax)
    vb = np.roll(v, 37) * rng.choice([-1, 1], len(v))
    strip(ax, v, 26, 34, 40, 3.5, label="A's residual, 0–127")
    strip(ax, vb, 26, 28.5, 40, 3.5, label="B's residual, 0–127")
    ax.add_patch(FancyArrowPatch((67, 30.2), (72, 33.5), arrowstyle="-|>", mutation_scale=11, color=INK2,
                                 connectionstyle="arc3,rad=0.3"))
    ax.text(73, 33.2, "copied into A", fontsize=8.5, color=INK2, va="center")
    ax.text(66, 26.6, "(illustration)", fontsize=8, color=MUTED, ha="right")
    bars = [("swap 0–127", p["patch_S"]), ("no swap", p["no_patch"]), ("swap another block", p["patch_C"])]
    for j, (lab, val) in enumerate(bars):
        y = 20 - j * 4.2
        ax.text(25, y, lab, fontsize=9, ha="right", va="center", color=INK2)
        ax.add_patch(Rectangle((26, y - 1.3), 60, 2.6, fill=False, edgecolor=RULE, linewidth=0.7))
        ax.add_patch(Rectangle((26, y - 1.3), 60 * val, 2.6, color=BLUE if j == 0 else MUTED))
        ax.text(87, y, f"{round(val * 100)} of 100", fontsize=9, va="center", color=INK)
    ax.text(26, 24.2, "the reader now prefers B's label", fontsize=8.5, color=MUTED)
    ax.text(4, 6.5, "Limits: the swap includes the last block's output, so it shows where the answer is read from, not\n"
                    "where the reading happens. The other block barely matters to the model, so it is a weak control.",
            fontsize=8.5, color=INK2, va="top")
    frames.append(to_img(fig)); dur.append(6500)

    # 7: summary (also the static fallback)
    fig, ax = canvas()
    caption(ax, f"How the reader might work  (7/{N})", "Summary: what is measured and what is guessed.")
    speculation_tag(ax)
    rows = [("Measured", "A weak, partial match to the shown slice in coordinates 0–127 from block 16 on, and a\n"
                         "strong one after the last block (cosine 0.54; closest of four on 200 of 200 items).", INK),
            ("Measured", "Swapping those 128 coordinates between two examples moves the answer to the other\n"
                         f"example's label on {round(p['patch_S'] * 100)} of 100 pairs ({round(p['no_patch'] * 100)} without the swap, "
                         f"{round(p['patch_C'] * 100)} for another block),\nso the answer is read from those coordinates.", INK),
            ("Guessed", "The model then reads the rebuilt slice out with its own final norm and unembedding, so\n"
                        "it computes the logit lens with the same machinery it uses to predict the next token.", "#6b4fa0")]
    for j, (k, t, col) in enumerate(rows):
        y = 37 - j * 11
        ax.text(4, y, k, fontsize=10, fontweight="bold", color=col, va="top")
        ax.text(17, y, t, fontsize=9.5, color=INK, va="top", linespacing=1.45)
    frames.append(to_img(fig)); dur.append(9000)
    save(frames, dur, "docs/blog/gif_mechanism.gif")


if __name__ == "__main__":
    main()
