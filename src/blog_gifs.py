"""Walkthrough GIFs for the blog, drawn from real data (runs/round4/assets_static.json, mech_4B_final.json).
Style: vectors as coloured cell strips, one step per frame, a running caption. Colour = sign and size of a value
(diverging blue <-> red through gray; blue = positive / promotes).

Usage: python src/blog_gifs.py  -> docs/blog/gif_logit_lens.gif, gif_data.gif, gif_reader.gif
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

INK, INK2, MUTED, RULE, BG = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, RED, GRAY = "#1c5cab", "#c23b3b", "#f0efec"
DIV = LinearSegmentedColormap.from_list("div", [RED, GRAY, BLUE])
plt.rcParams.update({"font.family": ["Helvetica", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"],
                     "text.color": INK, "figure.facecolor": BG})
W, H, DPI = 9.6, 5.0, 100


def canvas():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")
    return fig, ax


def caption(ax, step, text):
    ax.text(3, 48.5, step, fontsize=12, fontweight="bold", color=INK, va="top")
    ax.text(3, 45.3, text, fontsize=10.5, color=INK2, va="top", wrap=True)


def strip(ax, vals, x, y, w, h, vmax=None, edge=True, label=None):
    v = np.asarray(vals, float)
    vmax = vmax or (np.abs(v).max() + 1e-12)
    ax.imshow(v[None, :], cmap=DIV, vmin=-vmax, vmax=vmax, extent=(x, x + w, y, y + h), aspect="auto",
              interpolation="nearest", zorder=2)
    if edge:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=INK2, linewidth=0.8, zorder=3))
    if label:
        ax.text(x - 1, y + h / 2, label, ha="right", va="center", fontsize=9.5, color=INK2)


def to_img(fig):
    fig.canvas.draw()
    im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return im.quantize(colors=128, method=Image.Quantize.MEDIANCUT)


def save(frames, durations, out):
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    frames[-1].convert("RGB").save(out.replace(".gif", "_static.png"))  # reduced-motion fallback
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB")


def tok(t):
    return '"' + repr(t)[1:-1] + '"'


def gif_logit_lens(a, out):
    v = np.array(a["slice_values"])
    g = np.array(a["g_slice"])
    gv = g * v
    rows = a["rows"]
    show = [k for k in rows if rows[k]["kind"] == "random"][:3] + [k for k in rows if rows[k]["kind"] == "up"][:2] + \
           [k for k in rows if rows[k]["kind"] == "down"][:2]
    frames, dur = [], []

    # 1: the vector
    fig, ax = canvas()
    caption(ax, "The logit lens, step by step  (1/7)",
            f"An MLP neuron adds a fixed vector to the residual stream. Here are 128 of its {a['d']:,} numbers\n"
            f"(layer {a['layer']}, neuron {a['neuron']}). Colour shows sign and size: blue positive, red negative.")
    strip(ax, v, 14, 24, 82, 6, label="neuron's\nslice  v")
    ax.text(14, 22, "coordinate 0", fontsize=8.5, color=MUTED)
    ax.text(96, 22, "127", fontsize=8.5, color=MUTED, ha="right")
    frames.append(to_img(fig)); dur.append(3500)

    # 2: times the final-norm scale
    fig, ax = canvas()
    caption(ax, "The logit lens, step by step  (2/7)",
            "The model's last step before predicting a token is its final norm. Multiply each coordinate by the\n"
            "norm's learned scale (1 + w). This is the vector the output layer would see.")
    strip(ax, v, 14, 31, 82, 5, label="v")
    strip(ax, g, 14, 23.5, 82, 5, label="norm scale\n(1 + w)")
    strip(ax, gv, 14, 13, 82, 6, label="scaled  v")
    ax.text(55, 29.6, "×", fontsize=16, ha="center", va="center", color=INK2)
    ax.text(55, 21.3, "=", fontsize=16, ha="center", va="center", color=INK2)
    frames.append(to_img(fig)); dur.append(4000)

    # 3-4: one dot product, done explicitly, for a top token and for a random token (real rows, real scores)
    top_t = [k for k in rows if rows[k]["kind"] == "up"][0]
    rnd_t = [k for k in rows if rows[k]["kind"] == "random"][0]
    for k, t, note in ((3, top_t, "This row lines up with the vector, so most products are positive (blue)."),
                       (4, rnd_t, "A random token's row does not line up: the products cancel out.")):
        fig, ax = canvas()
        caption(ax, f"The logit lens, step by step  ({k}/7)",
                f"Every token has a row in the unembedding matrix. Take the row of {tok(t)}, multiply it with the\n"
                f"scaled vector coordinate by coordinate, and add up the products.\n{note}")
        u = np.array(rows[t]["u"])
        prod = gv * u
        pm = max(abs(gv * np.array(rows[x]["u"])).max() for x in (top_t, rnd_t))
        strip(ax, gv, 20, 30, 70, 4.5, label="scaled  v")
        strip(ax, u, 20, 23, 70, 4.5, label=f"row of {tok(t)}")
        strip(ax, prod, 20, 13, 70, 6, vmax=pm, label="product")
        ax.text(55, 28.6, "×", fontsize=15, ha="center", va="center", color=INK2)
        ax.text(55, 21.0, "=", fontsize=15, ha="center", va="center", color=INK2)
        sc = rows[t]["score"]
        ax.add_patch(Rectangle((20, 7.2), 1.6, 1.6, color=BLUE if sc > 0 else RED))
        ax.text(22.5, 8, f"sum of the products = score of {tok(t)} = {sc:+.4f}", fontsize=10.5, va="center", color=INK)
        frames.append(to_img(fig)); dur.append(5500)

    # 5: many rows at once
    for k, n_rows in ((5, len(show)),):
        fig, ax = canvas()
        caption(ax, f"The logit lens, step by step  ({k}/7)",
                "Same thing for other tokens. Rows that line up with the vector score high; rows that point the\n"
                "opposite way score low; most rows score near zero.")
        strip(ax, gv, 30, 37, 52, 3.4, label="scaled  v")
        vm = max(abs(np.array(rows[s]["u"])).max() for s in show)
        for i, s in enumerate(show[:n_rows]):
            y = 31 - i * 4.2
            strip(ax, rows[s]["u"], 30, y, 52, 3.0, vmax=vm, label=f"row of {tok(s)}")
            sc = rows[s]["score"]
            ax.add_patch(Rectangle((84, y + 0.8), 1.4, 1.4, color=BLUE if sc > 0.01 else (RED if sc < -0.01 else GRAY)))
            ax.text(86.5, y + 1.5, f"{sc:+.4f}", fontsize=10, va="center", color=INK)
        frames.append(to_img(fig)); dur.append(4500)

    # 5: all tokens
    fig, ax = canvas()
    caption(ax, "The logit lens, step by step  (6/7)",
            f"Do this for all {sum(a['score_hist']['counts']):,} tokens we keep. Most scores are near zero. "
            "A few stand out at each end.")
    c, e = np.array(a["score_hist"]["counts"]), np.array(a["score_hist"]["edges"])
    hx = fig.add_axes([0.12, 0.12, 0.8, 0.55])
    hx.bar((e[:-1] + e[1:]) / 2, c, width=(e[1] - e[0]) * 0.9,
           color=[BLUE if m > 0 else RED for m in (e[:-1] + e[1:]) / 2], zorder=3)
    hx.set_yscale("log")
    for sp in ("top", "right"):
        hx.spines[sp].set_visible(False)
    hx.set_xlabel("score", color=INK2)
    hx.set_ylabel("tokens (log scale)", color=INK2)
    hx.tick_params(colors=INK2, length=0)
    hx.set_facecolor(BG)
    frames.append(to_img(fig)); dur.append(4000)

    # 6: the label
    fig, ax = canvas()
    caption(ax, "The logit lens, step by step  (7/7)",
            "The 8 highest and 8 lowest scores are the label. The reader sees only the 128 numbers, rounded to\n"
            "integers, as text, and has to produce these two lists.")
    for j, (t, sc) in enumerate(a["top_up"]):
        y = 33 - j * 4.0
        ax.add_patch(Rectangle((8, y), 20 * sc / a["top_up"][0][1], 3, color=BLUE, zorder=2))
        ax.text(30, y + 1.5, tok(t), fontsize=10.5, va="center")
    for j, (t, sc) in enumerate(a["top_down"]):
        y = 33 - j * 4.0
        ax.add_patch(Rectangle((58, y), 20 * sc / a["top_down"][0][1], 3, color=RED, zorder=2))
        ax.text(80, y + 1.5, tok(t), fontsize=10.5, va="center")
    ax.text(8, 37.5, "PROMOTES", fontsize=10, fontweight="bold", color=INK)
    ax.text(58, 37.5, "SUPPRESSES", fontsize=10, fontweight="bold", color=INK)
    frames.append(to_img(fig)); dur.append(6000)
    save(frames, dur, out)


def gif_data(a, out):
    col = np.array(a["col_ints"], float)
    s0 = a["start"]
    frames, dur = [], []
    fig, ax = canvas()
    caption(ax, "Building one training example  (1/4)",
            f"Start from one neuron's full output vector: {len(col):,} numbers. Pick a training neuron at random for every example.")
    strip(ax, col, 4, 26, 92, 5, edge=True)
    ax.text(4, 23.5, "0", fontsize=8.5, color=MUTED)
    ax.text(96, 23.5, f"{len(col) - 1:,}", fontsize=8.5, color=MUTED, ha="right")
    frames.append(to_img(fig)); dur.append(3500)

    fig, ax = canvas()
    caption(ax, "Building one training example  (2/4)",
            "Keep coordinates 0–127. Scale so that 20 is one standard deviation of the full column, and round.")
    strip(ax, col, 4, 33, 92, 3, edge=True)
    x1 = 4 + 92 * 128 / len(col)
    ax.add_patch(Rectangle((4, 32.3), x1 - 4, 4.4, fill=False, edgecolor=INK, linewidth=1.6, zorder=4))
    ax.add_patch(FancyArrowPatch((x1 / 2 + 2, 32), (30, 24), arrowstyle="-|>", mutation_scale=12, color=MUTED))
    sl = col[s0:s0 + 32]
    for i, x in enumerate(sl):
        cx = 4 + i * 2.9
        ax.add_patch(Rectangle((cx, 14), 2.7, 8, color=DIV(0.5 + 0.5 * x / (np.abs(sl).max() + 1e-9)), zorder=2))
        ax.text(cx + 1.35, 18, f"{int(x)}", fontsize=7, ha="center", va="center", rotation=90, color=INK)
    ax.text(4, 11.5, "first 32 of the 128 integers", fontsize=8.5, color=MUTED)
    frames.append(to_img(fig)); dur.append(4500)

    lines = a["text"].split("\n")
    for k, n_chars in ((3, 140), (4, 140)):
        fig, ax = canvas()
        if k == 3:
            caption(ax, "Building one training example  (3/4)",
                    "Write the integers as text, 64 per line, each line starting with the index of its first coordinate.")
        else:
            caption(ax, "Building one training example  (4/4)",
                    "The target is the slice's logit lens. That is one example. Training uses 64,000 of them,\n"
                    "from 55,580 distinct neurons.")
        cut = lambda l, n=86: l[:n] + (" …" if len(l) > n else "")
        txt = ("Below is part of one of your own weight vectors: the output direction of one\n"
               "MLP neuron (a column of mlp.down_proj) … coordinates 0 to 127 of 2560 …\n\n"
               + "\n".join(cut(l) for l in lines) +
               "\n\nRead these weights. Through the logit lens (your final norm and unembedding),\n"
               "which tokens does this part of the vector promote most and suppress most? …")
        ax.text(4, 38.5, "prompt (shortened)", fontsize=9.5, fontweight="bold", color=INK2)
        ax.text(4, 36.3, txt, fontsize=8.2, family="monospace", va="top", color=INK, linespacing=1.45)
        if k == 4:
            up = ", ".join(tok(t) for t, _ in a["top_up"])
            dn = ", ".join(tok(t) for t, _ in a["top_down"])
            ax.text(4, 12.5, "target", fontsize=9.5, fontweight="bold", color=INK2)
            ax.text(4, 10.3, f"PROMOTES: [{up}]", fontsize=8.8, color=INK, va="top")
            ax.text(4, 6.6, f"SUPPRESSES: [{dn}]", fontsize=8.8, color=INK, va="top")
        frames.append(to_img(fig)); dur.append(4500 if k == 3 else 6500)
    save(frames, dur, out)


def gif_reader(m, out):
    fin, base = m["results"]["final"], m["results"]["base"]
    ct = np.array(fin["layer_cos_true"])
    cc = np.array(fin["layer_cos_ctrl"])
    L = len(ct)
    frames, dur = [], []
    cuts = [1, 9, 17, 21, 25, 29, 32, 33]
    for k, upto in enumerate(cuts):
        fig, ax = canvas()
        if upto <= 17:
            msg = ("The numbers go in as text. At each layer we check whether the residual stream, at the answer\n"
                   "position, points along the neuron's vector (only the 128 shown coordinates).")
        elif upto < 33:
            msg = ("From layer 17 on, the residual stream starts to point along the true neuron's vector, and not\n"
                   "along the 3 distractors'. It partly reconstructs the direction of the vector it read.")
        else:
            msg = ("Before the final norm it is closer to the true neuron's slice than to the distractors' for 200 of\n"
                   "200 test items (cosine 0.54). This measures alignment; it is not a causal test.")
        caption(ax, f"What the trained reader does  ({k + 1}/{len(cuts)})", msg)
        x0, x1, y0 = 8, 92, 9
        bw = (x1 - x0) / L
        for i in range(L):
            x = x0 + i * bw
            ax.add_patch(Rectangle((x, y0), bw * 0.82, 26, fill=False, edgecolor=RULE, linewidth=0.8, zorder=1))
            if i < upto:
                h = max(ct[i], 0) / 0.6 * 26
                ax.add_patch(Rectangle((x, y0), bw * 0.82, h, color=BLUE, zorder=2))
                hc = max(cc[i], 0) / 0.6 * 26
                ax.add_patch(Rectangle((x, y0), bw * 0.82, hc, color=RED, alpha=0.9, zorder=3))
        ax.text(x0, 6, "embeddings", fontsize=8.5, color=MUTED)
        ax.text(x0 + 16 * bw, 6, "layer 16", fontsize=8.5, color=MUTED)
        ax.text(x1, 6, "before final norm", fontsize=8.5, color=MUTED, ha="right")
        ax.text(x0, 37, "cosine with the true neuron's slice (blue) and with distractors' slices (red)",
                fontsize=9, color=INK2)
        if upto >= 33:
            ax.text(x1, y0 + ct[-1] / 0.6 * 26 + 1, f"{ct[-1]:.2f}", fontsize=10, ha="right", color=INK)
        frames.append(to_img(fig)); dur.append(2500 if k < len(cuts) - 1 else 6000)
    save(frames, dur, out)


if __name__ == "__main__":
    a = json.load(open("runs/round4/assets_static.json"))
    gif_logit_lens(a, "docs/blog/gif_logit_lens.gif")
    gif_data(a, "docs/blog/gif_data.gif")
    if os.path.exists("runs/round4/mech_4B_final.json"):
        gif_reader(json.load(open("runs/round4/mech_4B_final.json")), "docs/blog/gif_reader.gif")
