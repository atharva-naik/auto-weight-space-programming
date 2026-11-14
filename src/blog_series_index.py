"""The series landing page, https://atharva-naik.github.io/blog/awsp/ : one entry per part, plus a single citation
for the whole series (the series is cited as one work, not one citation per part).

Usage: python3 -I src/blog_series_index.py OUT_DIR   -> OUT_DIR/index.html
Add a dict to PARTS when a new part is published.
"""
import html
import os
import sys

URL = "https://atharva-naik.github.io/blog/awsp/"
TITLE = "Auto Weight-Space Programming"
AUTHORS = [("Atharva Naik", "Naik, Atharva"), ("Emmy Liu", "Liu, Emmy"), ("Varun Gangal", "Gangal, Varun")]
FIRST_DATE = "2026/11/13"
INTRO = ("Weight-space programming means reading what a model's weights do, or writing weights that make it behave a "
         "certain way, directly from the numbers. This series asks whether a model can learn to do this to itself, which "
         "we call auto weight-space programming. Each part comes with its code, data and models.")

PARTS = [
    {"date": "November 13, 2026", "read": "30 min read", "part": "Part 1",
     "title": "Defining the Problem and Initial Experiments", "href": "../awsp-part-1/", "img": "../awsp-part-1/card.png",
     "summary": ("We formalize weight-space programming and set the goal of auto-WSP. As a toy first step, Qwen3.5-4B "
                 "learns to apply the logit lens to 128-number slices of its own unseen neurons (99.8% on a four-way "
                 "test, chance 25%), while naming the tokens outright, writing weights, tracing circuits and predicting "
                 "behavior are next steps."),
     "tags": ["logit lens", "SFT", "interpretability"],
     "links": [("code", "https://github.com/atharva-naik/auto-weight-space-programming"),
               ("adapters", "https://huggingface.co/atharva-naik-1/awsp-readers"),
               ("data", "https://huggingface.co/datasets/atharva-naik-1/awsp-neuron-reading")]},
    {"date": "Coming next", "read": "", "part": "Part 2", "title": "Benchmarks and Evals for Auto-WSP", "href": "",
     "img": "", "summary": ("Collecting the weight-space programming benchmarks released so far and building a "
                            "templated eval suite for reading and writing tasks, with automated grading."),
     "tags": ["benchmarks", "evals"], "links": []},
]
RELATED = [("November 2, 2026", "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights",
            "Takehi, Teoh, Zhang and Khattab (MIT CSAIL)", "https://rikiyat.github.io/blog/weight-space-programming/")]

e = html.escape
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
CSS = f"""
:root{{--ink:#0b0b0b;--ink2:#52514e;--muted:#6c6a65;--rule:#e1e0d9;--bg:#fcfcfb;--accent:#2a78d6}}
body{{background:var(--bg);color:var(--ink);font:17px/1.6 Georgia,'Times New Roman',serif;margin:0}}
main{{max-width:780px;margin:0 auto;padding:56px 22px 80px}}
a{{color:var(--accent)}}
.kicker{{color:var(--muted);font:600 .8rem {SANS};letter-spacing:.06em;text-transform:uppercase;margin-bottom:.4rem}}
h1{{font:700 2.1rem/1.2 {SANS};margin:0 0 .4rem}}
.byline{{color:var(--muted);font:.95rem {SANS};margin-bottom:1.2rem}}
.intro{{color:var(--ink2);margin:0 0 2.2rem}}
h2{{font:600 1.05rem {SANS};text-transform:uppercase;letter-spacing:.05em;color:var(--ink2);margin:2.4rem 0 .8rem}}
.post{{display:flex;gap:20px;padding:20px 0;border-top:1px solid var(--rule)}}
.post:last-child{{border-bottom:1px solid var(--rule)}}
.post .text{{flex:1;min-width:0}}
.meta{{color:var(--muted);font:.85rem {SANS}}}
.part{{display:inline-block;font:600 .75rem {SANS};color:var(--ink2);border:1px solid var(--rule);border-radius:10px;padding:0 8px;margin-right:6px}}
.post h3{{font:600 1.25rem/1.3 {SANS};margin:.35rem 0 .4rem}}
.post h3 a{{color:var(--ink);text-decoration:none}}.post h3 a:hover{{color:var(--accent)}}
.post p{{margin:0 0 .5rem;color:var(--ink2);font-size:.98rem}}
.tags{{font:.8rem {SANS};color:var(--muted)}}
.links{{font:.85rem {SANS}}}.links a{{margin-right:12px}}
.thumb{{width:210px;flex:none}}.thumb img{{width:100%;border:1px solid var(--rule);border-radius:6px;display:block}}
.upcoming h3{{color:var(--muted)}}
.ext{{font:.75rem {SANS};color:var(--muted);margin-left:4px}}
pre{{background:#f4f3ee;border:1px solid var(--rule);border-radius:6px;padding:12px 14px;font:.82rem/1.5 ui-monospace,Menlo,monospace;overflow-x:auto}}
.note{{color:var(--muted);font:.88rem {SANS}}}
@media (max-width:640px){{.post{{flex-direction:column-reverse}}.thumb{{width:100%}}}}
"""


def page():
    authors = ", ".join(n for n, _ in AUTHORS)
    bib = " and ".join(b for _, b in AUTHORS)
    meta = "".join(f"<meta name='citation_author' content='{e(b)}'>" for _, b in AUTHORS)
    head = (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{TITLE}</title><meta name='description' content='{e(INTRO)}'>"
            f"<meta name='citation_title' content='{TITLE}'>{meta}<meta name='citation_publication_date' content='{FIRST_DATE}'>"
            f"<meta name='citation_abstract_html_url' content='{URL}'>"
            f"<meta property='og:title' content='{TITLE}'><meta property='og:description' content='{e(INTRO)}'>"
            f"<meta property='og:url' content='{URL}'><meta property='og:image' content='https://atharva-naik.github.io/blog/awsp-part-1/card.png'>"
            f"<meta name='twitter:card' content='summary_large_image'><link rel='canonical' href='{URL}'>"
            f"<style>{CSS}</style></head><body><main>")
    out = [head, "<div class='kicker'>Blog series</div>", f"<h1>{TITLE}</h1>", f"<div class='byline'>{e(authors)}</div>",
           f"<p class='intro'>{e(INTRO)}</p>", "<h2>Posts</h2>"]
    for p in PARTS:
        title = e(p["title"])
        h3 = f"<a href='{p['href']}'>{title}</a>" if p["href"] else title
        meta_line = e(p["date"]) + (f" · {e(p['read'])}" if p["read"] else "")
        links = "".join(f"<a href='{u}'>{e(t)}</a>" for t, u in p["links"])
        thumb = f"<a class='thumb' href='{p['href']}'><img src='{p['img']}' alt='{title}' loading='lazy'></a>" if p["img"] else ""
        out.append(f"<div class='post{'' if p['href'] else ' upcoming'}'><div class='text'><div class='meta'>"
                   f"<span class='part'>{e(p['part'])}</span>{meta_line}</div><h3>{h3}</h3><p>{e(p['summary'])}</p>"
                   f"<div class='tags'>{e(', '.join(p['tags']))}</div>" + (f"<div class='links'>{links}</div>" if links else "")
                   + f"</div>{thumb}</div>")
    out.append("<h2>Related</h2>")
    for date, t, who, u in RELATED:
        out.append(f"<div class='post'><div class='text'><div class='meta'>{e(date)}<span class='ext'>[external]</span></div>"
                   f"<h3><a href='{u}'>{e(t)}</a></h3><p>{e(who)}</p></div></div>")
    out.append("<h2>Citation</h2><p class='note'>Please cite the series as one work. To point to a specific result, name "
               "the part and finding, for example \"Part 1, Finding 2.3\".</p>"
               f"<pre>@misc{{naik2026autowsp,\n  title        = {{{TITLE}}},\n  author       = {{{e(bib)}}},\n"
               f"  year         = {{2026}},\n  howpublished = {{Blog series}},\n  url          = {{{URL}}}\n}}</pre>")
    out.append("</main></body></html>")
    return "".join(out)


if __name__ == "__main__":
    d = sys.argv[1]
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "index.html"), "w").write(page())
    print("wrote", os.path.join(d, "index.html"))
