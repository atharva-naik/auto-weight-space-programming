# Auto Weight-Space Programming

Code, data splits and result files for **Auto Weight-Space Programming (Part 1): Defining the Problem and Initial
Experiments**, by Atharva Naik, Emmy Liu and Varun Gangal.

Weight-space programming means reading what a model's weights do, or writing weights that make it behave a certain way,
directly from the numbers. Auto weight-space programming asks a model to do this to itself. Part 1 trains Qwen3.5-4B,
with a small LoRA adapter, to apply the logit lens to 128-number slices of its own MLP neurons, shown as plain text.

## Artifacts

- Reader adapters, with the per-item evaluation files behind every number: https://huggingface.co/atharva-naik-1/awsp-readers
- The reading task as text (training pool and the four-way evaluation sets): https://huggingface.co/datasets/atharva-naik-1/awsp-neuron-reading
- The post: https://atharva-naik.github.io/blog/awsp-part-1/ (rendered copy: `docs/blog/minimal.html`, source: `docs/blog/claims.json`)

## Layout

- `src/raw_splits.py`, `src/raw_reader.py`: splits, prompt format, labels, training and evaluation of the readers.
- `src/mech_probe.py`, `src/mech_patch.py`, `src/mech_causal.py`: the residual-stream alignment and swap experiments (Findings 2.2, 2.3).
- `src/slice_hardness.py`: effective rank and rounding checks per slice (Finding 3.4).
- `src/raw_writer.py`, `src/task_reader.py`: the writing, activation and circuit tasks (Findings 7.1-7.3).
- `src/patient.py`, `src/syco_data.py`, `src/levers_*.py`: the sycophancy clamp pilots, attribution patching and probes (Finding 7.4).
- `src/release_datasets.py`, `src/release_upload.py`, `src/release_models.py`: build and upload the released data and adapters.
- `src/blog_claims.py`, `src/review_checks.py`: every number in the post with its source file; `python3 src/blog_claims.py render` recomputes them.
- `src/blog_figures.py`, `src/blog_gifs.py`, `src/blog_gif_mechanism.py`: the figures.
- `data/`: the neuron splits for each model, and the special-token list used for the labels.
- `runs/round4/`: the result files the post's numbers come from (per-item evaluations, training logs, probe outputs).
- `ROUND4_LOG.md`, `PREREG_ROUND4*.md`: the lab log and the protocols written before the replication and scaling runs.

## Reproducing the main result

```
pip install -r requirements.txt
python src/raw_splits.py data/raw_splits.json
python src/raw_reader.py --model Qwen/Qwen3.5-4B --fmt h64 --stages 128:64000 --fixed_start 0 \
    --skip_zero_shot --val_every 250 --openvocab --out runs/reader_4B
```

This trains the rank-32 reader on 64,000 examples (one H100, a few hours) and writes the same evaluation files as
`runs/round4/sweep_4B/`. Other runs change `--model`, `--fixed_start`, `--stages`, `--fmt f3` (raw floats),
`--train_pool` or `--target_model`; each run's `config.json` in the adapters repo lists its exact arguments.

## Notes

- Every result is a single training run (one seed). The lab log is exploratory; it records changes of plan as they happened.
- `ROUND4_LOG.md` is a cleaned copy: infrastructure details (sandbox names and helper scripts) are redacted, and the
  final entries, about an on-policy self-distillation attempt that needs more debugging, are left out. Line numbers
  of the entries the post cites are unchanged.
- A few values exist only in the lab log because their result files were lost with a failed sandbox; the post marks them.

## License

Code: MIT. The released data quotes parts of the Qwen3.5 weights (Apache-2.0) and, for the Llama config, Llama 3.2
weights under the Llama 3.2 Community License.

## Citation

Please cite the series as one work; to point to a specific result, name the part and finding (for example "Part 1,
Finding 2.3"). Series page: https://atharva-naik.github.io/blog/awsp/

```
@misc{naik2026autowsp,
  title        = {Auto Weight-Space Programming},
  author       = {Naik, Atharva and Liu, Emmy and Gangal, Varun},
  year         = {2026},
  howpublished = {Blog series},
  url          = {https://atharva-naik.github.io/blog/awsp/}
}
```
