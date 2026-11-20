# XAI judgment study materials

Study interface, stimuli, de-identified response data, and analysis code for a study of explanation formats in AI-assisted misinformation evaluation.

## Contents

- `index.html`, `app.js`, `styles.css`: study interface.
- `list1.html`–`list5.html`, `lists.html`: counterbalanced list entry points.
- `data/final/`: fixed stimulus pool, stimulus CSV exports and source lists, task responses for 70 participants, and de-identified questionnaire data.
- `analysis/statistical/`: reproducible statistical analysis, pinned dependencies, and saved model/table outputs.
- `analysis/roberta_fake_news_shap_xai.ipynb`: model and explanation workflow.
- `scripts/`: stimulus-building and model-comparison utilities.
- `api/assign-list.js`: optional deployment list-assignment handler.

## Run the interface

From the repository root, run `python3 -m http.server 8000` and open `http://localhost:8000/index.html`.

The public copy disables remote response collection. Responses can be downloaded locally. This is a demonstration copy of the study interface; it does not reopen participant recruitment. Optional hosted list balancing requires separate deployment configuration.

## Reproduce the statistical analysis

Python 3.9.6 was used for the saved outputs. In a virtual environment, install `analysis/statistical/requirements.txt`, then run:

```sh
python analysis/statistical/verify_statistics.py
```

Inputs are read from `data/final/`; generated results are written to `analysis/statistical/results/`. Saved reference outputs are in `analysis/statistical/verified_outputs/`. Binary mixed models use participant and item intercepts with variational inference; the Brier model uses crossed participant and item intercepts fitted by maximum likelihood.

The fixed stimulus pool is the analysis input; generation utilities are not a substitute for this archived pool. See `analysis/statistical/README.txt` for details of the analysis inputs and models.

The model/explanation workflow has separate dependencies listed in the root `requirements.txt`.
