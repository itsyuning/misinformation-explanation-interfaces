Reproducible analysis for the IDA 2027 paper

Use Python 3.9.6 and install requirements.txt. From this directory run:
python verify_statistics.py

Inputs are in ../../data/final/. Output snapshots are in verified_outputs/. Regenerated results are written to results/.
The response file contains analysis summaries and derived event fields; provenance markers distinguish derived records from original event observations. Analysis uses the answer summaries.

Binary models use crossed participant/item intercepts and mean-field variational inference. M4 uses crossed participant/item intercepts, maximum likelihood, and full fixed-effect covariance for contrasts. Small variance-component warnings are retained.
