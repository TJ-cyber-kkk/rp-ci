# Experiments

Scripts for the experiments in the paper. They rerun each experiment with their own
random seeds and reference pools, so results agree with the paper up to Monte Carlo
error rather than exactly.

## Setup

```bash
python -m pip install -e ".[experiments]"
cd experiments
```

The baselines use causal-learn 0.1.4.3 and tigramite 5.2.10.1. All scripts except
`timing.py` accept `--n-jobs` for parallel runs. `cifar.py` and `norman.py` download
their data to `data/` on the first run.

## Commands

Tables 1, 2, 9 and 10; add `--z bounded` for Tables 4 and 5, and `--methods direct`
for direct CIT:

```bash
python synthetic.py --reps 20000 --methods rpcit kci cmiknn rcit max_gcm --topr beta markov --out main.csv
```

Table 6:

```bash
python synthetic.py --models N1 N2 N3 N4 N5 N6 --d 200 --n 200 500 --B 1000000 --reps 2000 --out large_n_null.csv
python synthetic.py --models A1 A2 A3 A4 A5 A6 A7 A8 A9 --d 200 --n 200 500 --B 1000000 --reps 1000 --out large_n_alt.csv
```

Figure 1:

```bash
python synthetic.py --models N1 N3 A1 A2 --d 50 --k-prefixes 5 10 20 50 100 200 500 1000 --reps 5000 --out figure1_n100.csv
python synthetic.py --models A1 A2 --d 50 --n 200 500 --k-prefixes 5 10 20 50 100 200 500 1000 --B 1000000 --reps 2000 --out figure1_large_n.csv
```

Tables 7 and 8:

```bash
python timing.py --out table7.csv
python topr_diagnostic.py --out table8.csv
```

Table 11:

```bash
python rpgcm.py --n 2000 --dz 1 5 10 50 --out table11_a.csv
python rpgcm.py --models N1 N2 --n 500 1000 2000 5000 --dz 50 --out table11_b.csv
```

Tables 3, 12 and 13; Tables 3 and 12 average the rows of `dag.csv`:

```bash
python dag.py --out dag.csv
python dag.py --matched-null --methods rpgcm max_gcm cmiknn rcit --reps 1000 --out table13.csv
python dag.py --matched-null --methods kci --reps 100 --out table13_kci.csv
```

Figures 2 and 3:

```bash
python cifar.py --out cifar.csv
```

Table 14, and with `--budgets $(seq 1 105)` the support curves of Appendix D.8:

```bash
python norman.py --out norman
```
