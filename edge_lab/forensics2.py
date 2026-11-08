"""Forensics round 2: impute first, then study correlation structure, signal
clusters, spike features, and the scale-block pattern."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).parent / "results"

X = pd.read_csv(DATA / "X_train.csv").drop(columns="id")
Xt = pd.read_csv(DATA / "X_test.csv").drop(columns="id")
y = pd.read_csv(DATA / "y_train.csv")["y"].to_numpy(float)
Xf = X.to_numpy(float)
n, p = Xf.shape

# median-impute for correlation analysis
med = np.nanmedian(Xf, axis=0)
Xi = np.where(np.isnan(Xf), med, Xf)

# spearman with y per feature
rho = np.array([spearmanr(Xi[:, j], y).statistic for j in range(p)])

# rank-transform then pearson = spearman matrix
Xr = pd.DataFrame(Xi).rank().to_numpy()
C = np.corrcoef(Xr, rowvar=False)
np.fill_diagonal(C, 0.0)
A = np.abs(C)
maxcorr = A.max(axis=1)
top5 = np.sort(A, axis=1)[:, -5:].mean(axis=1)

stats = pd.read_csv(OUT / "feature_stats.csv", index_col=0)
stats["spearman_y"] = rho
stats["max_abs_corr"] = maxcorr
stats["mean_top5_corr"] = top5

order = np.argsort(-np.abs(rho))
print("TOP 50 features by |spearman rho| with y:")
for j in order[:50]:
    print(f"  {X.columns[j]:>7} rho={rho[j]:+.3f} maxcorr={maxcorr[j]:.3f} top5={top5[j]:.3f} "
          f"miss={stats['miss'].iloc[j]:.3f} spikes={int(stats['spike_hi'].iloc[j])} "
          f"log10med={stats['log10med'].iloc[j]:.2f} skew={stats['skew'].iloc[j]:.2f} kurt={stats['kurt'].iloc[j]:.1f}")

print("\nBOTTOM 20 (noise candidates):")
for j in order[-20:]:
    print(f"  {X.columns[j]:>7} rho={rho[j]:+.3f} maxcorr={maxcorr[j]:.3f} top5={top5[j]:.3f} "
          f"log10med={stats['log10med'].iloc[j]:.2f}")

h, edges = np.histogram(maxcorr, bins=20, range=(0, 1))
print("\nmax|corr| histogram (imputed):")
for c, e in zip(h, edges):
    if c: print(f"  [{e:.2f},{e+0.05:.2f}): {c}")

h, edges = np.histogram(top5, bins=20, range=(0, 1))
print("\nmean-top5-|corr| histogram:")
for c, e in zip(h, edges):
    if c: print(f"  [{e:.2f},{e+0.05:.2f}): {c}")

# does the signal live in an index range?
print("\nblock summary (blocks of 52), by |rho|:")
for b in range(0, p, 52):
    sl = slice(b, min(b + 52, p))
    print(f"  x{b:>3}-x{min(b+51,p-1):>3}: |rho| med={np.median(np.abs(rho[sl])):.3f} "
          f"max={np.abs(rho[sl]).max():.3f}  maxcorr med={np.median(maxcorr[sl]):.3f} "
          f"top5 med={np.median(top5[sl]):.3f}  spikes={int(stats['spike_hi'].iloc[sl].sum())}")

# top-40 features: correlation among themselves
top40 = order[:40]
Ct = A[np.ix_(top40, top40)]
print(f"\ncorr among top-40 y-features: mean={Ct[np.triu_indices(40,1)].mean():.3f} "
      f"max={Ct.max():.3f} frac>0.5={np.mean(Ct[np.triu_indices(40,1)]>0.5):.2f}")

# graph: features connected if |corr|>0.5 -- connected components
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
G = csr_matrix(A > 0.5)
ncomp, labels = connected_components(G, directed=False)
sizes = np.bincount(labels)
print(f"\n|corr|>0.5 graph: {ncomp} components, largest sizes: {np.sort(sizes)[-10:]}")
big_comp = np.argsort(sizes)[-1]
members = np.where(labels == big_comp)[0]
print(f"largest component: {len(members)} features, mean|rho|={np.abs(rho[members]).mean():.3f}")
print("  members:", [X.columns[j] for j in members][:60])

# try several thresholds
for thr in [0.3, 0.4, 0.5, 0.6]:
    G = csr_matrix(A > thr)
    nc, lab = connected_components(G, directed=False)
    sz = np.bincount(lab)
    big = np.argsort(sz)[-1]
    mem = np.where(lab == big)[0]
    print(f"thr={thr}: {nc} comps, largest={len(mem)}, mean|rho|={np.abs(rho[mem]).mean():.3f}, "
          f"contains top10 y-feats: {sum(1 for j in order[:10] if lab[j]==big)}/10")

# spike features detail
spiky = np.where(stats["spike_hi"] > 0)[0]
print(f"\nspike features ({len(spiky)}):")
for j in spiky:
    col = Xf[:, j]
    big_vals = col[(col - med[j]) / (stats['mad'].iloc[j] if stats['mad'].iloc[j] > 0 else 1) > 20]
    print(f"  {X.columns[j]:>7} n_spikes={int(stats['spike_hi'].iloc[j])} rho={rho[j]:+.3f} "
          f"spike range=({np.nanmin(big_vals):.3g},{np.nanmax(big_vals):.3g}) med={med[j]:.3g}")

# scale blocks: look at log10|med| sequence to find the ~1e6 rescaled block
lm = stats["log10med"].to_numpy()
rescaled = np.where(lm > 5.5)[0]
print(f"\nrescaled (>~3e5 median) features: {len(rescaled)}")
print("  ", rescaled[:150])

# check if features repeat in patterns: print first 60 log10med
print("\nlog10|med| first 64 features:")
print(np.round(lm[:64], 1))

stats.to_csv(OUT / "feature_stats.csv")
print("\nsaved results/feature_stats.csv")
