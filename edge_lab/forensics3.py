"""Forensics round 3: is the noise detectable from *shape alone* (uniform vs
real), how do spikes look exactly, and what is the exact signal-block?"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, kurtosis, skew
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).parent / "results"

X = pd.read_csv(DATA / "X_train.csv").drop(columns="id")
Xt = pd.read_csv(DATA / "X_test.csv").drop(columns="id")
y = pd.read_csv(DATA / "y_train.csv")["y"].to_numpy(float)
Xf = X.to_numpy(float); Xtf = Xt.to_numpy(float)
n, p = Xf.shape

med = np.nanmedian(Xf, axis=0)
Xi = np.where(np.isnan(Xf), med, Xf)
rho = np.array([spearmanr(Xi[:, j], y).statistic if Xi[:, j].std() > 0 else np.nan for j in range(p)])

# kurtosis on non-missing values (spikes inflate kurt; use robust: kurt of values within 5 MAD)
mad = np.nanmedian(np.abs(Xf - med), axis=0) / 0.6745
scale = np.where(mad > 0, mad, np.inf)
keep = np.abs((Xf - med) / scale) < 5
kurt_clean = np.array([kurtosis(Xf[keep[:, j], j]) if keep[:, j].sum() > 100 else np.nan for j in range(p)])
skew_clean = np.array([skew(Xf[keep[:, j], j]) if keep[:, j].sum() > 100 else np.nan for j in range(p)])

# correlation graph
Xr = pd.DataFrame(Xi).rank().to_numpy()
C = np.corrcoef(Xr, rowvar=False); np.fill_diagonal(C, 0.0)
A = np.abs(C)
G = csr_matrix(A > 0.35)
nc, lab = connected_components(G, directed=False)
sz = np.bincount(lab)
big = np.argsort(sz)[-1]
in_block = lab == big
print(f"graph block (|corr|>0.35): {in_block.sum()} features, mean|rho|={np.nanmean(np.abs(rho[in_block])):.3f}")

uniform_like = kurt_clean < -0.7
print(f"uniform-like (kurt<-0.7, clean): {uniform_like.sum()} features")
ct = pd.crosstab(in_block, uniform_like)
print("crosstab in_block x uniform_like:\n", ct)

# how well does uniform-like separate signal?
sig = np.abs(rho) > 0.15
print(f"\n|rho|>0.15: {sig.sum()} features; of them uniform-like: {(sig & uniform_like).sum()}, in block: {(sig & in_block).sum()}")
print(f"uniform-like & in block: {(uniform_like & in_block).sum()}")
noise_def = (~in_block) & uniform_like
print(f"defined noise (not in block & uniform-like): {noise_def.sum()}, their max|rho|={np.nanmax(np.abs(rho[noise_def])):.3f}")

# candidate signal set = block ∪ (uniform-like == False) ∪ high |rho|
sig_set = in_block | (~uniform_like) | (np.abs(rho) > 0.12)
print(f"candidate signal set: {sig_set.sum()} features, min|rho| in set={np.nanmin(np.abs(rho[sig_set])):.3f}")
print(f"outside set: {(~sig_set).sum()} features, max|rho| outside={np.nanmax(np.abs(rho[~sig_set])):.3f}")

# --- spikes: exact structure ---
print("\n--- spike structure (entries beyond 8 robust sigma) ---")
for j in range(p):
    z = (Xf[:, j] - med[j]) / scale[j]
    idx = np.where(np.abs(z) > 8)[0]
    if len(idx):
        vals = Xf[idx, j]
        base = Xf[idx, j] / np.where(med[j] != 0, med[j], np.nan)
        print(f"  {X.columns[j]:>7} n={len(idx)} med={med[j]:.4g} vals={np.array2string(vals[:6], precision=3)} "
              f"val/med={np.array2string(base[:6], precision=2)} rho={rho[j]:+.3f} in_block={in_block[j]}")

# are spike magnitudes consistent with 'multiplied by ~1e17..1e23'?
# check rows: for spiked cells, value/ (value without spike) unknown; but check log10 of max
mx = np.nanmax(Xf, axis=0)
print("\nfeatures with max > 1e10:", np.where(mx > 1e10)[0], "max values:", np.array2string(mx[mx > 1e10][:10], precision=3))

# --- missingness: MCAR? correlation of missing-indicator with y ---
M = np.isnan(Xf)
miss_row = M.mean(axis=1)
from scipy.stats import spearmanr as sp
print(f"\nrow missing rate: mean={miss_row.mean():.4f} std={miss_row.std():.4f}")
print(f"corr(row missing rate, y) spearman={sp(miss_row, y).statistic:.3f}")
col_miss = M.mean(axis=0)
print(f"col missing rate: min={col_miss.min():.3f} max={col_miss.max():.3f} mean={col_miss.mean():.3f}")
print(f"corr(col missing rate, |rho|) = {np.corrcoef(col_miss, np.nan_to_num(np.abs(rho)))[0,1]:.3f}")

# --- y distribution ---
print(f"\ny: {np.percentile(y, [0,1,5,25,50,75,95,99,100])}")

# --- noise feature distribution check on train+test combined: uniform? ---
allX = np.vstack([Xf, Xtf])
med_all = np.nanmedian(allX, axis=0)
for j in [599, 553, 825, 168, 36, 309, 285]:
    v = allX[~np.isnan(allX[:, j]), j]
    q = np.percentile(v, [0, 25, 50, 75, 100])
    # uniformity: compare to uniform on [min,max]: kurtosis ~ -1.2, and quantiles linear
    print(f"  noise? {X.columns[j]:>6} q={np.round(q,3)} kurt={kurtosis(v):.2f}")

# save signal set
np.save(OUT / "signal_set.npy", sig_set)
np.save(OUT / "in_block.npy", in_block)
pd.DataFrame({"col": X.columns, "rho": rho, "kurt_clean": kurt_clean, "skew_clean": skew_clean,
              "in_block": in_block, "signal": sig_set}).to_csv(OUT / "signal_sets.csv", index=False)
print("\nsaved signal_set.npy, in_block.npy, signal_sets.csv")
