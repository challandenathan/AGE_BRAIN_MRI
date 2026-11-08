"""Forensics round 4: the 278 non-block non-uniform features, block stability
train vs train+test, scale histogram, and spike-log structure."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, kurtosis
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

in_block = np.load(OUT / "in_block.npy")
mad = np.nanmedian(np.abs(Xf - med), axis=0) / 0.6745
scale = np.where(mad > 0, mad, np.inf)
keep = np.abs((Xf - med) / scale) < 5
kurt_clean = np.array([kurtosis(Xf[keep[:, j], j]) if keep[:, j].sum() > 100 else np.nan for j in range(p)])
uniform_like = kurt_clean < -0.7

rest = (~in_block) & (~uniform_like)
print(f"non-block non-uniform features: {rest.sum()}")
print(f"  their |rho|: max={np.nanmax(np.abs(rho[rest])):.3f} p90={np.nanpercentile(np.abs(rho[rest]),90):.3f} "
      f"median={np.nanmedian(np.abs(rho[rest])):.3f}, n with |rho|>0.08: {(np.abs(rho[rest])>0.08).sum()}, >0.10: {(np.abs(rho[rest])>0.10).sum()}")
top_rest = np.argsort(-np.abs(np.nan_to_num(rho)))[:0]
idx = np.where(rest & (np.abs(rho) > 0.06))[0]
print("  those with |rho|>0.06:", [(X.columns[j], round(rho[j],3)) for j in idx][:30])

# block within block: |rho| distribution
print(f"\nblock features: |rho| p10={np.percentile(np.abs(rho[in_block]),10):.3f} p50={np.percentile(np.abs(rho[in_block]),50):.3f} "
      f"min={np.abs(rho[in_block]).min():.3f}, n with |rho|<0.05: {(np.abs(rho[in_block])<0.05).sum()}")

# --- block stability: recompute on train+test combined, and on test alone ---
def block_of(M):
    m = np.nanmedian(M, axis=0)
    Mi = np.where(np.isnan(M), m, M)
    R = pd.DataFrame(Mi).rank().to_numpy()
    Cm = np.corrcoef(R, rowvar=False); np.fill_diagonal(Cm, 0)
    G = csr_matrix(np.abs(Cm) > 0.35)
    nc, lab = connected_components(G, directed=False)
    sz = np.bincount(lab); big = np.argsort(sz)[-1]
    return lab == big

b_tr = in_block
b_all = block_of(np.vstack([Xf, Xtf]))
b_te = block_of(Xtf)
print(f"\nblock sizes: train={b_tr.sum()}, train+test={b_all.sum()}, test-only={b_te.sum()}")
print(f"agreement train vs train+test: {(b_tr == b_all).mean():.3f}")
print(f"train-block features missing from train+test block: {(b_tr & ~b_all).sum()}")
print(f"test-block features not in train block: {(b_te & ~b_tr).sum()}")

# --- scale histogram ---
lm = np.log10(np.abs(med) + 1e-12)
h, e = np.histogram(lm, bins=np.arange(-13, 25, 1))
print("\nlog10|median| histogram:")
for c, lo in zip(h, e):
    if c: print(f"  1e{lo:>3.0f}..1e{lo+1:>3.0f}: {c}")
print(f"features with |med|>1e15: {(np.abs(med)>1e15).sum()}, indices: {np.where(np.abs(med)>1e15)[0]}")

# --- spike log-structure: for spiked cells, log10(val/med) distribution ---
print("\nlog10(spike/med) for cells beyond 8 sigma (high side only):")
lr = []
for j in range(p):
    z = (Xf[:, j] - med[j]) / scale[j]
    idx = np.where(z > 8)[0]
    for i in idx:
        if med[j] > 0 and Xf[i, j] > 0:
            lr.append(np.log10(Xf[i, j] / med[j]))
        elif med[j] < 0 and Xf[i, j] < 0:
            lr.append(np.log10(abs(Xf[i, j] / med[j])))
lr = np.array(lr)
print(f"n={len(lr)}, quantiles log10 ratio: {np.round(np.percentile(lr, [0,25,50,75,100]), 2)}")

# --- does QT-boundary squashing lose the spike info? count spiked cells total ---
tot = sum(int(((Xf[:, j] - med[j]) / scale[j] > 8).sum()) for j in range(p))
print(f"total high-side >8sigma cells: {tot} ({tot/(n*p)*100:.2f}% of all cells)")

# --- y vs top feature nonlinearity: binned means ---
j = np.argmax(np.abs(rho))
print(f"\ntop feature {X.columns[j]} rho={rho[j]:.3f}; binned mean of y by feature decile:")
q = np.nanpercentile(Xf[:, j], np.linspace(0, 100, 11))
b = np.clip(np.digitize(Xf[:, j], q[1:-1]), 0, 9)
print(np.round([y[np.isnan(Xf[:, j]) == False][b[~np.isnan(Xf[:, j])] == k].mean() if (b[~np.isnan(Xf[:, j])]==k).any() else np.nan for k in range(10)], 1))
