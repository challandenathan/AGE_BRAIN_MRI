"""Data forensics: understand the perturbation structure of the dataset.

Goal: find *structural* signatures that separate signal features from the
~600 injected noise features, the ~100 rescaled (~1e6) features, and the
one-sided multiplicative spikes -- so the pipeline can keep exactly the
signal and nothing else, without dropping rows.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis, ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

X = pd.read_csv(DATA / "X_train.csv").drop(columns="id")
Xt = pd.read_csv(DATA / "X_test.csv").drop(columns="id")
y = pd.read_csv(DATA / "y_train.csv")["y"].to_numpy(float)

print(f"X_train {X.shape}, X_test {Xt.shape}, y {y.shape}")
print(f"y: min={y.min()} max={y.max()} mean={y.mean():.2f} std={y.std():.2f}")

Xf = X.to_numpy(float)
Xtf = Xt.to_numpy(float)
n, p = Xf.shape

# --- per-feature basic stats ---
miss = np.isnan(Xf).mean(axis=0)
miss_t = np.isnan(Xtf).mean(axis=0)
med = np.nanmedian(Xf, axis=0)
mad = np.nanmedian(np.abs(Xf - med), axis=0) / 0.6745
q1, q3 = np.nanpercentile(Xf, [25, 75], axis=0)
iqr = q3 - q1
mx = np.nanmax(Xf, axis=0)
mn = np.nanmin(Xf, axis=0)
sk = skew(Xf, axis=0, nan_policy="omit")
ku = kurtosis(Xf, axis=0, nan_policy="omit")

# spike detection: entries that are > 10x the feature's own 99.9th percentile of |non-spike|
# simpler: count entries > med + 20 * robust scale (one-sided high spikes)
spike_hi = ((Xf - med) / np.where(mad > 0, mad, np.inf) > 20).sum(axis=0)
spike_lo = ((Xf - med) / np.where(mad > 0, mad, np.inf) < -20).sum(axis=0)

stats = pd.DataFrame({
    "miss": miss, "miss_test": miss_t, "med": med, "mad": mad, "iqr": iqr,
    "min": mn, "max": mx, "skew": sk, "kurt": ku,
    "spike_hi": spike_hi, "spike_lo": spike_lo,
    "log10med": np.log10(np.abs(med) + 1e-12),
}, index=X.columns)

# correlation of each feature with y (on ranks, robust)
from scipy.stats import spearmanr, pearsonr
rho = np.array([spearmanr(X.iloc[:, j], y).statistic if X.iloc[:, j].notna().sum() > 50 else np.nan
                for j in range(p)])
f_corr = np.array([abs(pearsonr(X.iloc[:, j].dropna(), y[X.iloc[:, j].notna()]).statistic)
                   if X.iloc[:, j].notna().sum() > 50 else np.nan for j in range(p)])
stats["spearman_y"] = rho
stats["abspearson_y"] = f_corr

stats.to_csv(Path(__file__).parent / "results" / "feature_stats.csv")

# --- how many features look "huge scale" (~1e6 block)? ---
big = np.where(np.log10(np.abs(med) + 1e-12) > 5)[0]
print(f"\nfeatures with |median| > 1e5: {len(big)}")
print("  indices:", big[:120])

# --- features with spikes ---
spiky = np.where(spike_hi > 0)[0]
print(f"\nfeatures with high-side spikes (>20 robust sigma): {len(spiky)}")
print("  indices:", spiky[:200])

# --- missingness ---
print(f"\nmissing: train {np.isnan(Xf).mean():.4f}, test {np.isnan(Xtf).mean():.4f}")
missfeat = np.where(miss > 0)[0]
print(f"features with any missing: {len(missfeat)}, max miss rate {miss.max():.3f}")

# --- top features by |spearman| with y ---
order = np.argsort(-np.abs(np.nan_to_num(rho)))
print("\ntop 40 features by |spearman rho| with y:")
for j in order[:40]:
    print(f"  {X.columns[j]:>8} rho={rho[j]:+.3f} pearson={f_corr[j]:+.3f} miss={miss[j]:.3f} "
          f"med={med[j]:.3g} spikes={spike_hi[j]} log10med={np.log10(abs(med[j])+1e-12):.2f}")

# --- correlation structure among features: is there a signal cluster? ---
# use spearman correlation on imputed, rank-transformed data
Xr = pd.DataFrame(Xf).rank().to_numpy()
C = np.corrcoef(Xr, rowvar=False)
np.fill_diagonal(C, 0)
maxcorr = np.nanmax(np.abs(C), axis=1)
mean_top5 = np.sort(np.abs(C), axis=1)[:, -5:].mean(axis=1)
stats["max_abs_corr"] = maxcorr
stats["mean_top5_corr"] = mean_top5

hi = np.where(maxcorr > 0.9)[0]
print(f"\nfeatures with a >=0.9 |corr| partner: {len(hi)}")
print("  indices:", hi[:300])

# histogram of maxcorr
h, edges = np.histogram(maxcorr, bins=20, range=(0, 1))
print("\nmax|corr| histogram:")
for c, e in zip(h, edges):
    print(f"  [{e:.2f},{e+0.05:.2f}): {c}")

# --- are features in contiguous blocks? print maxcorr by index blocks of 50 ---
print("\nblock summary (blocks of 52 features):")
for b in range(0, p, 52):
    sl = slice(b, min(b + 52, p))
    print(f"  x{b:>3}-x{min(b+51,p-1):>3}: maxcorr med={np.median(maxcorr[sl]):.3f} "
          f"|rho_y| med={np.median(np.abs(rho[sl])):.3f} spikes={spike_hi[sl].sum()} "
          f"log10med med={np.median(stats['log10med'].to_numpy()[sl]):.2f} miss={miss[sl].mean():.3f}")

# --- KS test train vs test per feature: which features have different distributions? ---
ks_p = np.array([ks_2samp(Xf[:, j][~np.isnan(Xf[:, j])], Xtf[:, j][~np.isnan(Xtf[:, j])]).pvalue
                 for j in range(p)])
stats["ks_p_train_test"] = ks_p
print(f"\nfeatures with KS p<1e-3 (train vs test distribution differ): {(ks_p < 1e-3).sum()}")

# --- distribution shape of low-|rho| features: uniform? normal? ---
print("\nsample distributions of low-signal features:")
low = order[-30:]
for j in low[:10]:
    v = Xf[:, j][~np.isnan(Xf[:, j])]
    print(f"  {X.columns[j]:>8}: n={len(v)} min={v.min():.4g} max={v.max():.4g} "
          f"mean={v.mean():.4g} std={v.std():.4g} skew={sk[j]:.2f} kurt={ku[j]:.2f}")

stats.to_csv(Path(__file__).parent / "results" / "feature_stats.csv")
print("\nsaved results/feature_stats.csv")
