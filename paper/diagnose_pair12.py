"""Inventory the position 1->2 interventional set on full KDD."""
import numpy as np
import pandas as pd
from pathlib import Path

KDD = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")

agg = pd.read_parquet(KDD)
print(f"Total cells: {len(agg):,}")
print(f"  position 1 cells: {(agg['position']==1).sum():,}")
print(f"  position 2 cells: {(agg['position']==2).sum():,}")
print(f"  position 3 cells: {(agg['position']==3).sum():,}")

a = agg[agg["position"]==1][["queryID","adID","impressions","clicks"]].rename(columns={"impressions":"N1","clicks":"C1"})
b = agg[agg["position"]==2][["queryID","adID","impressions","clicks"]].rename(columns={"impressions":"N2","clicks":"C2"})
m = a.merge(b, on=["queryID","adID"], how="inner")
print(f"\n(q,ad) pairs at BOTH positions 1 and 2: {len(m):,}")
print(f"Unique queries in this set:               {m['queryID'].nunique():,}")
print(f"Unique ads:                                {m['adID'].nunique():,}")
print(f"Total impressions in this interventional set:")
print(f"  position 1: {m['N1'].sum():>12,}")
print(f"  position 2: {m['N2'].sum():>12,}")
print(f"  combined  : {(m['N1']+m['N2']).sum():>12,}")
print(f"Total clicks:")
print(f"  position 1: {m['C1'].sum():>12,}  (CTR {m['C1'].sum()/m['N1'].sum():.4%})")
print(f"  position 2: {m['C2'].sum():>12,}  (CTR {m['C2'].sum()/m['N2'].sum():.4%})")

# Distribution of impressions per (q, ad) pair
print(f"\nDistribution of N1 (position-1 impressions per pair):")
print(m["N1"].describe(percentiles=[.5,.75,.9,.99,.999]).to_string())
print(f"\nDistribution of N2 (position-2 impressions per pair):")
print(m["N2"].describe(percentiles=[.5,.75,.9,.99,.999]).to_string())
print(f"\nDistribution of N1 + N2 (total impressions per pair):")
print((m["N1"]+m["N2"]).describe(percentiles=[.5,.75,.9,.99,.999]).to_string())

# How many pairs have only 1 impression total?
total = m["N1"] + m["N2"]
print(f"\nPairs with N1+N2 == 2 (i.e., 1 impression each side): {(total==2).sum():,}  ({(total==2).mean()*100:.1f}%)")
print(f"Pairs with N1+N2 <= 5 :  {(total<=5).sum():,}  ({(total<=5).mean()*100:.1f}%)")
print(f"Pairs with N1+N2 >= 100: {(total>=100).sum():,}  ({(total>=100).mean()*100:.2f}%)")
print(f"Pairs with N1+N2 >= 1000: {(total>=1000).sum():,}  ({(total>=1000).mean()*100:.3f}%)")

# Concentration: top-1% pairs control what fraction of total impressions?
sorted_total = np.sort(total.values)[::-1]
cum = np.cumsum(sorted_total)
total_imp = cum[-1]
for frac in [0.001, 0.01, 0.1, 0.5]:
    n_top = int(frac * len(sorted_total))
    if n_top > 0:
        share = cum[n_top-1] / total_imp
        print(f"  Top {frac*100:>5.1f}% pairs ({n_top:>9,} pairs) hold {share*100:>5.1f}% of impressions")
