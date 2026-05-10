"""Stream training.txt (~150M rows, 9.9GB) and aggregate to
(queryID, adID, position) -> (impressions, clicks).

Original schema:
  click  impression  displayURL  adID  advertiserID  depth  position
  queryID  keywordID  titleID  descriptionID  userID

We want: per-(query, ad, position) total impressions and clicks.

Approach: pandas chunked read (5M rows at a time), groupby+sum, then
combine partial aggregations at the end. Output Parquet.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/track2/training.txt")
OUT  = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")

USECOLS = [0, 1, 3, 6, 7]  # click, impression, adID, position, queryID
NAMES = ["click", "impression", "adID", "position", "queryID"]
CHUNK = 5_000_000


def main():
    print(f"Reading {DATA}, chunks of {CHUNK:,} rows ...")
    t0 = time.time()
    parts = []
    total_rows = 0
    for i, chunk in enumerate(pd.read_csv(
        DATA, sep="\t", header=None, names=[
            "click", "impression", "displayURL", "adID", "advertiserID",
            "depth", "position", "queryID", "keywordID", "titleID",
            "descriptionID", "userID",
        ],
        usecols=["click", "impression", "adID", "position", "queryID"],
        dtype={"click": np.int32, "impression": np.int32, "adID": np.int64,
               "position": np.int8, "queryID": np.int64},
        chunksize=CHUNK,
    )):
        agg = (chunk.groupby(["queryID", "adID", "position"], sort=False)
                    .agg(impressions=("impression", "sum"),
                         clicks=("click", "sum"))
                    .reset_index())
        parts.append(agg)
        total_rows += len(chunk)
        print(f"  chunk {i:2d}: rows so far {total_rows:>11,}, "
              f"partial cells {len(agg):>9,}, elapsed {time.time()-t0:.1f}s")

    print(f"\nCombining {len(parts)} partial aggregations ...")
    big = pd.concat(parts, ignore_index=True)
    print(f"  pre-merge cells: {len(big):,}")
    final = (big.groupby(["queryID", "adID", "position"], sort=False)
                .agg(impressions=("impressions", "sum"),
                     clicks=("clicks", "sum"))
                .reset_index())
    print(f"  post-merge cells: {len(final):,}")
    print(f"  total impressions: {final['impressions'].sum():,}")
    print(f"  total clicks:      {final['clicks'].sum():,}")
    print(f"  CTR: {final['clicks'].sum()/final['impressions'].sum():.4%}")

    final.to_parquet(OUT, index=False)
    print(f"\nSaved {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
