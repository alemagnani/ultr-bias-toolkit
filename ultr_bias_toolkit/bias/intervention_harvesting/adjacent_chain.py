import logging
from typing import Optional, Literal

import pandas as pd

from ultr_bias_toolkit.bias.intervention_harvesting.util import build_intervention_sets
from ultr_bias_toolkit.util.assertions import assert_columns_in_df


logger = logging.getLogger(__name__)


class AdjacentChainEstimator:
    def __init__(self, weighting: Literal["original", "variance_reduced"] = "original"):
        """
        Initialize the Adjacent Chain estimator.
        
        Args:
            weighting: Weighting scheme to use (default="original")
                - "original": Standard weighting
                - "variance_reduced": Modified weighting that reduces variance while maintaining unbiasedness
        """
        self.weighting = weighting
        
    def __call__(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        doc_col: str = "doc_id",
        imps_col: Optional[str] = None,
        clicks_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Estimates position bias using the adjacent chain method.
        
        Args:
            df: DataFrame with click data
            query_col: Name of the column containing query identifiers
            doc_col: Name of the column containing document identifiers
            imps_col: Optional column with impression counts (for aggregated data)
            clicks_col: Optional column with click counts (for aggregated data)
            
        Returns:
            DataFrame with position and examination probability
        """
        logger.info(f"Position bias between adjacent/neighboring ranks")
        logger.info(f"Using weighting scheme: {self.weighting}")
        
        # Handle different input formats
        if imps_col is not None and clicks_col is not None:
            # Pre-aggregated format
            assert_columns_in_df(df, ["position", query_col, doc_col, imps_col, clicks_col])
            df = build_intervention_sets(
                df, query_col, doc_col, imps_col, clicks_col, 
                weighting=self.weighting
            )
        else:
            # Original binary format
            assert_columns_in_df(df, ["position", query_col, doc_col, "click"])
            df = build_intervention_sets(
                df, query_col, doc_col, 
                weighting=self.weighting
            )
        
        # Filter interventions between adjacent pairs, prepend exam=1.0 for position 1:
        pos_1_df = df[(df.position_0 == 1) & (df.position_1 == 1)]
        adjacent_pair_df = df[df.position_1 == df.position_0 + 1]
        adjacent_pair_df = adjacent_pair_df.sort_values(["position_0", "position_1"])
        df = pd.concat([pos_1_df, adjacent_pair_df])

        # Compute click ratio between neighboring ranks:
        df["examination"] = (df["c_1"] / df["c_0"]).fillna(0)
        df["examination"] = df.examination.cumprod()

        df = df.rename(columns={"position_1": "position"})
        return df[["position", "examination"]].reset_index(drop=True)
