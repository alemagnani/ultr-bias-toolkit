import logging
from typing import Optional, Literal

import pandas as pd

from ultr_bias_toolkit.bias.intervention_harvesting.util import build_intervention_sets
from ultr_bias_toolkit.bias.intervention_harvesting.util import normalize_bias
from ultr_bias_toolkit.util.assertions import assert_columns_in_df

logger = logging.getLogger(__name__)


class PivotEstimator:
    def __init__(self, pivot_rank: int = 1, weighting: Literal["original", "variance_reduced"] = "original"):
        """
        Initialize the Pivot estimator.
        
        Args:
            pivot_rank: Position to use as pivot (default=1)
            weighting: Weighting scheme to use (default="original")
                - "original": Standard weighting
                - "variance_reduced": Modified weighting that reduces variance while maintaining unbiasedness
        """
        self.pivot_rank = pivot_rank
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
        Estimates position bias using the pivot method.
        
        Args:
            df: DataFrame with click data
            query_col: Name of the column containing query identifiers
            doc_col: Name of the column containing document identifiers
            imps_col: Optional column with impression counts (for aggregated data)
            clicks_col: Optional column with click counts (for aggregated data)
            
        Returns:
            DataFrame with position and examination probability
        """
        logger.info(f"Position bias between rank k and pivot rank: {self.pivot_rank}")
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
        
        # Filter interventions with pivot rank in first positions:
        df = df[df.position_0 == self.pivot_rank]
        
        # Computing CTR ratio between position k and the pivot rank:
        df["examination"] = (df["c_1"] / df["c_0"]).fillna(0)
        df.examination = normalize_bias(df.examination)

        df = df.rename(columns={"position_1": "position"})
        return df[["position", "examination"]]
