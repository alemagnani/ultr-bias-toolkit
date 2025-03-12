import pandas as pd
from typing import Optional

from ultr_bias_toolkit.bias.intervention_harvesting.util import normalize_bias
from ultr_bias_toolkit.util.assertions import assert_columns_in_df


class NaiveCtrEstimator:
    def __call__(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        doc_col: str = "doc_id",
        imps_col: Optional[str] = None,
        clicks_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Estimates position bias using naive CTR method.
        
        Args:
            df: DataFrame with click data
            query_col: Name of the column containing query identifiers
            doc_col: Name of the column containing document identifiers
            imps_col: Optional column with impression counts (for aggregated data)
            clicks_col: Optional column with click counts (for aggregated data)
            
        Returns:
            DataFrame with position and examination probability
        """
        # Handle two different input formats
        if imps_col is not None and clicks_col is not None:
            # Pre-aggregated format
            assert_columns_in_df(df, ["position", query_col, doc_col, imps_col, clicks_col])
            
            # Validate that clicks <= impressions
            if (df[clicks_col] > df[imps_col]).any():
                raise ValueError(f"Found rows where {clicks_col} > {imps_col}. Clicks must be <= impressions.")
            
            # Group by position and calculate CTR
            df = df.groupby("position").agg(
                total_clicks=(clicks_col, "sum"),
                total_impressions=(imps_col, "sum")
            ).reset_index()
            
            df["examination"] = df["total_clicks"] / df["total_impressions"]
            
        else:
            # Original binary format
            assert_columns_in_df(df, ["position", query_col, doc_col, "click"])
            
            df = df.groupby("position").agg(
                examination=("click", "mean")
            ).reset_index()

        df.examination = normalize_bias(df.examination)
        return df[["position", "examination"]]
