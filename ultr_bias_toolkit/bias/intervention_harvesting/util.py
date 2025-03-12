import pandas as pd
from typing import Optional


def build_intervention_sets(
    df: pd.DataFrame,
    query_col: str,
    doc_col: str,
    imps_col: Optional[str] = None,
    clicks_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build intervention sets for position bias estimation.
    
    Args:
        df: DataFrame with click data
        query_col: Name of the column containing query identifiers
        doc_col: Name of the column containing document identifiers
        imps_col: Optional column with impression counts (for aggregated data)
        clicks_col: Optional column with click counts (for aggregated data)
        
    Returns:
        DataFrame with intervention pairs
    """
    df = df.copy()
    
    # Handle two different input formats:
    # 1. Binary format: Each row is a single impression with click=0/1
    # 2. Aggregated format: Rows contain impression and click counts
    if imps_col is not None and clicks_col is not None:
        # Pre-aggregated format
        return build_intervention_sets_aggregated(df, query_col, doc_col, imps_col, clicks_col)
    else:
        # Original binary format
        return build_intervention_sets_binary(df, query_col, doc_col)


def build_intervention_sets_binary(
    df: pd.DataFrame,
    query_col: str,
    doc_col: str,
) -> pd.DataFrame:
    """Original implementation for binary click data (0/1)"""
    df["no_click"] = 1 - df["click"]
    df = (
        df.groupby([query_col, doc_col, "position"])
        .agg(
            clicks=("click", "sum"),
            no_clicks=("no_click", "sum"),
            impressions=(doc_col, "count"),
        )
        .reset_index()
    )

    df["c"] = df["clicks"] / df["impressions"]
    df["not_c"] = df["no_clicks"] / df["impressions"]
    df = df.merge(df, on=[query_col, doc_col], suffixes=["_0", "_1"])

    df = (
        df.groupby(["position_0", "position_1"])
        .agg(
            c_0=("c_0", "sum"),
            c_1=("c_1", "sum"),
            not_c_0=("not_c_0", "sum"),
            not_c_1=("not_c_1", "sum"),
        )
        .reset_index()
    )

    return df


def build_intervention_sets_aggregated(
    df: pd.DataFrame,
    query_col: str,
    doc_col: str,
    imps_col: str,
    clicks_col: str,
) -> pd.DataFrame:
    """Implementation for pre-aggregated data with impression and click counts"""
    # Validate that clicks <= impressions
    if (df[clicks_col] > df[imps_col]).any():
        raise ValueError(f"Found rows where {clicks_col} > {imps_col}. Clicks must be <= impressions.")
    
    # Get aggregated data with click and no-click rates
    df = _get_aggregated_data(df, query_col, doc_col, imps_col, clicks_col)
    
    # Merge to create intervention pairs
    df = df.merge(df, on=[query_col, doc_col], suffixes=["_0", "_1"])
    
    # Aggregate by position pairs
    df = (
        df.groupby(["position_0", "position_1"])
        .agg(
            c_0=("c_0", "sum"),
            c_1=("c_1", "sum"),
            not_c_0=("not_c_0", "sum"),
            not_c_1=("not_c_1", "sum"),
        )
        .reset_index()
    )
    
    return df


def _get_aggregated_data(
    df: pd.DataFrame,
    query_col: str,
    doc_col: str,
    imps_col: str,
    clicks_col: str,
) -> pd.DataFrame:
    """
    Helper function to aggregate data by query, document, and position.
    Used for testing and by build_intervention_sets_aggregated.
    """
    df = df.copy()
    
    # Calculate no-clicks
    df["no_clicks"] = df[imps_col] - df[clicks_col]
    
    # Group by query, doc, position to handle any remaining aggregation needed
    df = (
        df.groupby([query_col, doc_col, "position"])
        .agg(
            clicks=(clicks_col, "sum"),
            no_clicks=("no_clicks", "sum"),
            impressions=(imps_col, "sum"),
        )
        .reset_index()
    )
    
    # Calculate click and no-click rates
    df["c"] = df["clicks"] / df["impressions"]
    df["not_c"] = df["no_clicks"] / df["impressions"]
    
    return df


def normalize_bias(examination: pd.Series) -> pd.Series:
    examination /= examination.values[0]
    return examination.fillna(0)
