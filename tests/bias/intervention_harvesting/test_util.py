import pandas as pd
import pytest
import numpy as np

from ultr_bias_toolkit.bias.intervention_harvesting.util import (
    build_intervention_sets,
    build_intervention_sets_binary,
    build_intervention_sets_aggregated
)


def test_binary_and_aggregated_produces_same_results():
    """Test that both input formats produce identical results"""
    # Create binary format dataset (one row per impression)
    binary_df = pd.DataFrame({
        "query_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "doc_id": ["a", "b", "a", "c", "d", "e", "d", "f"],
        "position": [1, 2, 3, 4, 1, 2, 3, 4],
        "click": [1, 0, 0, 0, 1, 1, 0, 0]
    })
    
    # Create pre-aggregated format dataset
    aggregated_df = pd.DataFrame({
        "query_id": [1, 1, 1, 2, 2, 2],
        "doc_id": ["a", "b", "c", "d", "e", "f"],
        "position": [1, 2, 4, 1, 2, 4],  # Missing position 3 for test
        "impressions": [2, 1, 1, 2, 1, 1],  # 2 impressions for some docs
        "clicks": [1, 0, 0, 1, 1, 0]
    })
    
    # Add the missing position with multiple impressions and clicks
    extra_row = pd.DataFrame({
        "query_id": [1, 2],
        "doc_id": ["a", "d"],
        "position": [3, 3],
        "impressions": [1, 1],
        "clicks": [0, 0]
    })
    
    aggregated_df = pd.concat([aggregated_df, extra_row], ignore_index=True)
    
    # Process both dataframes
    binary_result = build_intervention_sets(binary_df, "query_id", "doc_id")
    aggregated_result = build_intervention_sets(
        aggregated_df, 
        "query_id", 
        "doc_id", 
        imps_col="impressions", 
        clicks_col="clicks"
    )
    
    # Sort both results for comparison
    binary_result = binary_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    aggregated_result = aggregated_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    
    # Verify results are the same
    pd.testing.assert_frame_equal(binary_result, aggregated_result, check_dtype=False)

def test_repeated_rows_in_aggregated_format():
    """Test that repeated query-document-position rows get properly aggregated and
    produce the same results as pre-aggregated data"""
    
    # Create a dataset with repeated rows for the same query-doc-position combinations
    repeated_df = pd.DataFrame({
        "query_id": [1, 1, 1, 1, 1, 2, 2, 2],
        "doc_id":   ["a", "a", "b", "b", "a", "c", "c", "d"],
        "position": [1, 1, 2, 2, 3, 1, 1, 2],
        "impressions": [100, 200, 150, 50, 300, 400, 100, 250],  
        "clicks":     [30,  50,  40,  10, 60,  120, 30,  50]    
    })
    
    # Create the same dataset but already pre-aggregated
    preaggregated_df = pd.DataFrame({
        "query_id": [1, 1, 1, 2, 2],
        "doc_id":   ["a", "b", "a", "c", "d"],
        "position": [1, 2, 3, 1, 2],
        "impressions": [300, 200, 300, 500, 250],  # Already aggregated
        "clicks":     [80,  50,  60,  150, 50]     # Already aggregated
    })
    
    # Process both dataframes
    repeated_result = build_intervention_sets(
        repeated_df, 
        "query_id", 
        "doc_id", 
        imps_col="impressions", 
        clicks_col="clicks"
    )
    
    preaggregated_result = build_intervention_sets(
        preaggregated_df, 
        "query_id", 
        "doc_id", 
        imps_col="impressions", 
        clicks_col="clicks"
    )
    
    # Sort both results for comparison
    repeated_result = repeated_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    preaggregated_result = preaggregated_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    
    # Verify results are the same
    pd.testing.assert_frame_equal(repeated_result, preaggregated_result, check_dtype=False)
    
    # Also verify that the repeated dataset correctly aggregated the counts
    # For the (query_id=1, doc_id=a) the total should be 300 impressions and 80 clicks at position 1
    # Let's check the intermediary data before merging
    intermediary_df = build_intervention_sets_aggregated._get_aggregated_data(repeated_df, "query_id", "doc_id", "impressions", "clicks")
    row = intermediary_df[(intermediary_df.query_id == 1) & (intermediary_df.doc_id == "a") & (intermediary_df.position == 1)]
    assert row.impressions.iloc[0] == 300
    assert row.clicks.iloc[0] == 80


def test_aggregated_input_validation():
    """Test validation that clicks <= impressions"""
    invalid_df = pd.DataFrame({
        "query_id": [1],
        "doc_id": ["a"],
        "position": [1],
        "impressions": [5],
        "clicks": [6]  # More clicks than impressions
    })
    
    with pytest.raises(ValueError):
        build_intervention_sets(
            invalid_df, 
            "query_id", 
            "doc_id", 
            imps_col="impressions", 
            clicks_col="clicks"
        )


def test_aggregation_in_aggregated_format():
    """Test that repeated rows get properly aggregated"""
    # Create dataset with repeated rows that need aggregation
    df = pd.DataFrame({
        "query_id": [1, 1, 1, 1],
        "doc_id": ["a", "a", "b", "b"],
        "position": [1, 1, 2, 2],  # Repeated positions
        "impressions": [3, 2, 5, 3],  # Different impression counts
        "clicks": [2, 1, 3, 1]  # Different click counts
    })
    
    result = build_intervention_sets_aggregated(
        df, "query_id", "doc_id", "impressions", "clicks"
    )
    
    # Should have 2 rows after grouping: position_0=1,position_1=2 and position_0=2,position_1=1
    assert len(result) == 2
    
    # For position_0=1, position_1=2, we should have:
    # - Aggregated 5 impressions at position 1 with 3 clicks (c_0 = 3/5 = 0.6)
    # - Aggregated 8 impressions at position 2 with 4 clicks (c_1 = 4/8 = 0.5)
    row = result[result.position_0 == 1].iloc[0]
    assert np.isclose(row.c_0, 0.6)
    assert np.isclose(row.c_1, 0.5)


def test_large_aggregated_dataset():
    """Test with a larger aggregated dataset that would be inefficient in binary format"""
    # Create dataset with high impression counts that would be inefficient in binary format
    df = pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "doc_id": ["a", "b", "c", "d"],
        "position": [1, 2, 1, 2],
        "impressions": [1000, 1000, 2000, 2000],  # Large impression counts
        "clicks": [300, 200, 500, 250]  # Corresponding click counts
    })
    
    result = build_intervention_sets_aggregated(
        df, "query_id", "doc_id", "impressions", "clicks"
    )
    
    # Verify result shape
    assert len(result) == 2  # Should have 2 rows (position pairs)
    assert "c_0" in result.columns
    assert "c_1" in result.columns
    assert "not_c_0" in result.columns
    assert "not_c_1" in result.columns