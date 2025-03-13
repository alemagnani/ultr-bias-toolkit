import pandas as pd
import pytest
import numpy as np

from ultr_bias_toolkit.bias.intervention_harvesting.util import (
    build_intervention_sets,
    build_intervention_sets_binary,
    build_intervention_sets_aggregated,
    build_intervention_sets_variance_reduced
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


def test_variance_reduced_weighting_basic():
    """Test the basic functionality of variance-reduced weighting"""
    # Create dataset with imbalanced impressions across positions
    df = pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "doc_id": ["a", "a", "b", "b"],
        "position": [1, 2, 1, 2],
        "impressions": [1000, 100, 500, 2000],  # Large imbalance between positions
        "clicks": [300, 20, 150, 500]
    })
    
    # Get results with original weighting
    original_result = build_intervention_sets(
        df, "query_id", "doc_id", "impressions", "clicks", 
        weighting="original"
    )
    
    # Get results with variance-reduced weighting
    vr_result = build_intervention_sets(
        df, "query_id", "doc_id", "impressions", "clicks", 
        weighting="variance_reduced"
    )
    
    # Sort both results for comparison
    original_result = original_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    vr_result = vr_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    
    # For our specific test case, verify that the results are different (indicating weighting was applied)
    assert not np.allclose(original_result.c_0.values, vr_result.c_0.values)
    assert not np.allclose(original_result.c_1.values, vr_result.c_1.values)
    
    # Check that min weight was applied correctly
    # For doc_id="a", the min impressions is 100, so weight should be 100/1000 = 0.1 for position 1
    # Verify the c_0 value (position_0=1) in variance-reduced result is approximately 0.1 times 
    # the c_0 value in the original result for this specific test case
    original_value = original_result[original_result.position_0 == 1].c_0.iloc[0]
    vr_value = vr_result[vr_result.position_0 == 1].c_0.iloc[0]
    assert np.isclose(vr_value / original_value, 0.1, rtol=0.05)


def test_variance_reduced_weighting_repeated_rows():
    """Test variance-reduced weighting with repeated rows that need aggregation"""
    # Create a dataset with repeated rows for the same query-doc-position combinations
    repeated_df = pd.DataFrame({
        "query_id": [1, 1, 1, 1, 2, 2],
        "doc_id":   ["a", "a", "b", "b", "c", "c"],
        "position": [1, 1, 2, 2, 1, 2],
        "impressions": [800, 200, 300, 100, 500, 600],  # Multiple rows per (query,doc,position)
        "clicks":     [240, 60, 60, 20, 150, 120]
    })
    
    # Create the same dataset but already pre-aggregated
    preaggregated_df = pd.DataFrame({
        "query_id": [1, 1, 2],
        "doc_id":   ["a", "b", "c"],
        "position": [1, 2, 1],  # Only include position 1 for doc "c"
        "impressions": [1000, 400, 500],  # Already aggregated
        "clicks":     [300, 80, 150]
    })
    
    # Add position 2 for doc "c" separately
    preaggregated_extra = pd.DataFrame({
        "query_id": [2],
        "doc_id":   ["c"],
        "position": [2],
        "impressions": [600],
        "clicks":     [120]
    })
    preaggregated_df = pd.concat([preaggregated_df, preaggregated_extra], ignore_index=True)
    
    # Process both datasets with variance-reduced weighting
    repeated_result = build_intervention_sets(
        repeated_df, 
        "query_id", 
        "doc_id", 
        imps_col="impressions", 
        clicks_col="clicks",
        weighting="variance_reduced"
    )
    
    preaggregated_result = build_intervention_sets(
        preaggregated_df, 
        "query_id", 
        "doc_id", 
        imps_col="impressions", 
        clicks_col="clicks",
        weighting="variance_reduced"
    )
    
    # Sort both results for comparison
    repeated_result = repeated_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    preaggregated_result = preaggregated_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    
    # Verify results are the same - this confirms correct aggregation before applying weighting
    pd.testing.assert_frame_equal(repeated_result, preaggregated_result, check_dtype=False)


def test_variance_reduced_weighting_extreme_imbalance():
    """Test variance-reduced weighting with extreme impression imbalances"""
    # Create dataset with extremely imbalanced impressions
    df = pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "doc_id": ["a", "a", "b", "b"],
        "position": [1, 2, 1, 2],
        "impressions": [10000, 10, 20, 5000],  # Extreme imbalance (1000:1 and 1:250)
        "clicks": [3000, 2, 5, 1000]
    })
    
    # Get results with both weighting schemes
    original_result = build_intervention_sets(
        df, "query_id", "doc_id", "impressions", "clicks", 
        weighting="original"
    )
    
    vr_result = build_intervention_sets(
        df, "query_id", "doc_id", "impressions", "clicks", 
        weighting="variance_reduced"
    )
    
    # Sort both results for comparison
    original_result = original_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    vr_result = vr_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    
    # Verify ratio for doc_id="a" at position 1
    # Min weight is 10/10000 = 0.001
    original_value = original_result[
        (original_result.position_0 == 1) & (original_result.position_1 == 2)
    ].c_0.iloc[0]
    vr_value = vr_result[
        (vr_result.position_0 == 1) & (vr_result.position_1 == 2)
    ].c_0.iloc[0]
    assert np.isclose(vr_value / original_value, 0.001, rtol=0.01)
    
    # Verify ratio for doc_id="b" at position 2
    # Min weight is 20/5000 = 0.004
    original_value = original_result[
        (original_result.position_0 == 2) & (original_result.position_1 == 1)
    ].c_0.iloc[0]
    vr_value = vr_result[
        (vr_result.position_0 == 2) & (vr_result.position_1 == 1)
    ].c_0.iloc[0]
    assert np.isclose(vr_value / original_value, 0.004, rtol=0.01)


def test_variance_reduced_weights_calculation():
    """Test that the weight calculations are correct in the internal function"""
    # Create a simple dataset
    df = pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "doc_id": ["a", "a", "b", "b"],
        "position": [1, 2, 1, 2],
        "impressions": [500, 100, 200, 800],
        "clicks": [150, 20, 60, 160]
    })
    
    # Get aggregated data
    aggregated_df = _get_aggregated_data(df, "query_id", "doc_id", "impressions", "clicks")
    
    # Create intervention pairs with standard merging
    merged_df = aggregated_df.merge(aggregated_df, on=["query_id", "doc_id"], suffixes=["_0", "_1"])
    
    # Manually calculate the weights
    merged_df["manual_weight_0"] = np.minimum(merged_df["impressions_0"], merged_df["impressions_1"]) / merged_df["impressions_0"]
    merged_df["manual_weight_1"] = np.minimum(merged_df["impressions_0"], merged_df["impressions_1"]) / merged_df["impressions_1"]
    
    # Call the actual function
    result_df = build_intervention_sets_variance_reduced(df, "query_id", "doc_id", "impressions", "clicks")
    
    # Get the position pairs we want to check
    pos_pairs = [
        (1, 2),  # For query 1, doc a - should be min(500,100)/500 = 0.2 for pos 1
        (2, 1),  # For query 1, doc a - should be min(500,100)/100 = 1.0 for pos 2
        (1, 2),  # For query 2, doc b - should be min(200,800)/200 = 1.0 for pos 1
        (2, 1)   # For query 2, doc b - should be min(200,800)/800 = 0.25 for pos 2
    ]
    
    # Verify expected weights through manual calculation
    for i, (pos0, pos1) in enumerate(pos_pairs):
        query_id = 1 if i < 2 else 2
        doc_id = "a" if i < 2 else "b"
        
        # Find the row in merged_df
        mask = (
            (merged_df.query_id == query_id) & 
            (merged_df.doc_id == doc_id) & 
            (merged_df.position_0 == pos0) & 
            (merged_df.position_1 == pos1)
        )
        row = merged_df[mask]
        
        # Calculate expected c_0 after weighting
        expected_c0 = row.c_0.iloc[0] * row.manual_weight_0.iloc[0]
        expected_c1 = row.c_1.iloc[0] * row.manual_weight_1.iloc[0]
        
        # Find the corresponding row in result_df
        result_mask = (
            (result_df.position_0 == pos0) & 
            (result_df.position_1 == pos1)
        )
        result_row = result_df[result_mask]
        
        # Check that the weighted values were correctly calculated
        assert np.isclose(result_row.c_0.iloc[0], expected_c0, rtol=0.01)
        assert np.isclose(result_row.c_1.iloc[0], expected_c1, rtol=0.01)