# --- START OF FILE simple_synthetic_mod_final.py ---

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from ultr_bias_toolkit.bias.intervention_harvesting import AdjacentChainEstimator, AllPairsEstimator # Removed PivotEstimator import
import os
import logging # Added for suppressing toolkit info logs if desired

# Optional: Suppress INFO logs from the toolkit if they are too verbose
# logging.getLogger('ultr_bias_toolkit').setLevel(logging.WARNING)


class SimplifiedExperiment:
    """
    Synthetic experiment comparing original and variance-reduced weighting
    for AdjacentChain and AllPairs estimators. Excludes Pivot estimator.
    - Simulates data with induced impression imbalance to highlight variance issues.
    """

    def __init__(
            self,
            num_positions=10,
            num_products_per_pair=20,  # K products per position pair (for Adjacent focus)
            eta=1.0,                   # Position bias steepness parameter
            impressions_t1=100,        # High impressions for ~half the products
            impressions_t2=5,          # Low impressions for ~half the products
            higher_pos_ratio=0.8,      # Increase asymmetry (80/20 split)
            random_seed=42,
            results_dir="images/synthetic_results_final" # Updated directory
    ):
        """Initialize the experiment with configurable parameters."""
        self.num_positions = num_positions
        self.num_products_per_pair = num_products_per_pair
        self.eta = eta
        self.impressions_t1 = impressions_t1
        self.impressions_t2 = impressions_t2
        self.higher_pos_ratio = higher_pos_ratio
        self.random_seed = random_seed
        self.results_dir = results_dir
        self.propensity_col = 'examination' # Column name from estimators

        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Generate true position bias
        self.true_position_bias = np.array([1.0 / (k ** eta) for k in range(1, num_positions + 1)])
        # Normalize to make position 1 have bias 1.0
        self.true_position_bias = self.true_position_bias / self.true_position_bias[0]

        # --- Instantiate only the desired estimators ---
        self.estimators = {
            'AdjacentChain': {
                'original': AdjacentChainEstimator(weighting="original"),
                'modified': AdjacentChainEstimator(weighting="variance_reduced")
            },
            # 'Pivot' estimator is removed
            'AllPairs': {
                'original': AllPairsEstimator(weighting="original"),
                'modified': AllPairsEstimator(weighting="variance_reduced")
            }
        }
        self.estimator_names = list(self.estimators.keys()) # Will now be ['AdjacentChain', 'AllPairs']


    def generate_dataset(self):
        """
        Generate a simplified dataset focused on adjacent positions
        with deliberate impression imbalance. Also includes some non-adjacent
        pairs for AllPairs.
        """
        data = []
        product_id_counter = 0

        # Focus: Create imbalance mainly for adjacent pairs
        for pos in range(1, self.num_positions):
            higher_pos = pos
            lower_pos = pos + 1

            higher_bias = self.true_position_bias[higher_pos - 1]
            lower_bias = self.true_position_bias[lower_pos - 1]

            for k in range(self.num_products_per_pair):
                product_id = product_id_counter
                product_id_counter += 1
                product_relevance = 0.3 + 0.5 * np.random.random()

                # Assign imbalanced total impressions
                total_impressions = self.impressions_t1 if k < self.num_products_per_pair / 2 else self.impressions_t2

                # Split impressions asymmetrically
                higher_impressions = int(total_impressions * self.higher_pos_ratio)
                lower_impressions = total_impressions - higher_impressions

                # Record impression & click data for higher position
                if higher_impressions > 0:
                    click_prob_higher = product_relevance * higher_bias
                    higher_clicks = np.random.binomial(higher_impressions, click_prob_higher)
                    data.append({
                        'query_id': 0, 'item_id': product_id, 'position': higher_pos,
                        'impression': higher_impressions, 'click': higher_clicks,
                        'relevance': product_relevance
                    })

                # Record impression & click data for lower position
                if lower_impressions > 0:
                    click_prob_lower = product_relevance * lower_bias
                    lower_clicks = np.random.binomial(lower_impressions, click_prob_lower)
                    data.append({
                        'query_id': 0, 'item_id': product_id, 'position': lower_pos,
                        'impression': lower_impressions, 'click': lower_clicks,
                        'relevance': product_relevance
                    })

        # Add some data points for non-adjacent pairs for AllPairs to use
        num_extra_pairs_products = 5 # Products per non-adjacent pair
        extra_pairs = [(1, 5), (2, 6), (3, 7)] # Example non-adjacent pairs
        for pos1, pos2 in extra_pairs:
             bias1 = self.true_position_bias[pos1 - 1]
             bias2 = self.true_position_bias[pos2 - 1]
             for _ in range(num_extra_pairs_products): # Add products for this pair
                 product_id = product_id_counter
                 product_id_counter += 1
                 product_relevance = 0.3 + 0.5 * np.random.random()
                 # Give them balanced but maybe fewer impressions
                 total_impressions = (self.impressions_t1 + self.impressions_t2) // 4
                 imp1 = total_impressions // 2
                 imp2 = total_impressions - imp1

                 if imp1 > 0:
                     clicks1 = np.random.binomial(imp1, product_relevance * bias1)
                     data.append({'query_id': 0, 'item_id': product_id, 'position': pos1, 'impression': imp1, 'click': clicks1, 'relevance': product_relevance})
                 if imp2 > 0:
                     clicks2 = np.random.binomial(imp2, product_relevance * bias2)
                     data.append({'query_id': 0, 'item_id': product_id, 'position': pos2, 'impression': imp2, 'click': clicks2, 'relevance': product_relevance})

        return pd.DataFrame(data)

    def run_single_experiment(self):
        """Run a single experiment, return estimates for included estimators."""
        df = self.generate_dataset()
        all_estimates = {}

        # Define columns based on your generated DataFrame and estimator expectations
        doc_col_name = 'item_id'
        query_col_name = 'query_id'
        imps_col_name = 'impression'
        clicks_col_name = 'click'
        # rank_col is NOT passed, estimators expect 'position' column name in df

        for name, estimator_pair in self.estimators.items(): # Iterates over AdjacentChain, AllPairs
            estimates_for_name = {}
            for weighting_type, estimator in estimator_pair.items():
                try:
                    # Call estimator (no rank_col argument needed)
                    result_df = estimator(df,
                                          doc_col=doc_col_name,
                                          query_col=query_col_name,
                                          imps_col=imps_col_name,
                                          clicks_col=clicks_col_name)

                    # Handle output format (dict vs dataframe)
                    if isinstance(result_df, pd.DataFrame) and self.propensity_col in result_df.columns:
                       estimates_for_name[weighting_type] = result_df[self.propensity_col].values
                    elif isinstance(result_df, dict) and self.propensity_col in result_df:
                         estimates_for_name[weighting_type] = result_df[self.propensity_col]
                    else:
                        raise TypeError(f"Unexpected or incomplete output from {name} {weighting_type}: type={type(result_df)}, keys/cols={result_df.keys() if isinstance(result_df, dict) else result_df.columns}")

                    # Check for correct length and pad if necessary
                    if len(estimates_for_name[weighting_type]) != self.num_positions:
                         print(f"Warning: Estimator {name} {weighting_type} produced {len(estimates_for_name[weighting_type])} estimates, expected {self.num_positions}. Padding with NaN.")
                         padded_estimates = np.full(self.num_positions, np.nan)
                         len_to_copy = min(len(estimates_for_name[weighting_type]), self.num_positions)
                         padded_estimates[:len_to_copy] = estimates_for_name[weighting_type][:len_to_copy]
                         estimates_for_name[weighting_type] = padded_estimates

                except Exception as e:
                    print(f"ERROR: Estimator {name} ({weighting_type}) failed during call: {e}")
                    # Return NaNs if estimator fails
                    estimates_for_name[weighting_type] = np.full(self.num_positions, np.nan)
            all_estimates[name] = estimates_for_name

        return all_estimates, df

    def run_multiple_experiments(self, num_iterations=100):
        """Run multiple iterations to compute means and variances."""
        # Initialize results structure for included estimators
        results = {name: {'original': np.zeros((num_iterations, self.num_positions)),
                          'modified': np.zeros((num_iterations, self.num_positions))}
                   for name in self.estimator_names} # Only AdjacentChain, AllPairs

        for i in tqdm(range(num_iterations), desc=f"Running {num_iterations} iterations"):
            np.random.seed(self.random_seed + i)
            iteration_estimates, _ = self.run_single_experiment()

            # Store results, handling potential NaNs
            for name in self.estimator_names: # Only AdjacentChain, AllPairs
                 if name in iteration_estimates:
                     results[name]['original'][i, :] = iteration_estimates[name].get('original', np.full(self.num_positions, np.nan))
                     results[name]['modified'][i, :] = iteration_estimates[name].get('modified', np.full(self.num_positions, np.nan))
                 else:
                     results[name]['original'][i, :].fill(np.nan)
                     results[name]['modified'][i, :].fill(np.nan)
        return results

    def plot_and_analyze_results(self, results):
        """Plot comparisons and print analysis for each included estimator type."""
        positions = np.arange(1, self.num_positions + 1)
        summary_stats = {} # To store results for final summary table

        print("\n--- Analysis Results ---") # Clearer section header

        for name in self.estimator_names: # Only AdjacentChain, AllPairs
            print(f"\n--- Analyzing Estimator: {name} ---")
            res_orig = results[name]['original']
            res_mod = results[name]['modified']

            # Compute mean, variance, CI (using nanmean/nanvar)
            original_mean = np.nanmean(res_orig, axis=0)
            original_var = np.nanvar(res_orig, axis=0)
            original_std = np.nanstd(res_orig, axis=0)
            valid_counts_orig = np.sum(~np.isnan(res_orig), axis=0)
            original_ci = 1.96 * original_std / np.sqrt(np.maximum(valid_counts_orig, 1))

            modified_mean = np.nanmean(res_mod, axis=0)
            modified_var = np.nanvar(res_mod, axis=0)
            modified_std = np.nanstd(res_mod, axis=0)
            valid_counts_mod = np.sum(~np.isnan(res_mod), axis=0)
            modified_ci = 1.96 * modified_std / np.sqrt(np.maximum(valid_counts_mod, 1))

            # --- Plot 1: Mean Propensity Estimates with CI ---
            fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))
            ax1.plot(positions, self.true_position_bias, 'k-', label='Ground Truth', linewidth=2, zorder=5)
            ax1.errorbar(positions, original_mean, yerr=original_ci, fmt='ro-', label='Original Weighting', alpha=0.7, capsize=3, markersize=5)
            ax1.errorbar(positions, modified_mean, yerr=modified_ci, fmt='bo-', label='Modified Weighting', alpha=0.7, capsize=3, markersize=5)
            ax1.set_xlabel('Position', fontsize=14)
            ax1.set_ylabel('Relative Propensity (Normalized)', fontsize=14)
            ax1.set_title(f'{name} Estimator: Propensity Estimates (Avg over {res_orig.shape[0]} runs)', fontsize=16)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.4)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()
            plot_filename1 = os.path.join(self.results_dir, f"synthetic_{name}_estimates.png")
            plt.savefig(plot_filename1)
            print(f"Saved estimates plot: {plot_filename1}")
            plt.close(fig1)

            # --- Plot 2: Variance Comparison ---
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))
            bar_width = 0.35
            ax2.bar(positions - bar_width/2, original_var, width=bar_width, color='r', alpha=0.7, label='Original Weighting')
            ax2.bar(positions + bar_width/2, modified_var, width=bar_width, color='b', alpha=0.7, label='Modified Weighting')
            ax2.set_xlabel('Position', fontsize=14)
            ax2.set_ylabel('Variance', fontsize=14)
            ax2.set_title(f'{name} Estimator: Variance Comparison', fontsize=16)
            ax2.legend(fontsize=12)
            ax2.grid(True, axis='y', alpha=0.4)
            ax2.tick_params(axis='both', which='major', labelsize=12)

            # Use log scale if variances differ greatly AND are positive
            min_plot_var = min(np.nanmin(original_var[np.isfinite(original_var) & (original_var > 0)]) if np.any(np.isfinite(original_var) & (original_var > 0)) else 1,
                               np.nanmin(modified_var[np.isfinite(modified_var) & (modified_var > 0)]) if np.any(np.isfinite(modified_var) & (modified_var > 0)) else 1)
            max_plot_var = max(np.nanmax(original_var[np.isfinite(original_var)]) if np.any(np.isfinite(original_var)) else 0,
                               np.nanmax(modified_var[np.isfinite(modified_var)]) if np.any(np.isfinite(modified_var)) else 0)

            if max_plot_var / min_plot_var > 100 and min_plot_var > 0: # Check ratio and ensure min > 0
                 ax2.set_yscale('log')
                 ax2.set_title(f'{name} Estimator: Variance Comparison (Log Scale)', fontsize=16)

            plt.tight_layout()
            plot_filename2 = os.path.join(self.results_dir, f"synthetic_{name}_variance.png")
            plt.savefig(plot_filename2)
            print(f"Saved variance plot: {plot_filename2}")
            plt.close(fig2)

            # --- Calculate and Print Summary Statistics (Needed for LaTeX Table) ---
            mask = ~np.isnan(original_mean) & ~np.isnan(modified_mean) & ~np.isnan(self.true_position_bias)
            original_mse = np.mean((original_mean[mask] - self.true_position_bias[mask])**2) if np.any(mask) else np.nan
            modified_mse = np.mean((modified_mean[mask] - self.true_position_bias[mask])**2) if np.any(mask) else np.nan

            mean_orig_var = np.nanmean(original_var)
            mean_mod_var = np.nanmean(modified_var)
            # Calculate variance reduction carefully avoiding division by zero or NaN
            variance_reduction_pct = np.nan
            if mean_orig_var > 0 and not np.isnan(mean_orig_var) and not np.isnan(mean_mod_var):
                variance_reduction_pct = (1 - mean_mod_var / mean_orig_var) * 100

            # Print values needed for the table clearly
            print(f"  Avg MSE (Original): {original_mse:.6f}")
            print(f"  Avg MSE (Modified): {modified_mse:.6f}")
            print(f"  Mean Var (Original): {mean_orig_var:.6f}")
            print(f"  Mean Var (Modified): {mean_mod_var:.6f}")
            print(f"  Overall Variance Reduction: {variance_reduction_pct:.2f}%")

            # Store for final summary printout
            summary_stats[name] = {
                'mse_orig': original_mse, 'mse_mod': modified_mse,
                'var_orig': mean_orig_var, 'var_mod': mean_mod_var,
                'var_reduction_pct': variance_reduction_pct
            }

        return summary_stats


# --- Main Execution ---
if __name__ == "__main__":
    # Configure Experiment
    experiment = SimplifiedExperiment(
        num_positions=10,
        num_products_per_pair=10,  # Fewer products per pair
        eta=1.0,                   # Standard position bias decay
        impressions_t1=100,        # High impression count
        impressions_t2=5,          # Low impression count (high imbalance)
        higher_pos_ratio=0.8,      # Asymmetric split (80/20)
        random_seed=42,
        results_dir="images/synthetic_results_final" # Use updated directory name
    )

    # Optional: Generate and show a sample dataset
    # print("Generating a sample dataset...")
    # _, sample_df = experiment.run_single_experiment()
    # print("\nSample dataset (first 10 rows):")
    # print(sample_df.head(10))
    # print(f"\nTotal rows: {len(sample_df)}")
    # print(f"Number of unique items: {sample_df['item_id'].nunique()}")

    # Run multiple experiments
    print("\nRunning multiple experiments...")
    num_runs = 30 # Number of iterations
    results = experiment.run_multiple_experiments(num_iterations=num_runs)

    # Plot results and print analysis
    print("\nPlotting results and generating analysis...")
    summary = experiment.plot_and_analyze_results(results)

    print("\n--- LaTeX Table Data Summary ---")
    # Print a formatted summary ready for copy-pasting values into LaTeX table
    print(f"{'Estimator':<15} {'Weighting':<10} {'Avg MSE':<10} {'Mean Var':<10} {'Var Reduc (%)'}")
    print("-" * 60)
    for name in experiment.estimator_names: # Should be AdjacentChain, AllPairs
        stats = summary[name]
        print(f"{name:<15} {'Original':<10} {stats['mse_orig']:.6f}   {stats['var_orig']:.6f}   {'--':<15}")
        print(f"{'':<15} {'Modified':<10} {stats['mse_mod']:.6f}   {stats['var_mod']:.6f}   {stats['var_reduction_pct']:.2f} %")
        print("-" * 60)


    print(f"\nPlots saved in '{experiment.results_dir}' directory.")
    print("Script finished.")
    # plt.show() # Usually commented out when running non-interactively