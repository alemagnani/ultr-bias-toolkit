import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from ultr_bias_toolkit.bias.intervention_harvesting import AdjacentChainEstimator, PivotEstimator


class SimplifiedExperiment:
    """
    Simplified synthetic experiment for position bias estimation comparing two algorithms.
    - Single query
    - N products with random relevance
    - Position bias follows (1/r)^eta
    - K products for each pair of adjacent positions
    - Products are shown at higher and lower positions with 70:30 split
    - T1 or T2 impressions per product
    """

    def __init__(
            self,
            num_positions=10,
            num_products_per_pair=20,  # K products per position pair
            eta=1.0,  # Position bias steepness parameter
            impressions_t1=1000,  # T1 impressions for half the products
            impressions_t2=2000,  # T2 impressions for the other half
            higher_pos_ratio=0.7,  # 70% impressions at higher position, 30% at lower
            random_seed=42
    ):
        """Initialize the simplified experiment with configurable parameters."""
        self.num_positions = num_positions
        self.num_products_per_pair = num_products_per_pair
        self.eta = eta
        self.impressions_t1 = impressions_t1
        self.impressions_t2 = impressions_t2
        self.higher_pos_ratio = higher_pos_ratio
        self.random_seed = random_seed

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Generate true position bias
        self.true_position_bias = np.array([1.0 / (k ** eta) for k in range(1, num_positions + 1)])
        # Normalize to make position 1 have bias 1.0
        self.true_position_bias = self.true_position_bias / self.true_position_bias[0]

        # Create estimators with different weighting schemes
        self.estimator_original = AdjacentChainEstimator(weighting="original")
        self.estimator_var_reduce = AdjacentChainEstimator(weighting="variance_reduced")

    def generate_dataset(self):
        """
        Generate a simplified dataset with:
        - Single query
        - Products appearing at adjacent positions
        - Impressions split 70/30 between higher and lower positions
        - Half of products get T1 impressions, half get T2 impressions
        """
        data = []
        product_id_counter = 0

        # For each pair of adjacent positions
        for pos in range(1, self.num_positions):
            higher_pos = pos
            lower_pos = pos + 1

            # Position bias values
            higher_bias = self.true_position_bias[higher_pos - 1]  # -1 for 0-indexed
            lower_bias = self.true_position_bias[lower_pos - 1]

            # Generate K products for this position pair
            for k in range(self.num_products_per_pair):
                product_id = product_id_counter
                product_id_counter += 1

                # Assign random relevance between 0.3 and 0.8
                product_relevance = 0.3 + 0.5 * np.random.random()

                # Decide if this product gets T1 or T2 impressions
                total_impressions = self.impressions_t1 if k < self.num_products_per_pair / 2 else self.impressions_t2

                # Split impressions between higher and lower positions
                higher_impressions = int(total_impressions * self.higher_pos_ratio)
                lower_impressions = total_impressions - higher_impressions

                # Generate clicks for higher position
                if higher_impressions > 0:
                    click_prob_higher = product_relevance * higher_bias
                    higher_clicks = np.random.binomial(higher_impressions, click_prob_higher)

                    data.append({
                        'query_id': 0,  # Single query
                        'item_id': product_id,
                        'position': higher_pos,
                        'impression': higher_impressions,
                        'click': higher_clicks,
                        'relevance': product_relevance
                    })

                # Generate clicks for lower position
                if lower_impressions > 0:
                    click_prob_lower = product_relevance * lower_bias
                    lower_clicks = np.random.binomial(lower_impressions, click_prob_lower)

                    data.append({
                        'query_id': 0,  # Single query
                        'item_id': product_id,
                        'position': lower_pos,
                        'impression': lower_impressions,
                        'click': lower_clicks,
                        'relevance': product_relevance
                    })

        return pd.DataFrame(data)

    def run_single_experiment(self):
        """Run a single experiment and return the propensity estimates."""
        # Generate dataset
        df = self.generate_dataset()



        # Estimate propensities using both methods
        est_original = self.estimator_original(df, doc_col='item_id', imps_col='impression', clicks_col='click')['examination']
        est_modified = self.estimator_var_reduce(df, doc_col='item_id', imps_col='impression', clicks_col='click')['examination']

        return est_original, est_modified, df

    def run_multiple_experiments(self, num_iterations=100):
        """Run multiple iterations to compute means and variances."""
        results_original = np.zeros((num_iterations, self.num_positions))
        results_modified = np.zeros((num_iterations, self.num_positions))

        for i in tqdm(range(num_iterations), desc=f"Running {num_iterations} iterations"):
            # Reset random seed for each iteration but make it different
            np.random.seed(self.random_seed + i)

            # Run experiment
            est_original, est_modified, _ = self.run_single_experiment()

            # Store results
            results_original[i] = est_original
            results_modified[i] = est_modified

        return results_original, results_modified

    def plot_results(self, results_original, results_modified):
        """Plot the propensity estimates with confidence intervals and variance."""
        positions = np.arange(1, self.num_positions + 1)

        # Compute mean, confidence intervals, and variance
        original_mean = np.mean(results_original, axis=0)
        original_var = np.var(results_original, axis=0)
        original_std = np.std(results_original, axis=0)
        original_ci = 1.96 * original_std / np.sqrt(results_original.shape[0])  # 95% CI

        modified_mean = np.mean(results_modified, axis=0)
        modified_var = np.var(results_modified, axis=0)
        modified_std = np.std(results_modified, axis=0)
        modified_ci = 1.96 * modified_std / np.sqrt(results_modified.shape[0])  # 95% CI

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Plot 1: Propensity estimates with confidence intervals
        ax1.plot(positions, self.true_position_bias, 'k-', label='Ground Truth', linewidth=2)
        ax1.errorbar(positions, original_mean, yerr=original_ci, fmt='ro-', label='Original Estimator', alpha=0.7)
        ax1.errorbar(positions, modified_mean, yerr=modified_ci, fmt='bo-', label='Modified Estimator', alpha=0.7)

        ax1.set_xlabel('Position', fontsize=14)
        ax1.set_ylabel('Relative Propensity', fontsize=14)
        ax1.set_title('Propensity Estimates with 95% CI', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Variance of estimators and variance reduction
        ax2.bar(positions - 0.2, original_var, width=0.4, color='r', alpha=0.7, label='Original Estimator')
        ax2.bar(positions + 0.2, modified_var, width=0.4, color='b', alpha=0.7, label='Modified Estimator')

        ax2.set_xlabel('Position', fontsize=14)
        ax2.set_ylabel('Variance', fontsize=14)
        ax2.set_title('Estimator Variance by Position', fontsize=16)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Create a figure for variance reduction
        plt.figure(figsize=(10, 6))
        variance_reduction = 100 * (1 - modified_var / original_var)
        plt.bar(positions, variance_reduction, color='g', alpha=0.7)
        plt.xlabel('Position', fontsize=14)
        plt.ylabel('Variance Reduction (%)', fontsize=14)
        plt.title('Variance Reduction by Position', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Print summary statistics
        print("Average MSE:")
        original_mse = np.mean((original_mean - self.true_position_bias) ** 2)
        modified_mse = np.mean((modified_mean - self.true_position_bias) ** 2)
        print(f"Original Estimator MSE: {original_mse:.6f}")
        print(f"Modified Estimator MSE: {modified_mse:.6f}")
        print(f"Overall Variance Reduction: {(1 - modified_mse / original_mse) * 100:.2f}%")

        return fig


# Example usage
if __name__ == "__main__":
    # Create simplified experiment
    experiment = SimplifiedExperiment(
        num_positions=10,
        num_products_per_pair=10,  # K products per position pair
        eta=1.0,  # Position bias steepness
        impressions_t1=100,  # T1 impressions
        impressions_t2=5,  # T2 impressions
        higher_pos_ratio=0.7,  # 70% at higher position, 30% at lower
        random_seed=42
    )

    # Generate and visualize a single dataset
    print("Generating a sample dataset...")
    _, _, sample_df = experiment.run_single_experiment()
    print("\nSample dataset:")
    print(sample_df.head(10))

    print("\nDataset statistics:")
    print(f"Number of unique products: {sample_df['item_id'].nunique()}")
    print(f"Number of rows: {len(sample_df)}")
    print(f"Average impressions per product: {sample_df.groupby('item_id')['impression'].sum().mean():.1f}")
    print(f"Average clicks per product: {sample_df.groupby('item_id')['click'].sum().mean():.1f}")

    # Run multiple experiments and analyze results
    print("\nRunning multiple experiments to analyze variance...")
    results_original, results_modified = experiment.run_multiple_experiments(num_iterations=100)

    # Plot results
    print("\nPlotting results...")
    experiment.plot_results(results_original, results_modified)

    plt.show()