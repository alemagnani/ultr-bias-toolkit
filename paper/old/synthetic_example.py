import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from ultr_bias_toolkit.bias.intervention_harvesting import AdjacentChainEstimator, PivotEstimator
from ultr_bias_toolkit.bias.naive import NaiveCtrEstimator


class SyntheticExperiment:
    """
    Configurable synthetic experiment for position bias estimation with original and modified weights.
    """

    def __init__(
            self,
            num_queries=5000,
            num_items=100,
            num_positions=20,
            num_rankers=2,
            alpha=1.0,  # Position bias steepness
            rare_item_ratio=0.7,  # Percentage of rare items
            rare_item_prob=0.1,  # Probability of selecting a rare item
            relevance_variance=0.2,
            click_noise=0.0,  # Probability of random clicks/non-clicks
            random_seed=42
    ):
        """Initialize the synthetic experiment with configurable parameters."""
        self.num_queries = num_queries
        self.num_items = num_items
        self.num_positions = num_positions
        self.num_rankers = num_rankers
        self.alpha = alpha
        self.rare_item_ratio = rare_item_ratio
        self.rare_item_prob = rare_item_prob
        self.relevance_variance = relevance_variance
        self.click_noise = click_noise  # Click noise level
        self.random_seed = random_seed

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Generate true position bias
        self.true_position_bias = np.array([1.0 / (k ** alpha) for k in range(1, num_positions + 1)])
        # Normalize to make position 1 have bias 1.0
        self.true_position_bias = self.true_position_bias / self.true_position_bias[0]

        #self.estimator_pair_original = AdjacentChainEstimator(weighting="original")
        #self.estimator_pair_var_reduce = AdjacentChainEstimator(weighting="variance_reduced")

        self.estimator_pair_original = NaiveCtrEstimator()
        self.estimator_pair_var_reduce = NaiveCtrEstimator()



    def generate_synthetic_data(self, num_queries=None):
        """Generate synthetic impression and click data with common and rare items."""
        if num_queries is None:
            num_queries = self.num_queries

        # Generate item relevance (base relevance plus some noise)
        rare_items = int(self.num_items * self.rare_item_ratio)
        common_items = self.num_items - rare_items

        # Assign a base relevance to each item (higher for common items)
        item_relevance = np.zeros(self.num_items)
        item_relevance[:common_items] = 0.7 + self.relevance_variance * np.random.randn(common_items)
        item_relevance[common_items:] = 0.3 + self.relevance_variance * np.random.randn(rare_items)
        # Clip relevance to [0, 1]
        item_relevance = np.clip(item_relevance, 0, 1)

        # Store impressions and clicks
        impression_data = []

        # Generate queries, impressions and clicks
        for q in range(num_queries):
            # Choose a ranker
            ranker_id = np.random.randint(self.num_rankers)

            # Select items for this query (more likely to select common items)
            selected_items = []
            while len(selected_items) < self.num_positions:
                if np.random.random() < self.rare_item_prob:
                    # Select a rare item
                    item = np.random.randint(common_items, self.num_items)
                else:
                    # Select a common item
                    item = np.random.randint(common_items)

                if item not in selected_items:
                    selected_items.append(item)

            # If we didn't get enough items, fill with random ones
            while len(selected_items) < self.num_positions:
                item = np.random.randint(self.num_items)
                if item not in selected_items:
                    selected_items.append(item)

            # Each ranker orders items differently
            if ranker_id == 0:
                # First ranker just uses the items in the order selected
                ranking = selected_items[:self.num_positions]
            else:
                # Other rankers apply some randomization to the order
                ranking = selected_items.copy()
                np.random.shuffle(ranking)
                ranking = ranking[:self.num_positions]

            # Generate impressions and clicks according to PBM with noise
            for position, item in enumerate(ranking):
                position_bias = self.true_position_bias[position]
                rel = item_relevance[item]

                # Base click probability = position_bias * relevance
                click_prob = position_bias * rel

                # Apply click noise: with probability click_noise, flip the click decision
                if np.random.random() < self.click_noise:
                    # Random click decision (noise)
                    click = 1 if np.random.random() < 0.5 else 0
                else:
                    # Normal PBM click
                    click = 1 if np.random.random() < click_prob else 0

                impression_data.append({
                    'query_id': q,
                    'item_id': item,
                    'position': position + 1,  # 1-indexed positions
                    'impression': 1,  # All items are impressed
                    'click': click,
                    'ranker_id': ranker_id,
                    'relevance': rel,
                    'is_rare': item >= common_items
                })

        return pd.DataFrame(impression_data), item_relevance

    def run_single_experiment(self, num_queries=None):
        """Run a single experiment with the given number of queries."""
        # Generate synthetic data
        df, _ = self.generate_synthetic_data(num_queries)

        # Ensure we have the necessary columns for the estimators
        if 'impression' not in df.columns:
            df['impression'] = 1  # All rows are impressions by default

        # Estimate propensities using both methods
        est_original = self.estimator_pair_original(df, doc_col='item_id')
        est_modified = self.estimator_pair_var_reduce(df, doc_col='item_id')

        return est_original, est_modified

    def run_multiple_experiments(self, num_iterations=100, num_queries=None):
        """Run multiple iterations to compute means and confidence intervals."""
        if num_queries is None:
            num_queries = self.num_queries

        results_original = np.zeros((num_iterations, self.num_positions))
        results_modified = np.zeros((num_iterations, self.num_positions))

        for i in tqdm(range(num_iterations), desc=f"Running {num_iterations} iterations with {num_queries} queries"):
            # Reset random seed for each iteration but make it different
            np.random.seed(self.random_seed + i)

            # Run experiment
            est_original, est_modified = self.run_single_experiment(num_queries)

            # Store results
            results_original[i] = est_original['examination']
            results_modified[i] = est_modified['examination']

        return results_original, results_modified

    def analyze_data_size_impact(self, position_to_analyze=5, data_fractions=None, num_iterations=50):
        """
        Analyze how confidence interval width changes with increasing data size
        for a fixed position.
        """
        if data_fractions is None:
            data_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

        ci_widths_original = []
        ci_widths_modified = []
        mse_original = []
        mse_modified = []

        position_idx = position_to_analyze - 1  # Convert to 0-indexed
        true_value = self.true_position_bias[position_idx]

        for fraction in tqdm(data_fractions, desc="Testing different data sizes"):
            num_queries = int(self.num_queries * fraction)

            results_original, results_modified = self.run_multiple_experiments(
                num_iterations=num_iterations,
                num_queries=num_queries
            )

            # Extract results for the position we're analyzing
            pos_results_original = results_original[:, position_idx]
            pos_results_modified = results_modified[:, position_idx]

            # Compute 95% confidence interval width
            std_original = np.std(pos_results_original)
            std_modified = np.std(pos_results_modified)

            ci_width_original = 1.96 * std_original / np.sqrt(num_iterations)
            ci_width_modified = 1.96 * std_modified / np.sqrt(num_iterations)

            ci_widths_original.append(ci_width_original)
            ci_widths_modified.append(ci_width_modified)

            # Compute MSE
            mean_original = np.mean(pos_results_original)
            mean_modified = np.mean(pos_results_modified)

            mse_orig = np.mean((pos_results_original - true_value) ** 2)
            mse_mod = np.mean((pos_results_modified - true_value) ** 2)

            mse_original.append(mse_orig)
            mse_modified.append(mse_mod)

        return data_fractions, ci_widths_original, ci_widths_modified, mse_original, mse_modified

    def plot_confidence_intervals(self, results_original, results_modified):
        """Plot the propensity estimates with confidence intervals."""
        positions = np.arange(1, self.num_positions + 1)

        # Compute mean and confidence intervals
        original_mean = np.mean(results_original, axis=0)
        original_std = np.std(results_original, axis=0)
        original_ci = 1.96 * original_std / np.sqrt(results_original.shape[0])  # 95% CI

        modified_mean = np.mean(results_modified, axis=0)
        modified_std = np.std(results_modified, axis=0)
        modified_ci = 1.96 * modified_std / np.sqrt(results_modified.shape[0])  # 95% CI

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot ground truth
        plt.plot(positions, self.true_position_bias, 'k-', label='Ground Truth', linewidth=2)

        # Plot original estimator
        plt.errorbar(positions, original_mean, yerr=original_ci, fmt='ro-', label='Original Estimator', alpha=0.7)

        # Plot modified estimator
        plt.errorbar(positions, modified_mean, yerr=modified_ci, fmt='bo-', label='Modified Estimator', alpha=0.7)

        plt.xlabel('Position', fontsize=14)
        plt.ylabel('Relative Propensity', fontsize=14)
        plt.title('Comparison of Propensity Estimators', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Compare error metrics
        original_mse = np.mean((original_mean - self.true_position_bias) ** 2)
        modified_mse = np.mean((modified_mean - self.true_position_bias) ** 2)

        print(f"Original Estimator MSE: {original_mse:.6f}")
        print(f"Modified Estimator MSE: {modified_mse:.6f}")
        print(f"Variance Reduction: {(1 - modified_mse / original_mse) * 100:.2f}%")

        return original_mean, modified_mean, original_std, modified_std

    def plot_data_size_impact(self, data_fractions, ci_widths_original, ci_widths_modified, mse_original, mse_modified,
                              position):
        """Plot the impact of data size on confidence interval width and MSE."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot confidence interval width
        ax1.plot(data_fractions, ci_widths_original, 'ro-', label='Original Estimator')
        ax1.plot(data_fractions, ci_widths_modified, 'bo-', label='Modified Estimator')
        ax1.set_xlabel('Data Fraction', fontsize=12)
        ax1.set_ylabel('95% CI Width', fontsize=12)
        ax1.set_title(f'Confidence Interval Width at Position {position}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot MSE
        ax2.plot(data_fractions, mse_original, 'ro-', label='Original Estimator')
        ax2.plot(data_fractions, mse_modified, 'bo-', label='Modified Estimator')
        ax2.set_xlabel('Data Fraction', fontsize=12)
        ax2.set_ylabel('Mean Squared Error', fontsize=12)
        ax2.set_title(f'MSE at Position {position}', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Also plot the variance reduction percentage
        plt.figure(figsize=(10, 6))
        variance_reduction = [100 * (1 - m2 / m1) for m1, m2 in zip(mse_original, mse_modified)]
        plt.plot(data_fractions, variance_reduction, 'go-', linewidth=2)
        plt.xlabel('Data Fraction', fontsize=12)
        plt.ylabel('Variance Reduction (%)', fontsize=12)
        plt.title(f'Variance Reduction at Position {position}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)
        plt.tight_layout()

        return fig

    def analyze_click_noise_impact(self, position_to_analyze=5, noise_levels=None, num_iterations=50):
        """
        Analyze how click noise affects the accuracy of propensity estimation
        for both original and modified estimators.
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        mse_original = []
        mse_modified = []
        ci_widths_original = []
        ci_widths_modified = []

        position_idx = position_to_analyze - 1  # Convert to 0-indexed
        true_value = self.true_position_bias[position_idx]

        # Store the original click noise value
        original_noise = self.click_noise

        for noise in tqdm(noise_levels, desc="Testing different noise levels"):
            # Set the current noise level
            self.click_noise = noise

            # Run experiments with this noise level
            results_original, results_modified = self.run_multiple_experiments(
                num_iterations=num_iterations
            )

            # Extract results for the position we're analyzing
            pos_results_original = results_original[:, position_idx]
            pos_results_modified = results_modified[:, position_idx]

            # Compute 95% confidence interval width
            std_original = np.std(pos_results_original)
            std_modified = np.std(pos_results_modified)

            ci_width_original = 1.96 * std_original / np.sqrt(num_iterations)
            ci_width_modified = 1.96 * std_modified / np.sqrt(num_iterations)

            ci_widths_original.append(ci_width_original)
            ci_widths_modified.append(ci_width_modified)

            # Compute MSE
            mse_orig = np.mean((pos_results_original - true_value) ** 2)
            mse_mod = np.mean((pos_results_modified - true_value) ** 2)

            mse_original.append(mse_orig)
            mse_modified.append(mse_mod)

        # Restore the original noise value
        self.click_noise = original_noise

        return noise_levels, ci_widths_original, ci_widths_modified, mse_original, mse_modified

    def plot_noise_impact(self, noise_levels, ci_widths_original, ci_widths_modified, mse_original, mse_modified,
                          position):
        """Plot the impact of click noise on confidence interval width and MSE."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot confidence interval width
        ax1.plot(noise_levels, ci_widths_original, 'ro-', label='Original Estimator')
        ax1.plot(noise_levels, ci_widths_modified, 'bo-', label='Modified Estimator')
        ax1.set_xlabel('Click Noise Level', fontsize=12)
        ax1.set_ylabel('95% CI Width', fontsize=12)
        ax1.set_title(f'Confidence Interval Width at Position {position}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot MSE
        ax2.plot(noise_levels, mse_original, 'ro-', label='Original Estimator')
        ax2.plot(noise_levels, mse_modified, 'bo-', label='Modified Estimator')
        ax2.set_xlabel('Click Noise Level', fontsize=12)
        ax2.set_ylabel('Mean Squared Error', fontsize=12)
        ax2.set_title(f'MSE at Position {position}', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Also plot the variance reduction percentage at different noise levels
        plt.figure(figsize=(10, 6))
        variance_reduction = [100 * (1 - m2 / m1) if m1 > 0 else 0 for m1, m2 in zip(mse_original, mse_modified)]
        plt.plot(noise_levels, variance_reduction, 'go-', linewidth=2)
        plt.xlabel('Click Noise Level', fontsize=12)
        plt.ylabel('Variance Reduction (%)', fontsize=12)
        plt.title(f'Variance Reduction at Position {position}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)
        plt.tight_layout()

        return fig

    def create_impression_click_dataframe(self, df=None, num_queries=None):
        """
        Create a dataframe with impressions and clicks.
        If df is provided, it will use that dataframe; otherwise, it will generate synthetic data.
        """
        if df is None:
            df, _ = self.generate_synthetic_data(num_queries)

        # Make sure we have the impression column
        if 'impression' not in df.columns:
            df['impression'] = 1  # All rows are impressions

        return df


# Example usage
if __name__ == "__main__":
    # Create experiment with default parameters
    experiment = SyntheticExperiment(
        num_queries=5000,
        num_items=100,
        num_positions=10,
        num_rankers=3,
        alpha=0.8,  # Position bias steepness
        rare_item_ratio=0.7,
        rare_item_prob=0.1,
        relevance_variance=0.2,
        click_noise=0.0,  # Start with no noise
        random_seed=42
    )

    # Generate and use a dataframe with impressions and clicks
    print("Creating impression-click dataframe...")
    impression_click_df = experiment.create_impression_click_dataframe(num_queries=1000)

    # Example of using the dataframe with the estimators directly
    print("Using dataframe with estimators directly...")
    est_original = experiment.estimator_pair_original(impression_click_df, doc_col='item_id')
    est_modified = experiment.estimator_pair_var_reduce(impression_click_df, doc_col='item_id')

    print("First few rows of the dataframe:")
    print(impression_click_df.head())

    print("Original estimator results:", est_original)
    print("Modified estimator results:", est_modified)

    # Run basic experiment with confidence intervals
    print("\nRunning basic experiment with both estimators...")
    results_original, results_modified = experiment.run_multiple_experiments(num_iterations=50)
    original_mean, modified_mean, original_std, modified_std = experiment.plot_confidence_intervals(results_original,
                                                                                                    results_modified)

    # Analyze impact of data size on confidence intervals for position 5
    print("\nAnalyzing impact of data size on confidence intervals...")
    position_to_analyze = 5
    data_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    data_fractions, ci_widths_original, ci_widths_modified, mse_original, mse_modified = experiment.analyze_data_size_impact(
        position_to_analyze=position_to_analyze,
        data_fractions=data_fractions,
        num_iterations=30
    )

    experiment.plot_data_size_impact(
        data_fractions,
        ci_widths_original,
        ci_widths_modified,
        mse_original,
        mse_modified,
        position_to_analyze
    )

    # Analyze impact of click noise
    print("\nAnalyzing impact of click noise...")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    noise_levels, ci_widths_original, ci_widths_modified, mse_original, mse_modified = experiment.analyze_click_noise_impact(
        position_to_analyze=position_to_analyze,
        noise_levels=noise_levels,
        num_iterations=30
    )

    experiment.plot_noise_impact(
        noise_levels,
        ci_widths_original,
        ci_widths_modified,
        mse_original,
        mse_modified,
        position_to_analyze
    )

    plt.show()