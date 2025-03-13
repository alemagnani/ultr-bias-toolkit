import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns
from tqdm import tqdm


class SyntheticExperiment:
    """
    Configurable synthetic experiment for position bias estimation with original and modified weights.
    """

    def __init__(
            self,
            num_queries=5000,
            num_items=100,
            num_positions=10,
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

    def generate_synthetic_data(self, num_queries=None):
        """Generate synthetic click data with common and rare items."""
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

        # Store clicks and rankings
        click_data = []

        # Generate queries and clicks
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

            # Generate clicks according to PBM with noise
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

                click_data.append({
                    'query_id': q,
                    'item_id': item,
                    'position': position + 1,  # 1-indexed positions
                    'click': click,
                    'ranker_id': ranker_id,
                    'relevance': rel,
                    'is_rare': item >= common_items
                })

        return pd.DataFrame(click_data), item_relevance

    def compute_weights(self, df, position_col='position'):
        """Compute original weights w(q,d,k) for each (query, item, position) tuple."""
        weight_dict = defaultdict(int)

        # Group by query_id, item_id and count occurrences at each position
        for _, group in df.groupby(['query_id', 'item_id']):
            query = group['query_id'].iloc[0]
            item = group['item_id'].iloc[0]

            # Count how many times this (query, item) appears at each position
            for pos, count in group[position_col].value_counts().items():
                weight_dict[(query, item, pos)] = count

        return weight_dict

    def compute_interventional_sets(self, df):
        """Compute interventional sets S_{k,k'} and extract clicks."""
        interventional_sets = defaultdict(list)

        # Group by query_id and item_id
        for (query, item), group in df.groupby(['query_id', 'item_id']):
            positions = group['position'].unique()

            # If this item appears at multiple positions for this query, it's part of interventional sets
            if len(positions) > 1:
                for i, pos1 in enumerate(positions):
                    for pos2 in positions[i + 1:]:
                        interventional_sets[(pos1, pos2)].append((query, item))
                        interventional_sets[(pos2, pos1)].append((query, item))

        return interventional_sets

    def estimate_propensities_original(self, df, interventional_sets, weights, max_position=None):
        """Estimate propensities using the original AllPairs method."""
        if max_position is None:
            max_position = self.num_positions

        click_rates = defaultdict(float)

        # Compute click rates for each position pair
        for (pos1, pos2), query_items in interventional_sets.items():
            if pos1 > max_position or pos2 > max_position:
                continue

            for query, item in query_items:
                # Find all occurrences of this query-item pair
                mask = (df['query_id'] == query) & (df['item_id'] == item)
                item_data = df[mask]

                # Calculate weighted click rates
                for _, row in item_data.iterrows():
                    if row['position'] == pos1:
                        weight = weights[(query, item, pos1)]
                        click_rates[(pos1, pos2)] += row['click'] / weight
                    elif row['position'] == pos2:
                        weight = weights[(query, item, pos2)]
                        click_rates[(pos2, pos1)] += row['click'] / weight

        # Estimate relative propensities using position 1 as reference
        estimated_propensities = np.ones(max_position)

        # For each position, find pairs involving position 1 and this position
        for pos in range(2, max_position + 1):
            if click_rates[(1, pos)] > 0 and click_rates[(pos, 1)] > 0:
                estimated_propensities[pos - 1] = click_rates[(pos, 1)] / click_rates[(1, pos)]

        return estimated_propensities

    def estimate_propensities_modified(self, df, interventional_sets, weights, max_position=None):
        """Estimate propensities using our modified weighting scheme."""
        if max_position is None:
            max_position = self.num_positions

        click_rates = defaultdict(float)

        # Compute click rates with modified weights
        for (pos1, pos2), query_items in interventional_sets.items():
            if pos1 > max_position or pos2 > max_position:
                continue

            for query, item in query_items:
                # Find all occurrences of this query-item pair
                mask = (df['query_id'] == query) & (df['item_id'] == item)
                item_data = df[mask]

                # Get original weights
                w1 = weights[(query, item, pos1)]
                w2 = weights[(query, item, pos2)]

                # Compute modified weights
                min_weight = min(w1, w2)
                modified_w1 = w1 / min_weight
                modified_w2 = w2 / min_weight

                # Calculate weighted click rates with modified weights
                for _, row in item_data.iterrows():
                    if row['position'] == pos1:
                        click_rates[(pos1, pos2)] += row['click'] / modified_w1
                    elif row['position'] == pos2:
                        click_rates[(pos2, pos1)] += row['click'] / modified_w2

        # Estimate relative propensities using position 1 as reference
        estimated_propensities = np.ones(max_position)

        # For each position, find pairs involving position 1 and this position
        for pos in range(2, max_position + 1):
            if click_rates[(1, pos)] > 0 and click_rates[(pos, 1)] > 0:
                estimated_propensities[pos - 1] = click_rates[(pos, 1)] / click_rates[(1, pos)]

        return estimated_propensities

    def run_single_experiment(self, num_queries=None):
        """Run a single experiment with the given number of queries."""
        # Generate synthetic data
        df, _ = self.generate_synthetic_data(num_queries)

        # Compute weights and interventional sets
        weights = self.compute_weights(df)
        interventional_sets = self.compute_interventional_sets(df)

        # Estimate propensities using both methods
        est_original = self.estimate_propensities_original(df, interventional_sets, weights)
        est_modified = self.estimate_propensities_modified(df, interventional_sets, weights)

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
            results_original[i] = est_original
            results_modified[i] = est_modified

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

    # Run basic experiment with confidence intervals
    print("Running basic experiment...")
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