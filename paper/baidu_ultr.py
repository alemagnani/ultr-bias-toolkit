from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple
from ultr_bias_toolkit.bias.intervention_harvesting import AdjacentChainEstimator
from ultr_bias_toolkit.bias.intervention_harvesting import AllPairsEstimator
from ultr_bias_toolkit.bias.intervention_harvesting import PivotEstimator
from ultr_bias_toolkit.bias.naive import NaiveCtrEstimator

FEATURE_URL = "https://huggingface.co/datasets/philipphager/baidu-ultr_baidu-mlm-ctr/blob/main/parts/train-features.feather?download=true"


def bootstrap_confidence_interval(data, estimator_func, n_bootstrap=100, ci=0.95):
    """
    Calculate bootstrap confidence intervals for an estimator function.

    Args:
        data: DataFrame to bootstrap from
        estimator_func: Function that takes data and returns position-examination estimates
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level

    Returns:
        DataFrame with position, mean examination, lower and upper bound
    """
    results = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = data.sample(frac=1.0, replace=True)

        # Apply estimator to bootstrap sample
        examination_df = estimator_func(bootstrap_sample)

        # Store results
        results.append(examination_df.set_index('position')['examination'])

    # Combine bootstrap results
    bootstrap_df = pd.concat(results, axis=1)

    # Calculate mean and confidence intervals
    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100

    ci_df = pd.DataFrame({
        'mean': bootstrap_df.mean(axis=1),
        'lower': bootstrap_df.apply(lambda x: np.percentile(x, lower_percentile), axis=1),
        'upper': bootstrap_df.apply(lambda x: np.percentile(x, upper_percentile), axis=1)
    }).reset_index()

    return ci_df


def run_comparison(
        df: pd.DataFrame,
        estimators: Dict,
        data_sizes: List[int],
        n_samples: int = 5,
        n_bootstrap: int = 100,
        query_col: str = "query_md5",
        doc_col: str = "url_md5",
        random_state: int = 2024,
):
    """
    Run comparison of estimators at different data sizes with confidence intervals.

    Args:
        df: DataFrame with click data
        estimators: Dictionary of estimator names and objects
        data_sizes: List of data sizes to evaluate
        n_samples: Number of samples to take for each data size
        n_bootstrap: Number of bootstrap samples for confidence intervals
        query_col: Name of query column
        doc_col: Name of document column
        random_state: Random seed

    Returns:
        DataFrame with comparison results
    """
    np.random.seed(random_state)

    # Get the largest data size for "ground truth"
    full_data_size = len(df)
    if max(data_sizes) < full_data_size:
        data_sizes.append(full_data_size)

    # Calculate ground truth using all data
    ground_truth = {}
    print("Calculating ground truth with all data...")
    for name, estimator in estimators.items():
        examination_df = estimator(df, query_col=query_col, doc_col=doc_col)
        ground_truth[name] = examination_df.set_index('position')['examination']

    results = []

    # Run estimators for each data size and sample
    for data_size in tqdm(data_sizes, desc="Data sizes"):
        # Skip bootstrap for full dataset (already calculated above)
        if data_size == full_data_size:
            for name, gt_values in ground_truth.items():
                for position, value in gt_values.items():
                    results.append({
                        'estimator': name,
                        'data_size': data_size,
                        'position': position,
                        'mean': value,
                        'lower': value,  # No CI for ground truth
                        'upper': value,  # No CI for ground truth
                        'sample': 0,
                        'is_ground_truth': True
                    })
            continue

        # For smaller data sizes, take multiple samples
        for sample_idx in range(n_samples):
            # Sample data (without replacement for fair comparison)
            sample_df = df.sample(data_size, random_state=random_state + sample_idx)

            for name, estimator in estimators.items():
                # Define a function that applies the current estimator to sampled data
                def apply_estimator(data):
                    return estimator(data, query_col=query_col, doc_col=doc_col)

                # Calculate confidence intervals using bootstrap
                ci_df = bootstrap_confidence_interval(
                    sample_df,
                    apply_estimator,
                    n_bootstrap=n_bootstrap
                )

                # Calculate distance to ground truth
                ci_df['ground_truth'] = ci_df['position'].map(ground_truth[name])
                ci_df['error'] = abs(ci_df['mean'] - ci_df['ground_truth'])

                # Store results
                for _, row in ci_df.iterrows():
                    results.append({
                        'estimator': name,
                        'data_size': data_size,
                        'position': row['position'],
                        'mean': row['mean'],
                        'lower': row['lower'],
                        'upper': row['upper'],
                        'error': row['error'],
                        'sample': sample_idx,
                        'is_ground_truth': False
                    })

    return pd.DataFrame(results)


def plot_comparison(results_df, output_dir="comparison_plots"):
    """Generate plots to visualize the comparison results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot 1: Propensity curves for each estimator at different data sizes
    data_sizes = sorted(results_df['data_size'].unique())
    estimators = sorted(results_df['estimator'].unique())

    # Select a representative sample
    sample_results = results_df[results_df['sample'] == 0]

    for data_size in data_sizes:
        if data_size == max(data_sizes):  # Skip the ground truth plot
            continue

        plt.figure(figsize=(12, 8))

        for estimator in estimators:
            subset = sample_results[(sample_results['data_size'] == data_size) &
                                    (sample_results['estimator'] == estimator)]

            # Sort by position
            subset = subset.sort_values('position')

            plt.plot(subset['position'], subset['mean'], label=estimator)
            plt.fill_between(
                subset['position'],
                subset['lower'],
                subset['upper'],
                alpha=0.2
            )

        plt.title(f'Examination Propensity Estimates with {data_size:,} samples')
        plt.xlabel('Position')
        plt.ylabel('Examination Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/propensity_data_size_{data_size}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 2: Error compared to ground truth by data size
    plt.figure(figsize=(12, 8))

    # Calculate mean error across positions and samples for each estimator and data size
    error_summary = results_df[~results_df['is_ground_truth']].groupby(
        ['estimator', 'data_size']
    )['error'].mean().reset_index()

    # Create the plot
    sns.lineplot(
        data=error_summary,
        x='data_size',
        y='error',
        hue='estimator',
        marker='o'
    )

    plt.title('Mean Absolute Error vs Ground Truth')
    plt.xlabel('Data Size (Number of Samples)')
    plt.ylabel('Mean Absolute Error')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/error_by_data_size.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Variance by data size (using CI width)
    results_df['ci_width'] = results_df['upper'] - results_df['lower']

    # Calculate mean CI width across positions and samples for each estimator and data size
    ci_summary = results_df[~results_df['is_ground_truth']].groupby(
        ['estimator', 'data_size']
    )['ci_width'].mean().reset_index()

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=ci_summary,
        x='data_size',
        y='ci_width',
        hue='estimator',
        marker='o'
    )

    plt.title('Estimation Uncertainty (CI Width) by Data Size')
    plt.xlabel('Data Size (Number of Samples)')
    plt.ylabel('Mean 95% CI Width')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/uncertainty_by_data_size.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Position-specific error plots for each estimator
    for estimator in estimators:
        plt.figure(figsize=(12, 8))

        estimator_results = results_df[
            (results_df['estimator'] == estimator) &
            (~results_df['is_ground_truth'])
            ]

        # Get top 5 positions
        top_positions = sorted(estimator_results['position'].unique())[:5]

        for position in top_positions:
            position_data = estimator_results[estimator_results['position'] == position]

            # Calculate mean error across samples for each data size
            error_by_size = position_data.groupby('data_size')['error'].mean()

            plt.plot(error_by_size.index, error_by_size.values,
                     label=f'Position {position}', marker='o')

        plt.title(f'Position-specific Error by Data Size: {estimator}')
        plt.xlabel('Data Size (Number of Samples)')
        plt.ylabel('Mean Absolute Error')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/{estimator}_position_error.png", dpi=300, bbox_inches='tight')
        plt.close()


def export_results(results_df, output_dir="comparison_results"):
    """Export the results to CSV files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Export full results
    results_df.to_csv(f"{output_dir}/full_results.csv", index=False)

    # Export summary by data size
    summary = results_df[~results_df['is_ground_truth']].groupby(
        ['estimator', 'data_size']
    ).agg({
        'error': ['mean', 'std'],
        'ci_width': ['mean', 'std']
    }).reset_index()

    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary.to_csv(f"{output_dir}/summary_by_data_size.csv", index=False)

    # Export position-specific results for each estimator
    for estimator in results_df['estimator'].unique():
        estimator_results = results_df[results_df['estimator'] == estimator]

        # Ground truth values
        ground_truth = estimator_results[estimator_results['is_ground_truth']]
        ground_truth = ground_truth[['position', 'mean']].rename(columns={'mean': 'ground_truth'})

        # Results at each data size
        for data_size in sorted(estimator_results['data_size'].unique()):
            if data_size == max(estimator_results['data_size'].unique()):
                continue  # Skip ground truth

            size_results = estimator_results[
                (estimator_results['data_size'] == data_size) &
                (~estimator_results['is_ground_truth'])
                ]

            # Average across samples
            avg_results = size_results.groupby('position').agg({
                'mean': 'mean',
                'lower': 'mean',
                'upper': 'mean',
                'error': 'mean'
            }).reset_index()

            # Merge with ground truth
            merged = pd.merge(avg_results, ground_truth, on='position')

            # Export
            merged.to_csv(
                f"{output_dir}/{estimator}_data_size_{data_size}.csv",
                index=False
            )

    print(f"Results exported to {output_dir}/")


def main(
        cache_directory: str = "/beegfs/scratch/user/rdeffaye/baidu-bert/features/",
        random_state: int = 2024,
        data_sizes: List[int] = [1000, 5000, 10000, 50000, 100000, 500000],
        n_samples: int = 5,
        n_bootstrap: int = 100,
):
    # For the cross-entropy maximization in AllPairs
    torch.manual_seed(random_state)

    cache_directory = Path(cache_directory).expanduser()
    cache_directory.mkdir(parents=True, exist_ok=True)

    feature_path = cache_directory / "train-features.feather"
    if not feature_path.exists():
        print("Downloading Baidu features from huggingface...")
        df = pd.read_feather(FEATURE_URL, columns=["query_md5", "url_md5", "position", "click"])
        df.to_feather(feature_path)
    else:
        df = pd.read_feather(feature_path, columns=["query_md5", "url_md5", "position", "click"])

    # Set up estimators
    estimators = {
        "ctr": NaiveCtrEstimator(),
        "pivot_one": PivotEstimator(pivot_rank=1),
        "adjacent_chain": AdjacentChainEstimator(),
        "global_all_pairs": AllPairsEstimator(),
    }

    # Run the comparison
    print(f"Running comparison with data sizes: {data_sizes}")
    print(f"Taking {n_samples} samples for each data size")
    print(f"Using {n_bootstrap} bootstrap samples for confidence intervals")

    results_df = run_comparison(
        df=df,
        estimators=estimators,
        data_sizes=data_sizes,
        n_samples=n_samples,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    # Plot and export results
    plot_comparison(results_df)
    export_results(results_df)

    print("Comparison complete!")


if __name__ == "__main__":
    # You can modify these parameters
    main(
        # Smaller numbers for testing, increase for production
        data_sizes=[1000, 5000, 10000, 50000, 100000],
        n_samples=3,  # Number of samples per data size
        n_bootstrap=50,  # Number of bootstrap samples for CI
    )