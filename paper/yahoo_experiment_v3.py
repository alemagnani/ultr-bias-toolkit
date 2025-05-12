# --- START OF FILE yahoo_simulation_stress_test.py ---

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import gc # Garbage collector

# Import your estimator(s)
from ultr_bias_toolkit.bias.intervention_harvesting import AllPairsEstimator

# --- Configuration ---
YAHOO_TRAIN_PATH = "/home/alessandro/Documents/data/Yahoo_ltr/set1.train.txt"
YAHOO_TEST_PATH = "/home/alessandro/Documents/data/Yahoo_ltr/set1.test.txt"
RESULTS_DIR = "images/yahoo_results_stress_test" # New directory
RANDOM_SEED = 42
# Reduce NUM_RUNS initially for testing, increase for final results
NUM_RUNS = 30 # Number of simulation repetitions (e.g., use 5 for test, 20 for final)

ANCHOR_ITEM_PERCENTAGE = 0.01  # 5% of items are "anchor" items
SPARSE_ITEM_PROBABILITY = 1.  # Very rarely log sparse items
ACHOR_IMPS = 50
OTHER_IMPS = 10

# LTR Model Params
LGBM_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [10],
    'boosting_type': 'gbdt',
    'n_estimators': 100,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'seed': RANDOM_SEED,
    'n_jobs': -1,
    'verbose': -1
}
# --- Modifications for Stress Test ---
NUM_RANKERS = 10
# Train each ranker on a smaller, potentially less overlapping fraction
RANKER_TRAIN_QUERY_FRACTION = 0.2
# Assign highly varied relative logging volumes (these are multipliers for base passes)
# Example: Ranker 1 has 20x base passes, Ranker 4 has 0.5x
#RANKER_LOGGING_RATIOS = [100.0, 10.0, 1.0, 0.1]
RANKER_LOGGING_RATIOS = [1.0 for _ in range(NUM_RANKERS)]
# ----------------------------------

# Simulation Params
NUM_POSITIONS = 10
ETA = 1.0
EPSILON_MINUS = 0.1

# Estimator Params (Maybe slightly more epochs needed with more complex logs)
ALLPAIRS_EPOCHS = 4000

# --- Helper Functions (mostly unchanged, minor adjustments noted) ---

def load_yahoo_data(file_path):
    """Loads Yahoo LTR data (SVMLight format) into features, labels, query_ids."""
    X, y, qids = load_svmlight_file(file_path, query_id=True)
    y = y.astype(int)
    return X, y, qids

def get_query_doc_info(X, y, qids):
    """Organizes data into a DataFrame with doc IDs assigned per query."""
    rows = []
    unique_qids = np.unique(qids)
    qid_group_indices = np.split(np.arange(len(qids)), np.unique(qids, return_index=True)[1][1:])

    for i, qid in enumerate(unique_qids):
        indices = qid_group_indices[i]
        query_len = len(indices)
        for doc_idx_in_query in range(query_len):
            original_index = indices[doc_idx_in_query]
            rows.append({
                'query_id': qid,
                'doc_id': doc_idx_in_query,
                'relevance_grade': y[original_index],
                'feature_vector': X[original_index]
            })
    df = pd.DataFrame(rows)
    return df

def train_ranker(train_df, query_subset_ids, ranker_idx):
    """Trains an LGBM Ranker on a subset of queries."""
    print(f"  Training ranker {ranker_idx+1}/{NUM_RANKERS} on {len(query_subset_ids)} queries...")
    subset_df = train_df[train_df['query_id'].isin(query_subset_ids)].copy()
    if subset_df.empty:
        print(f"Warning: No data for ranker {ranker_idx+1} with provided query subset.")
        return None # Handle case of empty subset

    X_train = np.vstack(subset_df['feature_vector'].apply(lambda x: x.toarray()).values)
    y_train = subset_df['relevance_grade'].values
    group_train = subset_df.groupby('query_id').size().tolist()

    # Ensure group_train is not empty and matches X_train length
    if not group_train or sum(group_train) != len(X_train):
         print(f"Warning: Group information mismatch for ranker {ranker_idx+1}. Skipping training.")
         return None

    # Use a different seed for each ranker for variety if desired, but base seed is okay
    params = LGBM_PARAMS.copy()
    params['seed'] = RANDOM_SEED + ranker_idx

    lgbm_ranker = lgb.LGBMRanker(**params)
    lgbm_ranker.fit(X_train, y_train, group=group_train)
    del subset_df, X_train, y_train, group_train # Memory management
    gc.collect()
    return lgbm_ranker

def predict_rankings_for_ranker(model, data_df, ranker_idx, M=NUM_POSITIONS):
    """Predicts rankings for a single ranker."""
    rankings = {}
    unique_qids = data_df['query_id'].unique()
    print(f"  Predicting rankings for ranker {ranker_idx+1}/{NUM_RANKERS}...")

    # Process queries in chunks to manage memory
    chunk_size = 500
    for i in tqdm(range(0, len(unique_qids), chunk_size), desc=f"Ranker {ranker_idx+1} Predict", leave=False):
        batch_qids = unique_qids[i:min(i + chunk_size, len(unique_qids))]
        batch_data = data_df[data_df['query_id'].isin(batch_qids)].copy()

        if batch_data.empty: continue

        X_batch = np.vstack(batch_data['feature_vector'].apply(lambda x: x.toarray()).values)
        batch_groups = batch_data.groupby('query_id').size().tolist()
        doc_ids_flat = batch_data['doc_id'].values
        qids_flat = batch_data['query_id'].values # Needed if using predict's group argument

        if X_batch.shape[0] == 0: continue

        scores = model.predict(X_batch)

        # Reconstruct rankings per query
        current_idx = 0
        grouped_doc_ids = batch_data.groupby('query_id')['doc_id'].apply(list)
        for qid in batch_qids: # Iterate in the original order of the batch
             if qid not in grouped_doc_ids: continue # Should not happen if batch_data is correct
             num_docs_in_query = len(grouped_doc_ids[qid])
             query_scores = scores[current_idx : current_idx + num_docs_in_query]
             query_doc_ids = grouped_doc_ids[qid]

             ranked_indices = np.argsort(query_scores)[::-1]
             ranked_doc_ids = np.array(query_doc_ids)[ranked_indices] # Use numpy array for indexing
             rankings[qid] = ranked_doc_ids[:M]

             current_idx += num_docs_in_query
        del batch_data, X_batch, batch_groups, scores # Memory management
        gc.collect()

    return rankings


def calculate_prob_relevant(grade):
    """Maps relevance grade to probability."""
    if grade < 0 or grade > 4: return 0.0
    return (2**grade - 1) / (2**4 - 1)

def simulate_click(grade, position, eta=ETA, epsilon_minus=EPSILON_MINUS):
    """Simulates a click based on PBM, relevance, and noise."""
    if position <= 0: return 0
    prob_exam = (1.0 / position)**eta
    prob_rel = calculate_prob_relevant(grade)
    is_irrelevant = grade <= 1

    click_prob = prob_exam * (prob_rel + epsilon_minus * float(is_irrelevant))
    click = np.random.binomial(1, min(click_prob, 1.0))
    return click

def should_log_interaction(doc_id):
    if doc_id % int(1/ANCHOR_ITEM_PERCENTAGE) == 0:  # Anchor items occur frequently
        return True, True  # always log anchor items
    else:
        return np.random.rand() < SPARSE_ITEM_PROBABILITY, False

def generate_multi_ranker_log(all_rankings, test_df, base_num_impressions_per_ranker, M=NUM_POSITIONS):
    """
    Generates click log DataFrame from multiple rankers with specified impression counts
    AND simulates item-level frequency imbalance if enabled.
    """
    log_data = []
    num_rankers = len(all_rankings)
    print("Simulating clicks from multiple rankers with varied volumes AND item frequency imbalance...")

    test_qids = list(test_df['query_id'].unique())
    all_doc_grades = test_df.set_index(['query_id', 'doc_id'])['relevance_grade']

    total_simulated_impressions = sum(int(imp) for imp in base_num_impressions_per_ranker)
    processed_impressions = 0

    # We need to simulate total impressions and assign them proportionally to rankers
    # to ensure the item frequency applies correctly across ranker imbalances.
    # Let's simulate one impression at a time and decide which ranker 'served' it.

    ranker_indices = list(range(num_rankers))
    # Calculate probabilities based on impression counts/ratios
    total_base_imps = sum(base_num_impressions_per_ranker)
    ranker_probs = [imp / total_base_imps for imp in base_num_impressions_per_ranker]

    print(f"Total target simulated impressions: {total_base_imps}")
    print(f"Ranker serving probabilities: {ranker_probs}")

    # Simulate impression by impression
    # Use total_base_imps as the target number of potential logging opportunities
    for _ in tqdm(range(int(total_base_imps)), desc="Simulating Impressions", leave=False):
        # 1. Choose which ranker served this impression based on probs
        serving_ranker_idx = np.random.choice(ranker_indices, p=ranker_probs)
        rankings_for_r = all_rankings[serving_ranker_idx]

        # 2. Choose a query for this impression
        qid = random.choice(test_qids)
        if qid not in rankings_for_r: continue # Skip if ranker had no ranking

        # 3. Process the ranked list for this query/ranker
        ranked_docs = rankings_for_r[qid]
        for k, doc_id in enumerate(ranked_docs):
            position = k + 1

            # 4. ITEM FREQUENCY IMBALANCE CHECK
            should_log, is_anchor = should_log_interaction(doc_id)
            if not should_log:
                continue  # Skip logging this sparse interaction
            if is_anchor:
                num_imps = random.randint(1, ACHOR_IMPS)
            else:
                num_imps = random.randint(1, OTHER_IMPS)
            # 5. If logging is allowed, simulate click


            try:
                for t in range(num_imps):
                    grade = all_doc_grades.loc[(qid, doc_id)]
                    sim_click = simulate_click(grade, position)
                    # Log every considered interaction (impression) or only clicks?
                    # Logging impressions gives richer data for w(q,d,k) but is much larger.
                    # Logging only clicks is simpler for this estimator setup.
                    # Let's log the *impression* here to accurately reflect w(q,d,k) counts
                    # The estimator needs the 'impression' column if counts aren't uniform.
                    log_data.append({
                        'query_id': qid,
                        'item_id': doc_id,
                        'position': position,
                        'click': sim_click,
                        'impression': 1 # Log this as one impression event
                    })
            except KeyError:
                continue # Doc not found

    if not log_data:
         print("Warning: Log data is empty after simulation!")
         return pd.DataFrame(columns=['query_id', 'item_id', 'position', 'click', 'impression'])

    # Aggregate log data: sum clicks and count impressions for each (q, d, pos)
    # This is needed if the estimator expects aggregated input, or helpful for analysis.
    # The AllPairsEstimator in the toolkit *can* handle binary logs, but also aggregated.
    # Let's aggregate to be explicit.
    print("Aggregating simulated log data...")
    agg_log_df = pd.DataFrame(log_data).groupby(
        ['query_id', 'item_id', 'position']
    ).agg(
        click=('click', 'sum'),
        impression=('impression', 'count') # Count rows = number of impressions
    ).reset_index()

    print(f"Generated aggregated log with {len(agg_log_df)} (q, d, pos) entries.")
    return agg_log_df



def run_single_simulation(train_df, test_df, train_qids, test_qids, run_seed):
    """
    Performs one full simulation run with multiple rankers, varied logging volumes,
    and item frequency imbalance stress-testing. Returns estimated propensities.
    """
    # --- 1. Train Rankers ---
    print(f"\n--- Training {NUM_RANKERS} Rankers (Run Seed: {run_seed}) ---")
    # Seed random generators for reproducibility within the run
    random.seed(run_seed)
    np.random.seed(run_seed)
    trained_rankers = []

    # Create distinct query subsets for training
    train_qids_shuffled = random.sample(train_qids, len(train_qids))  # Shuffle to make subsets random
    queries_per_ranker = max(int(len(train_qids_shuffled) * RANKER_TRAIN_QUERY_FRACTION), 1)  # Ensure at least 1 query

    # Instead of distinct slices, use a sliding window approach with overlap
    for i in range(NUM_RANKERS):
        # Calculate start index with overlap to ensure data for all rankers
        start_idx = (i * queries_per_ranker) % len(train_qids_shuffled)
        # Handle wrap-around for the end index
        if start_idx + queries_per_ranker <= len(train_qids_shuffled):
            subset_qids = train_qids_shuffled[start_idx:start_idx + queries_per_ranker]
        else:
            # If we reach the end, wrap around to the beginning
            remaining = queries_per_ranker - (len(train_qids_shuffled) - start_idx)
            subset_qids = train_qids_shuffled[start_idx:] + train_qids_shuffled[:remaining]

        # No need to check for empty subset_qids as we've ensured each ranker gets data
        ranker = train_ranker(train_df, subset_qids, i)  # Pass ranker index 'i'
        trained_rankers.append(ranker)
        gc.collect()

    # --- 2. Predict Rankings on Test Set ---
    print(f"\n--- Predicting Rankings for {NUM_RANKERS} Trained Rankers ---")
    all_rankings = []
    for i, ranker in enumerate(trained_rankers):
        if ranker is None:
             print(f"Skipping prediction for ranker {i+1} (not trained).")
             all_rankings.append({}) # Store empty dict if ranker wasn't trained
             continue
        rankings = predict_rankings_for_ranker(ranker, test_df, i) # Pass ranker index
        all_rankings.append(rankings)
        gc.collect() # Clean up memory after prediction

    # --- 3. Generate Imbalanced Log with Item Frequency Sampling ---
    print("\n--- Generating Imbalanced Log with Item Freq Sampling ---")
    # Calculate target impression counts based on ratios and base (num test queries)
    num_test_queries = len(test_qids)
    base_impressions_per_pass = num_test_queries # Or another base value if desired
    impressions_per_ranker = [int(ratio * base_impressions_per_pass) for ratio in RANKER_LOGGING_RATIOS]

    # This function simulates impressions/clicks and returns an *aggregated* log DataFrame
    multi_ranker_log_df = generate_multi_ranker_log(all_rankings, test_df, impressions_per_ranker)

    # Check if log generation was successful
    if multi_ranker_log_df.empty:
        print("Warning: Generated aggregated log is empty after simulation! Returning NaN.")
        # Ensure memory is freed even on early exit
        del all_rankings, trained_rankers
        gc.collect()
        return {'original': np.full(NUM_POSITIONS, np.nan), 'modified': np.full(NUM_POSITIONS, np.nan)}



    print(f"Generated aggregated log with {len(multi_ranker_log_df)} (q, d, pos) entries.")
    # Free memory from rankings and models as they are no longer needed
    del all_rankings, trained_rankers
    gc.collect()

    # --- 4. Run Estimators on Aggregated Log ---
    print("\n--- Running AllPairs Estimators on Aggregated Log ---")
    results = {}
    estimator_orig = AllPairsEstimator(weighting="original", epochs=ALLPAIRS_EPOCHS)
    estimator_mod = AllPairsEstimator(weighting="variance_reduced", epochs=ALLPAIRS_EPOCHS)

    # Define column names expected by the estimator call for aggregated data
    query_col_name = 'query_id'
    doc_col_name = 'item_id'
    imps_col_name = 'impression' # Column with impression counts
    clicks_col_name = 'click'    # Column with click counts (sum)
    propensity_col = 'examination' # Expected output column name

    print("Running AllPairs (Original)...")
    try:
        # Call estimator with aggregated columns specified
        res_orig_df = estimator_orig(multi_ranker_log_df,
                                   query_col=query_col_name,
                                   doc_col=doc_col_name,
                                   imps_col=imps_col_name,
                                   clicks_col=clicks_col_name)

        # Process results (check column name and length)
        if isinstance(res_orig_df, pd.DataFrame) and propensity_col in res_orig_df.columns:
            estimates_orig = res_orig_df[propensity_col].values
            if len(estimates_orig) >= NUM_POSITIONS:
                results['original'] = estimates_orig[:NUM_POSITIONS]
            else:
                # Pad if shorter than expected
                print(f"Warning: Original estimator output length ({len(estimates_orig)}) < NUM_POSITIONS ({NUM_POSITIONS}). Padding.")
                padded_estimates = np.full(NUM_POSITIONS, np.nan)
                padded_estimates[:len(estimates_orig)] = estimates_orig
                results['original'] = padded_estimates
        # Optional: Handle dict output if estimator might return that
        # elif isinstance(res_orig_df, dict) and propensity_col in res_orig_df:
        #    # Process dict output appropriately
        #    results['original'] = np.array(res_orig_df[propensity_col][:NUM_POSITIONS]) # Example
        else:
             print(f"Warning: Original estimator output invalid or missing '{propensity_col}' column. Type: {type(res_orig_df)}")
             results['original'] = np.full(NUM_POSITIONS, np.nan)

    except Exception as e:
        print(f"ERROR running AllPairs (Original): {e}")
        results['original'] = np.full(NUM_POSITIONS, np.nan)

    print("Running AllPairs (Modified)...")
    try:
        # Call modified estimator with aggregated columns specified
        res_mod_df = estimator_mod(multi_ranker_log_df,
                                  query_col=query_col_name,
                                  doc_col=doc_col_name,
                                  imps_col=imps_col_name,
                                  clicks_col=clicks_col_name)

        # Process results (check column name and length)
        if isinstance(res_mod_df, pd.DataFrame) and propensity_col in res_mod_df.columns:
            estimates_mod = res_mod_df[propensity_col].values
            if len(estimates_mod) >= NUM_POSITIONS:
                results['modified'] = estimates_mod[:NUM_POSITIONS]
            else:
                # Pad if shorter than expected
                print(f"Warning: Modified estimator output length ({len(estimates_mod)}) < NUM_POSITIONS ({NUM_POSITIONS}). Padding.")
                padded_estimates = np.full(NUM_POSITIONS, np.nan)
                padded_estimates[:len(estimates_mod)] = estimates_mod
                results['modified'] = padded_estimates
        # Optional: Handle dict output
        # elif isinstance(res_mod_df, dict) and propensity_col in res_mod_df:
        #    results['modified'] = np.array(res_mod_df[propensity_col][:NUM_POSITIONS]) # Example
        else:
             print(f"Warning: Modified estimator output invalid or missing '{propensity_col}' column. Type: {type(res_mod_df)}")
             results['modified'] = np.full(NUM_POSITIONS, np.nan)

    except Exception as e:
        print(f"ERROR running AllPairs (Modified): {e}")
        results['modified'] = np.full(NUM_POSITIONS, np.nan)

    del multi_ranker_log_df # Clean up log DataFrame
    gc.collect()
    return results

# --- plot_and_analyze_yahoo (unchanged from previous version) ---
def plot_and_analyze_yahoo(results_original, results_modified, eta=ETA, M=NUM_POSITIONS, results_dir=RESULTS_DIR):
    """Plots results and calculates metrics for the Yahoo simulation."""
    print("\n--- Analyzing Yahoo Simulation Results ---")
    positions = np.arange(1, M + 1)

    # Ground Truth
    true_bias = np.array([(1.0 / k)**eta for k in positions])
    true_bias = true_bias / true_bias[0] # Normalize

    # Compute mean, variance, CI (using nanmean/nanvar)
    original_mean = np.nanmean(results_original, axis=0)
    original_var = np.nanvar(results_original, axis=0)
    original_std = np.nanstd(results_original, axis=0)
    valid_counts_orig = np.sum(~np.isnan(results_original), axis=0)
    original_ci = 1.96 * original_std / np.sqrt(np.maximum(valid_counts_orig, 1))

    modified_mean = np.nanmean(results_modified, axis=0)
    modified_var = np.nanvar(results_modified, axis=0)
    modified_std = np.nanstd(results_modified, axis=0)
    valid_counts_mod = np.sum(~np.isnan(results_modified), axis=0)
    modified_ci = 1.96 * modified_std / np.sqrt(np.maximum(valid_counts_mod, 1))

    # --- Plot Estimates ---
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    ax1.plot(positions, true_bias, 'k-', label='Ground Truth', linewidth=2, zorder=5)
    ax1.errorbar(positions, original_mean, yerr=original_ci, fmt='ro-', label='Original Weighting', alpha=0.7, capsize=3, markersize=5)
    ax1.errorbar(positions, modified_mean, yerr=modified_ci, fmt='bo-', label='Modified Weighting', alpha=0.7, capsize=3, markersize=5)
    ax1.set_xlabel('Position', fontsize=14)
    ax1.set_ylabel('Relative Propensity (Normalized)', fontsize=14)
    # Update title to reflect new setup
    ax1.set_title(f'AllPairs Estimates on Simulated Yahoo Data ({NUM_RANKERS} Rankers, Varied Imbalance, Avg over {results_original.shape[0]} runs)', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.4)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plot_filename1 = os.path.join(results_dir, "yahoo_AllPairs_estimates_stress_test.png")
    plt.savefig(plot_filename1)
    print(f"Saved estimates plot: {plot_filename1}")
    plt.close(fig1)

    # --- Calculate Metrics ---
    mask = ~np.isnan(original_mean) & ~np.isnan(modified_mean) & ~np.isnan(true_bias)
    original_mse = np.mean((original_mean[mask] - true_bias[mask])**2) if np.any(mask) else np.nan
    modified_mse = np.mean((modified_mean[mask] - true_bias[mask])**2) if np.any(mask) else np.nan

    mean_orig_var = np.nanmean(original_var)
    mean_mod_var = np.nanmean(modified_var)
    variance_reduction_pct = np.nan
    if mean_orig_var > 0 and not np.isnan(mean_orig_var) and not np.isnan(mean_mod_var):
        variance_reduction_pct = (1 - mean_mod_var / mean_orig_var) * 100

    print(f"\n--- Quantitative Results (Avg over {results_original.shape[0]} runs) ---")
    print(f"  Avg MSE (Original): {original_mse:.6f}")
    print(f"  Avg MSE (Modified): {modified_mse:.6f}")
    print(f"  Mean Var (Original): {mean_orig_var:.6f}")
    print(f"  Mean Var (Modified): {mean_mod_var:.6f}")
    print(f"  Overall Variance Reduction: {variance_reduction_pct:.2f}%")

    # Format for LaTeX table
    print("\n--- LaTeX Table Data ---")
    print(f"{'Estimator':<10} {'Weighting':<10} {'Avg MSE':<10} {'Mean Var':<10} {'Var Reduc (%)'}")
    print("-" * 55)
    print(f"{'AllPairs':<10} {'Original':<10} {original_mse:.6f}   {mean_orig_var:.6f}   {'--':<15}")
    print(f"{'':<10} {'Modified':<10} {modified_mse:.6f}   {mean_mod_var:.6f}   {variance_reduction_pct:.2f} %")
    print("-" * 55)

    return {'mse_orig': original_mse, 'mse_mod': modified_mse,
            'var_orig': mean_orig_var, 'var_mod': mean_mod_var,
            'var_reduction_pct': variance_reduction_pct}


# --- Main Execution ---
if __name__ == "__main__":
    # File path checks... (same as before)
    if not (os.path.exists(YAHOO_TRAIN_PATH) and os.path.exists(YAHOO_TEST_PATH)):
        print("ERROR: Yahoo train.txt or test.txt not found.")
        # ... (print instructions)
        exit()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Yahoo training data...")
    X_train_raw, y_train_raw, qids_train_raw = load_yahoo_data(YAHOO_TRAIN_PATH)
    train_df = get_query_doc_info(X_train_raw, y_train_raw, qids_train_raw)
    train_qids = list(train_df['query_id'].unique())
    del X_train_raw, y_train_raw, qids_train_raw; gc.collect()

    print("Loading Yahoo test data...")
    X_test_raw, y_test_raw, qids_test_raw = load_yahoo_data(YAHOO_TEST_PATH)
    test_df = get_query_doc_info(X_test_raw, y_test_raw, qids_test_raw)
    test_qids = list(test_df['query_id'].unique())
    del X_test_raw, y_test_raw, qids_test_raw; gc.collect()

    print(f"Loaded {len(train_qids)} training queries and {len(test_qids)} test queries.")

    # --- Run Multiple Simulations ---
    results_all_runs_orig = np.zeros((NUM_RUNS, NUM_POSITIONS))
    results_all_runs_mod = np.zeros((NUM_RUNS, NUM_POSITIONS))

    for i in range(NUM_RUNS):
        print(f"\n=== Starting STRESS TEST Simulation Run {i+1}/{NUM_RUNS} ===")
        run_seed = RANDOM_SEED + i
        run_results = run_single_simulation(train_df, test_df, train_qids, test_qids, run_seed)
        results_all_runs_orig[i, :] = run_results.get('original', np.full(NUM_POSITIONS, np.nan))
        results_all_runs_mod[i, :] = run_results.get('modified', np.full(NUM_POSITIONS, np.nan))
        gc.collect() # Clean up memory between runs

    # --- Analyze and Plot ---
    summary_stats = plot_and_analyze_yahoo(results_all_runs_orig, results_all_runs_mod, results_dir=RESULTS_DIR)

    print("\nYahoo STRESS TEST simulation finished.")

# --- END OF FILE yahoo_simulation_stress_test.py ---