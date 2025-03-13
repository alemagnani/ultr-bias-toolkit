# Personal toolkit for bias estimation in unbiased learning to rank

## Installation
```
pip install ultr-bias-toolkit
```

## Offline bias estimation methods
We implement multiple offline position bias estimation methods, including [three intervention harvesting](https://arxiv.org/abs/1812.05161) approaches:

```python
import pandas as pd
from ultr_bias_toolkit.bias.naive import NaiveCtrEstimator
from ultr_bias_toolkit.bias.intervention_harvesting import PivotEstimator, AdjacentChainEstimator, AllPairsEstimator

estimators = {
    "CTR Rate": NaiveCtrEstimator(),
    "Pivot One": PivotEstimator(pivot_rank=1),
    "Adjacent Chain": AdjacentChainEstimator(),
    "Global All Pairs": AllPairsEstimator(),
}
examination_dfs = []

for name, estimator in estimators.items():
    examination_df = estimator(df)
    examination_df["estimator"] = name
    examination_dfs.append(examination_df)

examination_df = pd.concat(examination_dfs)
examination_df.head()
```

## Usage Examples

### Example 1: Naive CTR Estimator
The simplest approach - estimates examination probability at each position using click-through rate:

```python
import pandas as pd
from ultr_bias_toolkit.bias.naive import NaiveCtrEstimator

# Sample click log data
click_data = pd.DataFrame({
    "query_id": [1, 1, 1, 2, 2, 2],
    "doc_id": ["a", "b", "c", "d", "e", "f"],
    "position": [1, 2, 3, 1, 2, 3],
    "click": [1, 0, 0, 1, 1, 0]
})

# Estimate position bias (examination probability)
estimator = NaiveCtrEstimator()
examination_df = estimator(click_data)
print(examination_df)
# Output:
#    position  examination
# 0         1          1.0
# 1         2          0.5
# 2         3          0.0
```

### Example 2: Pivot Estimator
Uses a pivot rank to estimate examination probability at other positions:

```python
import pandas as pd
from ultr_bias_toolkit.bias.intervention_harvesting import PivotEstimator

# Sample click log data with intervention pairs (same document at different positions)
click_data = pd.DataFrame({
    "query_id": [1, 1, 1, 1, 2, 2, 2, 2],
    "doc_id": ["a", "b", "a", "c", "d", "e", "d", "f"],
    "position": [1, 2, 3, 4, 1, 2, 3, 4],
    "click": [1, 0, 0, 0, 1, 1, 0, 0]
})

# Create estimator with position 1 as pivot
pivot_estimator = PivotEstimator(pivot_rank=1)
examination_df = pivot_estimator(click_data)
print(examination_df)
# This will show examination probability at each position relative to position 1
```

### Example 3: Adjacent Chain Estimator
Estimates examination probability using click ratios between adjacent positions:

```python
import pandas as pd
from ultr_bias_toolkit.bias.intervention_harvesting import AdjacentChainEstimator

# Sample click log data with intervention pairs
click_data = pd.DataFrame({
    "query_id": [1, 1, 1, 1, 2, 2, 2, 2],
    "doc_id": ["a", "b", "b", "c", "d", "e", "e", "f"],
    "position": [1, 2, 3, 4, 1, 2, 3, 4],
    "click": [1, 1, 0, 0, 1, 1, 0, 0]
})

# Estimate examination probability using adjacent chain method
adjacent_estimator = AdjacentChainEstimator()
examination_df = adjacent_estimator(click_data)
print(examination_df)
# Shows examination probability for each position
```

### Example 4: Comparing Multiple Estimators
Visualize and compare examination probabilities from different methods:

```python
import pandas as pd
import matplotlib.pyplot as plt
import random
from ultr_bias_toolkit.bias.naive import NaiveCtrEstimator
from ultr_bias_toolkit.bias.intervention_harvesting import PivotEstimator, AdjacentChainEstimator

# Create a larger sample dataset with intervention pairs
data = []
for q in range(1, 101):
    for p in range(1, 11):
        # Decreasing probability of click by position
        click_prob = max(0, 1 - (p-1)*0.1)
        click = 1 if random.random() < click_prob else 0
        doc_id = f"doc_{q}_{p}"
        
        # Add document at original position
        data.append({
            "query_id": q,
            "doc_id": doc_id,
            "position": p,
            "click": click
        })
        
        # Add same document at a different position (for intervention pairs)
        if p < 10:
            other_pos = p + 1
            click_prob = max(0, 1 - (other_pos-1)*0.1)
            click = 1 if random.random() < click_prob else 0
            data.append({
                "query_id": q + 1000,  # Different query
                "doc_id": doc_id,      # Same document
                "position": other_pos,
                "click": click
            })

click_data = pd.DataFrame(data)

# Define estimators
estimators = {
    "Naive CTR": NaiveCtrEstimator(),
    "Pivot (rank 1)": PivotEstimator(pivot_rank=1),
    "Adjacent Chain": AdjacentChainEstimator()
}

# Apply each estimator
results = []
for name, estimator in estimators.items():
    result = estimator(click_data)
    result["method"] = name
    results.append(result)

all_results = pd.concat(results)

# Visualize examination probabilities
plt.figure(figsize=(10, 6))
for name, group in all_results.groupby("method"):
    plt.plot(group["position"], group["examination"], marker='o', label=name)

plt.xlabel("Position")
plt.ylabel("Examination Probability")
plt.title("Position Bias Estimation Comparison")
plt.legend()
plt.grid(True)
plt.show()
```

### Example 5: Using Aggregated Click Data Format
For large datasets, you can use the more efficient aggregated format with impression and click counts instead of binary click values:

```python
import pandas as pd
from ultr_bias_toolkit.bias.intervention_harvesting import PivotEstimator

# Create pre-aggregated click data
# Each row represents multiple impressions with their click counts
aggregated_data = pd.DataFrame({
    "query_id": [1, 1, 2, 2, 3, 3],
    "doc_id": ["a", "a", "b", "b", "c", "c"],
    "position": [1, 2, 1, 3, 2, 4],
    "impressions": [500, 500, 1000, 1000, 750, 750],  # Number of impressions
    "clicks": [150, 75, 300, 100, 180, 40]           # Number of clicks
})

# Create estimator and call with aggregated data
pivot_estimator = PivotEstimator(pivot_rank=1)
examination_df = pivot_estimator(
    aggregated_data, 
    query_col="query_id", 
    doc_col="doc_id",
    imps_col="impressions",  # Specify impression count column
    clicks_col="clicks"      # Specify click count column
)

print(examination_df)
```

This format is much more efficient for large datasets as it avoids creating millions of rows with binary clicks.

### Example 6: Using Variance-Reduced Weighting
The toolkit supports a modified weighting scheme that reduces variance while maintaining unbiasedness:

```python
import pandas as pd
import matplotlib.pyplot as plt
from ultr_bias_toolkit.bias.intervention_harvesting import PivotEstimator, AdjacentChainEstimator

# Create click data
click_data = pd.DataFrame({
    "query_id": [1, 1, 1, 1, 2, 2, 2, 2],
    "doc_id": ["a", "b", "a", "c", "d", "e", "d", "f"],
    "position": [1, 2, 3, 4, 1, 2, 3, 4],
    "impressions": [500, 300, 100, 200, 800, 400, 150, 250],
    "clicks": [150, 60, 10, 15, 240, 80, 15, 20]
})

# Compare original and variance-reduced weighting
estimators = {
    "Original Pivot": PivotEstimator(pivot_rank=1, weighting="original"),
    "Variance-Reduced Pivot": PivotEstimator(pivot_rank=1, weighting="variance_reduced"),
    "Original Adjacent Chain": AdjacentChainEstimator(weighting="original"),
    "Variance-Reduced Adjacent Chain": AdjacentChainEstimator(weighting="variance_reduced")
}

results = []
for name, estimator in estimators.items():
    result = estimator(
        click_data, 
        imps_col="impressions", 
        clicks_col="clicks"
    )
    result["method"] = name
    results.append(result)

all_results = pd.concat(results)

# Visualize and compare the results
plt.figure(figsize=(12, 6))
for name, group in all_results.groupby("method"):
    plt.plot(group["position"], group["examination"], marker='o', label=name)
plt.xlabel("Position")
plt.ylabel("Examination Probability")
plt.title("Comparison of Original vs. Variance-Reduced Weighting")
plt.legend()
plt.grid(True)
plt.show()
```

The variance-reduced weighting scheme (`weighting="variance_reduced"`) produces more stable estimates particularly when there are large imbalances in impression counts across positions.

For more complex scenarios and the All Pairs Estimator which uses a neural network approach, refer to the code examples in the documentation.
