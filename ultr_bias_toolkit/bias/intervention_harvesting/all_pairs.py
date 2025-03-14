import logging
from typing import Optional, Literal

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ultr_bias_toolkit.bias.intervention_harvesting.util import build_intervention_sets
from ultr_bias_toolkit.bias.intervention_harvesting.util import normalize_bias
from ultr_bias_toolkit.util.assertions import assert_columns_in_df

logger = logging.getLogger(__name__)


class AllPairsEstimator:
    def __init__(
        self,
        epochs: int = 5_000,
        lr: float = 0.01,
        batch_size: int = 512,
        weighting: Literal["original", "variance_reduced"] = "original",
    ):
        """
        Initialize the All Pairs estimator.
        
        Args:
            epochs: Number of training epochs (default=5000)
            lr: Learning rate (default=0.01)
            batch_size: Batch size for training (default=512)
            weighting: Weighting scheme to use (default="original")
                - "original": Standard weighting
                - "variance_reduced": Modified weighting that reduces variance while maintaining unbiasedness
        """
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weighting = weighting

    def __call__(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        doc_col: str = "doc_id",
        imps_col: Optional[str] = None,
        clicks_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Estimates position bias using the all pairs method with neural network training.
        
        Args:
            df: DataFrame with click data
            query_col: Name of the column containing query identifiers
            doc_col: Name of the column containing document identifiers
            imps_col: Optional column with impression counts (for aggregated data)
            clicks_col: Optional column with click counts (for aggregated data)
            
        Returns:
            DataFrame with position and examination probability
        """
        logger.info(f"Position bias estimation using global all pairs estimator")
        logger.info(f"Using weighting scheme: {self.weighting}")
        
        n_positions = df.position.nunique()
        max_position = df.position.max()
        
        # Handle different input formats
        if imps_col is not None and clicks_col is not None:
            # Pre-aggregated format
            assert_columns_in_df(df, ["position", query_col, doc_col, imps_col, clicks_col])
            df = build_intervention_sets(
                df, query_col, doc_col, imps_col, clicks_col, 
                weighting=self.weighting
            )
        else:
            # Original binary format
            assert_columns_in_df(df, ["position", query_col, doc_col, "click"])
            df = build_intervention_sets(
                df, query_col, doc_col, 
                weighting=self.weighting
            )
        
        df = df[df.position_0 != df.position_1].copy()

        dataset = AllPairsDataset(df)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = AllPairsPBM(n_positions)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for _ in tqdm(range(self.epochs), desc="Maximizing weighted cross entropy..."):
            losses = []

            for batch in loader:
                k, k_prime, c_0, not_c_0 = batch
                optimizer.zero_grad()

                y_predict = model(k, k_prime)
                loss = -(
                    c_0 * torch.log(y_predict) + not_c_0 * torch.log(1 - y_predict)
                ).sum()
                loss.backward()
                losses.append(loss.detach().item())

                optimizer.step()

        position, examination = model.get_position_bias(max_position)
        df = pd.DataFrame({"position": position, "examination": examination})
        df.examination = normalize_bias(df.examination)
        return df


class AllPairsPBM(nn.Module):
    def __init__(self, n_positions: int):
        super().__init__()
        k = n_positions + 1
        self.examination = nn.Sequential(nn.Embedding(k, 1), nn.Sigmoid())
        self.relevance = nn.Sequential(nn.Embedding(k * k, 1), nn.Sigmoid())

    def forward(self, k: torch.Tensor, k_prime: torch.Tensor):
        examination = self.examination(k)
        relevance = self.relevance(k * k_prime)
        return (examination * relevance).squeeze(dim=-1)

    def get_position_bias(self, k: int):
        positions = torch.arange(k) + 1
        examination = self.examination(positions).squeeze()
        return positions, examination.detach()


class AllPairsDataset(Dataset):
    def __init__(self, df):
        self.k = torch.tensor(df["position_0"].values)
        self.k_prime = torch.tensor(df["position_1"].values)
        self.c_0 = torch.tensor(df["c_0"].values, dtype=torch.float)
        self.not_c_0 = torch.tensor(df["not_c_0"].values, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.k)

    def __getitem__(self, idx: int):
        return self.k[idx], self.k_prime[idx], self.c_0[idx], self.not_c_0[idx]
