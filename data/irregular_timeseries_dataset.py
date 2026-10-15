# data/irregular_timeseries_dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

class IrregularTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-IV style irregular physiological time series.
    Features: vitals + Δt (time since last measurement) + mask (1=observed, 0=imputed)
    """
    def __init__(
        self,
        data_list: List[pd.DataFrame],           # list of per-stay DataFrames [charttime, feature1, feature2, ...]
        feature_cols: List[str],
        time_col: str = 'charttime',
        label_col: Optional[str] = 'label',      # 1 if decomp within h after window end
        window_len: int = 48,
        stride: int = 12,
        max_delta_hours: float = 4.0,            # cap large gaps
        fill_method: str = 'ffill'
    ):
        self.windows = []
        self.labels = []
        self.stay_ids = []  # for debugging/tracking

        for stay_id, df in enumerate(data_list):
            if len(df) < window_len:
                continue

            df = df.sort_values(time_col).reset_index(drop=True)
            df[time_col] = pd.to_datetime(df[time_col])

            # Compute deltas (in hours)
            deltas = df[time_col].diff().dt.total_seconds() / 3600.0
            deltas = deltas.fillna(0).clip(upper=max_delta_hours)

            # Forward fill values
            if fill_method == 'ffill':
                values = df[feature_cols].ffill().fillna(0).values  # or mean imputation
            else:
                values = df[feature_cols].fillna(0).values

            deltas = deltas.values.reshape(-1, 1)
            mask = (~df[feature_cols].isna().values).astype(np.float32)  # 1=observed

            # Concat: values + delta + mask → shape [T, num_features * 3]
            full_data = np.concatenate([values, deltas, mask], axis=1)
            num_features = full_data.shape[1]

            # Sliding windows
            for start in range(0, len(full_data) - window_len + 1, stride):
                window = full_data[start : start + window_len]
                # Label: look ahead from END of window
                end_idx = start + window_len - 1
                if label_col and label_col in df.columns:
                    label = df.iloc[end_idx][label_col]
                else:
                    label = 0  # placeholder

                self.windows.append(torch.tensor(window, dtype=torch.float32))
                self.labels.append(label)
                self.stay_ids.append(stay_id)

        print(f"Generated {len(self.windows)} windows from {len(data_list)} stays")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.windows[idx]          # [L, d*3]  values + Δt + mask
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
