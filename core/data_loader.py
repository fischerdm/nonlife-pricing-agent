import pandas as pd


def load_dataset(data_cfg: dict) -> pd.DataFrame:
    """Load the raw dataset and apply the zero-premium filter, per project_config.yaml."""
    df = pd.read_csv(
        data_cfg["path"],
        sep=data_cfg.get("sep", ","),
        low_memory=False,
    )
    if data_cfg.get("filter_zeros"):
        target_col = data_cfg["target_col"]
        df = df[df[target_col] > 0].copy()
    return df
