from ahl_targets.utils.io import load_s3_data
from ahl_targets.utils.io import load_with_encoding
from ahl_targets import PROJECT_DIR

import pandas as pd


def energy_density_agg() -> pd.DataFrame:
    """ """

    return pd.read_csv(
        load_with_encoding(
            "ahl-private-data", "in_home/data_outputs/targets_annex/ed_agg.csv"
        ),
        encoding="ISO-8859-1",
    )


def energy_density_full() -> pd.DataFrame:
    """ """

    return pd.read_csv(
        load_s3_data(
            "ahl-private-data", "in_home/data_outputs/targets_annex/ed_full.csv"
        ),
    )


def npm_agg() -> pd.DataFrame:
    """ """

    return pd.read_csv(
        load_with_encoding("ahl-private-data", "in_home/processed/targets/npm_agg.csv"),
        encoding="ISO-8859-1",
    )


def coefficients_df() -> pd.DataFrame:
    return pd.read_csv(
        load_with_encoding(
            "ahl-private-data", "in_home/processed/targets/coefficients.csv"
        ),
        encoding="ISO-8859-1",
    )


def hfss_agg() -> pd.DataFrame:
    """ """

    return pd.read_csv(
        load_with_encoding(
            "ahl-private-data", "in_home/data_outputs/targets_annex/hfss_agg.csv"
        ),
        encoding="ISO-8859-1",
    )


def hfss_full() -> pd.DataFrame:
    return pd.read_csv(PROJECT_DIR / "inputs/processed/hfss_full.csv")
