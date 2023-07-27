from ahl_targets.utils.io import load_s3_data
from ahl_targets.utils.io import load_with_encoding

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
