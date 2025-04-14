from ahl_targets.utils.io import load_s3_data
from ahl_targets.utils.io import load_with_encoding
from ahl_targets import PROJECT_DIR, BUCKET_NAME
from nesta_ds_utils.loading_saving.S3 import download_obj
from boto3.s3.transfer import TransferConfig

import pandas as pd


def energy_density_agg() -> pd.DataFrame:
    """ """
    return download_obj(
        BUCKET_NAME,
        "in_home/processed/targets/ed_agg.csv",
        download_as="dataframe",
    )


def npm_agg(detailed=False) -> pd.DataFrame:
    """ """
    if detailed:
        return download_obj(
            BUCKET_NAME,
            "in_home/processed/targets/npm_agg_detailed.parquet",
            download_as="dataframe",
            kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
        )
    else:
        return download_obj(
            BUCKET_NAME,
            "in_home/processed/targets/npm_agg.csv",
            download_as="dataframe",
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


def regression_df() -> pd.DataFrame:
    return pd.read_csv(
        load_with_encoding(
            "ahl-private-data", "in_home/processed/targets/ed_npm_regression_output.csv"
        ),
        encoding="ISO-8859-1",
    )


def npm_robustness() -> pd.DataFrame:
    return download_obj(
        BUCKET_NAME,
        "in_home/processed/targets/npm_robustness.csv",
        download_as="dataframe",
    )
