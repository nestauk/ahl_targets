"""Getters for files required for the 2024 updates. Some of these are taken from the diets repository: https://github.com/nestauk/ahl_diets_evidence"""

# Imports
from nesta_ds_utils.loading_saving.S3 import download_obj
import pandas as pd
from ahl_targets import BUCKET_NAME
from boto3.s3.transfer import TransferConfig


# Functions
def new_model_data() -> pd.DataFrame:
    """Reads the new model data file.

    Returns:
        pd.DataFrame: new model data
    """

    return download_obj(
        BUCKET_NAME,
        "in_home/processed/retailer_targets_baseline/baseline_retailer_targets_input_2024.parquet",
        download_as="dataframe",
        kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
    )


def coefficients_2024() -> pd.DataFrame:
    """Reads the new coefficients file.

    Returns:
        pd.DataFrame: new coefficients
    """

    return download_obj(
        BUCKET_NAME,
        "in_home/processed/targets/oct_24_update/coefficients.parquet",
        download_as="dataframe",
        kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
    )


def inhome_purchase_apr_dec():
    """Returns the in-home purchase subset based on project requirements.
    Returns:
        pd.DataFrame: in-home purchase subset
    """
    return download_obj(
        BUCKET_NAME,
        "diets_sotn/purchase_files_subsetted/inhome_purchase_apr_dec.parquet",
        download_as="dataframe",
        kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
    )


def nutrition_inhome() -> pd.DataFrame:
    """Reads the nutrition data file.

    Returns:
        pd.DataFrame: nutrition data
    """

    return download_obj(
        BUCKET_NAME,
        "in_home/processed/nutrition_clean.parquet",
        download_as="dataframe",
        kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
    )


def product_table_inhome() -> pd.DataFrame:
    """Reads the product table file.

    Returns:
        pd.DataFrame: product table
    """

    return download_obj(
        BUCKET_NAME,
        "in_home/processed/product_metadata.csv",
        download_as="dataframe",
    )


def store_table_inhome() -> pd.DataFrame:
    """Reads the store table file.

    Returns:
        pd.DataFrame: store table
    """

    return download_obj(
        BUCKET_NAME,
        "in_home/processed/store_table.csv",
        download_as="dataframe",
    )


def get_demographics_data():
    """Returns the household demographics data for all individuals in the both in home and out of home panels.
    Returns:
        pd.DataFrame: demographics data
    """
    return download_obj(
        BUCKET_NAME,
        "ooh/processed/household_demog_table_v3.csv",
        download_as="dataframe",
    )


def get_products_to_drop():
    """Returns a series of products that were not included in the original analysis (mostly due to missing NPM scores).
    Returns:
        pd.Series: unique_ids of products to drop
    """
    return download_obj(
        BUCKET_NAME,
        "in_home/processed/targets/oct_24_update/additions_to_remove.csv",
        download_as="dataframe",
    )


def df_npm_2024():
    """Returns the new model data (2024) with npm merged on.
    Returns:
        pd.DataFrame: new model data
    """
    return download_obj(
        BUCKET_NAME,
        "in_home/processed/targets/oct_24_update/df_npm.parquet",
        download_as="dataframe",
        kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
    )


def get_agg_data_2024():
    """Returns the new model data (2024) aggregated by store. *This is the input to the npm_simulation model in* `ahl_targets/pipeline/2024_update/simulation_npm.py`."""
    return download_obj(
        BUCKET_NAME,
        "in_home/processed/targets/oct_24_update/store_weight.parquet",
        download_as="dataframe",
        kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
    )
