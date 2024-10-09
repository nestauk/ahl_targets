"""Getters for up to date files taken from diets"""

# Imports
from nesta_ds_utils.loading_saving.S3 import download_obj
import pandas as pd
from ahl_targets import BUCKET_NAME
from ahl_targets import PROJECT_DIR
import boto3
from typing import Dict, Any
from boto3.s3.transfer import TransferConfig


# Functions
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
