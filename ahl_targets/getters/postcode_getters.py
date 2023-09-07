# Getters for misc datasets

from typing import Dict
import pandas as pd
from ahl_targets import PROJECT_DIR


def nspl():
    """Read the National Postcode lookup"""

    return pd.read_csv(
        f"{PROJECT_DIR}/inputs/raw/NSPL/Data/NSPL_UK.csv", low_memory=False
    )


def region_name_lookup() -> Dict[str, str]:
    """Read the region to name lookup"""

    return (
        pd.read_csv(
            f"{PROJECT_DIR}/inputs/raw/NSPL/Documents/Region names and codes EN as at 12_20 (RGN).csv"
        )
        .assign(
            RGN20NM=lambda df: df["RGN20NM"].str.replace("\(pseudo\) ", "", regex=True)
        )
        .set_index("RGN20CD")["RGN20NM"]
        .to_dict()
    )


def lsoa_name_lookup() -> Dict[str, str]:
    """Read the region to name lookup"""

    return (
        pd.read_csv(
            f"{PROJECT_DIR}/inputs/raw/NSPL/Documents/LSOA (2011) names and codes UK as at 12_12.csv"
        )
        .set_index("LSOA11CD")["LSOA11NM"]
        .to_dict()
    )


def laua_name_lookup() -> Dict[str, str]:
    """Read the region to name lookup"""

    return (
        pd.read_csv(
            f"{PROJECT_DIR}/inputs/raw/NSPL/Documents/LA_UA names and codes UK as at 04_21.csv"
        )
        .set_index("LAD21CD")["LAD21NM"]
        .to_dict()
    )


def postcode_region_lookup() -> Dict[str, str]:
    """Creates a postcode - region name lookup"""

    return (
        nspl()[["pcds", "rgn"]]
        .assign(pcode_distr=lambda df: df["pcds"].str.split(" ").str[0])
        .assign(region_name=lambda df: df["rgn"].map(region_name_lookup()))
        .set_index("pcode_distr")["region_name"]
        .to_dict()
    )


def postcode_lsoa_lookup() -> Dict[str, str]:
    """Creates a postcode - region name lookup"""

    return (
        nspl()[["pcds", "lsoa11"]]
        .assign(pcode_distr=lambda df: df["pcds"].str.split(" ").str[0])
        .assign(lsoa=lambda df: df["lsoa11"].map(lsoa_name_lookup()))
        .set_index("pcode_distr")["lsoa"]
        .to_dict()
    )


def postcode_laua_lookup() -> Dict[str, str]:
    """Creates a postcode - region name lookup"""

    return (
        nspl()[["pcds", "laua"]]
        .assign(pcode_distr=lambda df: df["pcds"].str.split(" ").str[0])
        .assign(la_ua=lambda df: df["laua"].map(laua_name_lookup()))
        .set_index("pcode_distr")["la_ua"]
        .to_dict()
    )
