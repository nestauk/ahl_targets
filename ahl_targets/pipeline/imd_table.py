"""
Table in S3 with local authority, household size, age of main shopper,
social class, ethnicity, life stage
"""

# Import packages
import pandas as pd
from ahl_targets.getters import get_data
from ahl_targets.utils import create_tables as tables
from ahl_targets.getters.postcode_getters import (
    postcode_region_lookup,
    postcode_lsoa_lookup,
    postcode_laua_lookup,
)
from ahl_targets import PROJECT_DIR

# Read in data
pan_mast = get_data.household_master()
demog_coding = get_data.demog_coding()
demog_val = get_data.demog_val()
# IMD
imd = pd.read_csv(
    f"{PROJECT_DIR}/inputs/raw/imd/File_1_-_IMD2019_Index_of_Multiple_Deprivation.csv"
)


def postcode_clean(original_dict):
    """Removes whitespace from the postcode part of a postcode - area name
    dictionary.
    """
    # Remove whitespace from postcode
    return {
        postcode.replace(" ", ""): area_name
        for postcode, area_name in original_dict.items()
    }


def households_table_with_location(
    demog_coding: pd.DataFrame, demog_val: pd.DataFrame, pan_mast: pd.DataFrame
) -> pd.DataFrame:
    """Creates a table with household demographics including region,
    lsoa and laua names.
    Args:
        demog_coding: demographics coding table
        demog_val: demographics value table
        pan_mast: household master table
    Returns:
        households_table_with_location: table with household demographics
    """

    # Create table
    hh_demographics = tables.hh_demographic_table(demog_coding, demog_val, pan_mast)[
        [
            "Postocde District",
        ]
    ]

    hh_demographics["Postocde District"] = hh_demographics[
        "Postocde District"
    ].str.strip()

    # Add region, lsoa and laua names to table using postcode lookup
    return (
        hh_demographics.assign(
            region=lambda hh_demographics: hh_demographics["Postocde District"].map(
                postcode_clean(postcode_region_lookup())
            )
        )
        .assign(
            lsoa_name=lambda hh_demographics: hh_demographics["Postocde District"].map(
                postcode_clean(postcode_lsoa_lookup())
            )
        )
        .assign(
            laua_name=lambda hh_demographics: hh_demographics["Postocde District"].map(
                postcode_clean(postcode_laua_lookup())
            )
        )
    )


hh_table = households_table_with_location(demog_coding, demog_val, pan_mast)

hh_table_imd = (
    hh_table.reset_index()
    .copy()
    .merge(
        imd[["LSOA name (2011)", "Index of Multiple Deprivation (IMD) Decile"]],
        left_on="lsoa_name",
        right_on="LSOA name (2011)",
        how="left",
    )
    .drop(columns=["LSOA name (2011)"])
)

# Check % of missing in lsoa_name and laua_name
print(
    f"Missing lsoa_name: {hh_table['lsoa_name'].isna().sum() / len(hh_table) * 100:.2f}%"
)
# Check % of missing on IMD decile
print(
    f"Missing IMD decile: {hh_table_imd['Index of Multiple Deprivation (IMD) Decile'].isna().sum() / len(hh_table_imd) * 100:.2f}%"
)

# Save household demographics table
hh_table_imd.to_csv(f"{PROJECT_DIR}/outputs/data/hh_table_imd.csv", index=False)
