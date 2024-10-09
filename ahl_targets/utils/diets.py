"""These are utils functions taken from the diets repo"""

import pandas as pd
import numpy as np


def ind_size_conv(pan_ind: pd.DataFrame) -> pd.DataFrame:
    """
    !!! Outdated function !!! - Use ind_size_conv_v2 instead

    Using the adult equivilent conversion factor for reccomended daily intake of kcal.
    Creates prop_intake field for each individual, which represents the proportion of the household size
    (using the adult equivilent conversion factor for reccomended daily intake of kcal) that the individual represents.
    Link: https://www.researchgate.net/figure/Adult-equivalent-conversion-factors-for-estimated-calorie-requirements-according-to-age_tbl1_49704894

    Args:
        pan_ind (pd.DataFrame): Pandas dataframe of household members
    Returns:
        pd.DateFrame: Household converted total size
    """
    age_group = ["0-1", "1-3", "4-6", "7-10", "11-14", "15-18", "19-24", "25-50", "51+"]

    d = {
        "age_group": [
            "0-1",
            "1-3",
            "4-6",
            "7-10",
            "11-14",
            "15-18",
            "19-24",
            "25-50",
            "51+",
            "11-14",
            "15-18",
            "19-24",
            "25-50",
            "51+",
        ],
        "group": [
            "children",
            "children",
            "children",
            "children",
            "M",
            "M",
            "M",
            "M",
            "M",
            "F",
            "F",
            "F",
            "F",
            "F",
        ],
        "conversion": [
            0.29,
            0.51,
            0.71,
            0.78,
            0.98,
            1.18,
            1.14,
            1.14,
            0.90,
            0.86,
            0.86,
            0.86,
            0.86,
            0.75,
        ],
    }
    conversion_table = pd.DataFrame(data=d)

    bins = [0, 1, 4, 7, 11, 15, 19, 25, 51, 150]

    pan_ind["age_group"] = pd.cut(
        pan_ind["Age"], bins=bins, labels=age_group, right=False
    )
    pan_ind["group"] = np.where(pan_ind["Age"] < 11, "children", pan_ind["Gender"])
    pan_ind_conv = pan_ind.merge(
        conversion_table, how="left", on=["age_group", "group"]
    ).copy()
    hh_conv = (
        pan_ind_conv.groupby(["Panel Id"])["conversion"]
        .sum()
        .reset_index(name="hh_conv")
    )
    # merge back to pan_ind_conv
    pan_ind_conv = pan_ind_conv.merge(hh_conv, how="left", on="Panel Id")
    pan_ind_conv["prop_intake"] = pan_ind_conv["conversion"] / pan_ind_conv["hh_conv"]
    return pan_ind_conv


def adult_intake(demog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the proportion of alcohol intake for each individual over 18.

    Args:
        demog_df (pd.DataFrame): The demographic DataFrame with conversion information.
        child_energy_intake (pd.DataFrame): The child energy intake DataFrame.

    Returns:
        pd.DataFrame: The individual table with the proportion of alcohol intake.
    """
    ind_table_conv = ind_size_conv(demog_df)
    # subset to >= 18
    ind_table_conv_18 = ind_table_conv[ind_table_conv["Age"] >= 18].copy()
    return ind_table_conv_18
