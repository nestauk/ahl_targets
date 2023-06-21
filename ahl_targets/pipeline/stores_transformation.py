import re
import numpy as np


def taxonomy(store_code, store_line):
    """Returns merged dataset with store types"""
    return store_code.merge(
        store_line, on=["itemisation_id", "itemisation_line_id"]
    ).query("itemisation_id == 1")


def online(dat):
    """Returns Series with indicator for online stores"""
    dat["online"] = dat["itemisation_level_4"].str.contains("Internet")
    return dat


def custom_taxonomy(store_levels):
    """Returns store categorisation as required by AHL
    Args:
        store_levels (pd.DataFrame): combined dataset of store codes and taxonomy
    """

    store_levels["store_cat"] = np.where(
        store_levels["itemisation_level_3"].isin(
            [
                "Total Bargain Stores",
                "Other Non-Grocers",
                "Total Other Independents",
                "Total Hard Discounters",
            ]
        ),
        store_levels["itemisation_level_4"],
        store_levels["itemisation_level_3"],
    )
    return store_levels


def store_subset(store_levels):
    """Returns subsetted store levels (based on AHL requirements)

    Args:
        store_levels (pd.DataFrame): combined dataset of store codes and taxonomy
    """

    keep_stores = [
        "Total Tesco",
        "Total Sainsbury's",
        "Total Asda",
        "Total Morrisons",
        "Aldi",
        "Lidl",
        "Total Waitrose",
        "The Co-Operative",
        "Total Marks & Spencer",
        "Total Iceland",
        "Ocado Internet",
    ]
    return store_levels[store_levels["store_cat"].isin(keep_stores)]
