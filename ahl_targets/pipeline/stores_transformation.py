import re
import numpy as np


def aldi(store_levels):
    """Splits out Aldi from level 4 to 3"""
    aldi_stores = np.where(
        store_levels["itemisation_level_4"] == "Aldi",
        "Aldi",
        store_levels["itemisation_level_3"],
    )
    store_levels["itemisation_level_3"] = aldi_stores
    return store_levels


def lidl(store_levels):
    """Splits out Lidl from level 4 to 3"""
    lidl_stores = np.where(
        store_levels["itemisation_level_4"] == "Lidl",
        "Lidl",
        store_levels["itemisation_level_3"],
    )
    store_levels["itemisation_level_3"] = lidl_stores
    return store_levels


def taxonomy(store_code, store_line):
    """Returns merged dataset with store types"""
    return (
        store_code.merge(store_line, on=["itemisation_id", "itemisation_line_id"])
        .query("itemisation_id == 1")
        .pipe(aldi)
        .pipe(lidl)
    )


def online(dat):
    """Returns Series with indicator for online stores"""
    dat["online"] = dat["itemisation_level_4"].str.contains("Internet")
    return dat
