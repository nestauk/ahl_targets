import re


def taxonomy(store_code, store_line):
    """Returns merged dataset with store types"""
    return store_code.merge(
        store_line, on=["itemisation_id", "itemisation_line_id"]
    ).query("itemisation_id == 1")


def online(dat):
    """Returns Series with indicator for online stores"""
    dat["online"] = dat["itemisation_level_4"].str.contains("Internet")
    return dat
