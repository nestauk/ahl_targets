from ahl_targets.getters import get_data
import numpy as np


def type(dat):
    """Returns product type given the combination of rst_4_market and rst_4_market_sector"""
    mapping = get_data.product_type()
    dat["key"] = dat["rst_4_market"] + dat["rst_4_market_sector"]
    dat["type"] = dat["key"].map(mapping)
    return dat


def is_food(dat):
    """Returns indicator whether product is food or drink based on rst_4_market_sector"""
    dat["is_food"] = np.where(
        (dat["rst_4_market_sector"] == "Alcohol")
        | (dat["rst_4_market_sector"] == "Chilled Drinks")
        | (dat["rst_4_market_sector"] == "Take Home Soft Drinks"),
        0,
        1,
    )
    return dat


def in_scope(dat):
    """Returns info on whether product is in scope or not"""
    dat_type = dat.pipe(type).pipe(is_food)
    dat_type["in_scope"] = np.where(
        (dat_type["type"] == "discretionary")
        & (dat_type["is_food"] == 1)
        & (dat_type["npm_score"] >= 4),
        1,
        np.where(
            (dat_type["type"] == "discretionary")
            & (dat_type["is_food"] == 0)
            & (dat_type["npm_score"] >= 1),
            1,
            0,
        ),
    )
    return dat_type
