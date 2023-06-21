from ahl_targets.getters import get_data
import numpy as np


def type(dat):
    """Returns product type given the combination of rst_4_market and rst_4_market_sector"""
    mapping = get_data.product_type()
    dat["key"] = dat["rst_4_market"] + dat["rst_4_market_sector"]
    dat["type"] = dat["key"].map(mapping)
    return dat


# def is_food(dat):
#     """Returns indicator whether product is food or drink based on rst_4_market_sector"""
#     dat["is_food"] = np.where(
#         (dat["rst_4_market_sector"] == "Alcohol")
#         | (dat["rst_4_market_sector"] == "Chilled Drinks")
#         | (dat["rst_4_market_sector"] == "Take Home Soft Drinks"),
#         0,
#         1,
#     )
#     return dat

import numpy as np


def is_food(dat):
    """
    Determines if a product is food or drink based on its market and market sector.

    Args:
        dat (pandas.DataFrame): Dataframe containing the market and market sector columns.

    Returns:
        pandas.DataFrame: A copy of the input dataframe with an additional column indicating if the product is food or drink.
    """

    # Extract market and market sector columns from the input dataframe
    col1 = dat["rst_4_market"]
    col2 = dat["rst_4_market_sector"]

    # Define the market and market sector categories for food products
    market = [
        "Chilled Flavoured Milk",
        "Food Drinks",
        "Herbal Tea",
        "Instant Coffee",
        "Instant Milk",
        "Liquid+Grnd Coffee+Beans",
        "Tea",
    ]

    market_sector = [
        "Alcohol",
        "Chilled Drinks",
        "Take Home Soft Drinks",
    ]

    # Determine if a product is food or drink based on its market and market sector
    conditions = col1.isin(market) | col2.isin(market_sector)
    dat["is_food"] = np.where(conditions, 0, 1)

    # Return a copy of the input dataframe with an additional column indicating if the product is food or drink
    return dat.copy()


def in_scope(dat_type):
    """Returns info on whether product is in scope or not"""
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
