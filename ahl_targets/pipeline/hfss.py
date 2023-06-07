from ahl_targets.getters import get_data
from ahl_targets.utils import create_tables as tables
import pandas as pd
import numpy as np
import logging


def merge_nut(df, nut_rec):
    """Returns dataframe of purchases merged with nutritional record"""
    return df.merge(
        nut_rec,
        left_on=["PurchaseId", "Period"],
        right_on=["Purchase Number", "Purchase Period"],
    )


def clean_tbl(prod_meta, pur_rec):
    """Returns dataframe of purchases with category information (prod_tbl)"""
    prod_tbl = prod_meta[
        [
            "product_code",
            "rst_4_extended",
            "rst_4_market",
            "rst_4_market_sector",
            "rst_4_sub_market",
            "rst_4_trading_area",
        ]
    ].merge(pur_rec, left_on="product_code", right_on="Product Code")[
        [
            "product_code",
            "rst_4_extended",
            "rst_4_market",
            "rst_4_market_sector",
            "rst_4_sub_market",
            "rst_4_trading_area",
            "PurchaseId",
            "Period",
            "reported_volume_up",
            "volume_up",
            "volume_per",
        ]
    ]

    return prod_tbl[prod_tbl["volume_up"] > 0].copy()


def split_volumes(prod_tbl):
    """Returns two dataframes, split by whether the product's volumes are in kg or litres"""
    grouped = prod_tbl.groupby(prod_tbl.reported_volume_up)

    prod_kg = grouped.get_group("Kilos")
    prod_lt = grouped.get_group("Litres")

    return prod_kg, prod_lt


def prod_kilos(prod_tbl, nut_rec):
    """Returns dataframe with merged nutritional info for kg products (prod_kg_nut)"""
    return merge_nut(split_volumes(prod_tbl)[0], nut_rec)


def kcal_per_100g_foods(df):
    """Returns series with kcal per 100 g"""
    return df["Energy KCal"] / (df["volume_up"] * 10)


def nut_per_100g_foods(df, col):
    """Returns series with macro per 100 g"""
    return 100 * (df[col] / df["volume_up"])


def food_per_100g(prod_kg_nut):
    """Assign new colums with standardised nutritional info and remove implausible values"""
    prod_kg_nut["kcal_per_100g"] = kcal_per_100g_foods(prod_kg_nut)
    prod_kg_nut["sat_per_100g"] = nut_per_100g_foods(prod_kg_nut, "Saturates KG")
    prod_kg_nut["prot_per_100g"] = nut_per_100g_foods(prod_kg_nut, "Protein KG")
    prod_kg_nut["sug_per_100g"] = nut_per_100g_foods(prod_kg_nut, "Sugar KG")
    prod_kg_nut["sod_per_100g"] = nut_per_100g_foods(prod_kg_nut, "Sodium KG")
    prod_kg_nut["fibre_per_100g"] = nut_per_100g_foods(prod_kg_nut, "Fibre KG Flag")

    # remove implausible values
    prod_kg_nut = prod_kg_nut[prod_kg_nut["kcal_per_100g"] < 900].copy()

    return prod_kg_nut


def specific_gravity(prod_lt, gravity):
    """Creates df with specific gravity values per category

    Args:
        prod_lt (pd.DataFrame): table of products with categories
        gravity (pd.DataFrame): lookup of specific gravity values per category

    Returns:
        pd.DataFrame: df with specific gravity values of products
    """
    return prod_lt.merge(gravity, on="rst_4_extended")


def prod_litres(prod_tbl, gravity, nut_rec):
    """Returns dataframe with merged nutritional info for litres products (prod_lt_nut)"""
    return merge_nut(specific_gravity(split_volumes(prod_tbl)[1], gravity), nut_rec)


def kcal_per_100g_drinks(df):
    """Returns series with kcal per 100 g"""
    return df["Energy KCal"] / (10 * df["volume_up"] * df["sg"])


def nut_per_100g_drinks(df, col):
    """Returns series with macro per 100 g"""
    return 100 * (df[col] / (df["volume_up"] * df["sg"]))


def drink_per_100g(prod_lt_nut):
    """Assign new colums with standardised nutritional info and remove implausible values"""
    prod_lt_nut["kcal_per_100g"] = kcal_per_100g_drinks(prod_lt_nut)
    prod_lt_nut["sat_per_100g"] = nut_per_100g_drinks(prod_lt_nut, "Saturates KG")
    prod_lt_nut["prot_per_100g"] = nut_per_100g_drinks(prod_lt_nut, "Protein KG")
    prod_lt_nut["sug_per_100g"] = nut_per_100g_drinks(prod_lt_nut, "Sugar KG")
    prod_lt_nut["sod_per_100g"] = nut_per_100g_drinks(prod_lt_nut, "Sodium KG")
    prod_lt_nut["fibre_per_100g"] = nut_per_100g_drinks(prod_lt_nut, "Fibre KG Flag")

    return prod_lt_nut


def all_prod(prod_lt_nut, prod_kg_nut, fvn):
    """Concatenate kg and litre product dataframes and merge fruit veg and nut points"""
    return pd.concat([prod_lt_nut, prod_kg_nut]).merge(
        fvn, on=["rst_4_trading_area", "rst_4_market_sector", "rst_4_market"]
    )


def assign_energy_score(df, column_name):
    """Calculates NPM energy density score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [
        -1,
        335,
        670,
        1005,
        1340,
        1675,
        2010,
        2345,
        2680,
        3015,
        3350,
        float("inf"),
    ]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name] * 4.184, bins=thresholds, labels=scores, right=True
    )
    return binned_cats


def assign_satf_score(df, column_name):
    """Calculates NPM saturated fats score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float("inf")]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


def assign_sugars_score(df, column_name):
    """Calculates NPM sugars score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45, float("inf")]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


def assign_sodium_score(df, column_name):
    """Calculates NPM sodium score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 90, 180, 270, 360, 450, 540, 630, 720, 810, 900, float("inf")]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


def assign_protein_score(df, column_name):
    """Calculates NPM proteins score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 1.6, 3.2, 4.8, 6.4, 8, float("inf")]
    scores = [0, 1, 2, 3, 4, 5]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


def assign_fiber_score(df, column_name):
    """Use 2004-2005 NPM to assign C points for fiber (AOAC)."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [
        0,
        0.9,
        1.9,
        2.8,
        3.7,
        4.7,
        float("inf"),
    ]  # NSP thresholds [0, 0.7, 1.4, 2.1, 2.8, 3.5]
    scores = [0, 1, 2, 3, 4, 5]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats


def assign_scores(prod_all):
    """Assign columns with scores for each NPM factor"""
    prod_all["energy_score"] = assign_energy_score(prod_all, "kcal_per_100g")
    prod_all["satf_score"] = assign_satf_score(prod_all, "sat_per_100g")
    prod_all["sug_score"] = assign_sugars_score(prod_all, "sug_per_100g")
    prod_all["sodium_score"] = assign_sodium_score(prod_all, "sod_per_100g")
    prod_all["protein_score"] = assign_protein_score(prod_all, "prot_per_100g")
    prod_all["fiber_score"] = assign_fiber_score(prod_all, "fibre_per_100g")

    return prod_all


def a_points_cols():
    """Define columns for calculation of A points"""
    return ["energy_score", "satf_score", "sug_score", "sodium_score"]


def calculate_npm_score(df, a_points, fiber_col, protein_col, fvn_col):
    """Calculate total points using 2004-2005 guidelines."""
    df[a_points] = df[a_points].astype("int")  # convert to int dtype
    df[fiber_col] = df[fiber_col].astype("float")  # convert to float dtype
    df[protein_col] = df[protein_col].astype("float")  # convert to float dtype
    df[fvn_col] = df[fvn_col].astype("float")  # convert to float dtype

    a_points_sum = df[a_points].sum(axis=1)
    fvn_col_values = df[fvn_col].values

    total = np.where(
        (a_points_sum >= 11) & (fvn_col_values >= 5),
        a_points_sum - df[fiber_col] - df[protein_col] - fvn_col_values,
        a_points_sum - df[fiber_col] - fvn_col_values,
    )

    return total.tolist()


def scoring_df(prod_meta, pur_rec, gravity, nut_rec, fvn):
    """Returns df with columns needed for scoring"""
    out = assign_scores(
        all_prod(
            drink_per_100g(
                prod_litres(clean_tbl(prod_meta, pur_rec), gravity, nut_rec)
            ),
            food_per_100g(prod_kilos(clean_tbl(prod_meta, pur_rec), nut_rec)),
            fvn,
        )
    )
    return out[
        [
            "product_code",
            "PurchaseId",
            "Period",
            "energy_score",
            "satf_score",
            "sug_score",
            "sodium_score",
            "protein_score",
            "fiber_score",
            "Score",
        ]
    ]


def npm_score(
    prod_meta, pur_rec, gravity, nut_rec, fvn, a_points, fiber_col, protein_col, fvn_col
):
    """Returns a df of purchases and NPM scores"""
    logging.info("The function npm_score takes about 7 minutes to run")
    tbl = scoring_df(prod_meta, pur_rec, gravity, nut_rec, fvn)
    tbl["npm_score"] = calculate_npm_score(
        tbl, a_points, fiber_col, protein_col, fvn_col
    )
    return tbl


def npm_score_unique(
    prod_meta, pur_rec, gravity, nut_rec, fvn, a_points, fiber_col, protein_col, fvn_col
) -> pd.DataFrame:
    """Returns a df with unique NPM scores (the highest score observed)"""
    tbl = npm_score(
        prod_meta,
        pur_rec,
        gravity,
        nut_rec,
        fvn,
        a_points,
        fiber_col,
        protein_col,
        fvn_col,
    )
    return tbl.groupby(["product_code"])["npm_score"].max().reset_index()
