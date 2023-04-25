from ahl_targets import PROJECT_DIR
from ahl_targets.utils import create_tables as table
from ahl_targets.pipeline import hfss
import pandas as pd
import numpy as np


def prod_energy_100(
    cat: str,
    pur_recs: pd.DataFrame,
    nut_recs: pd.DataFrame,
    prod_meta: pd.DataFrame,
    prod_meas: pd.DataFrame,
    gravity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates a unique list of product with kcal/100ml(g). Excludes products with a different unit of
    measurement to their category (based on the 75% rule).
    To note: Requires the purchases records with the 'reported volume' field added
    Args:
        cat (str): one product category
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data (with reported volume added)
        nut_recs (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        prod_meta (pd.DataFrame): Pandas dataframe with product descriptions
        prod_meas (pd.DataFrame): Pandas dataframe with additional conversions to g and ml for unit and serving products
        gravity (pd.DataFrame): Pandas dataframe with gravity values per product category
    Returns:
        pd.DataFrame: Dataframe with average kcal/100ml(gr) and reported volume
    """

    # add file with additional conversions (for units and servings)
    tbl = table.measure_table(prod_meas)

    # merge to extract additional measures
    pur_rec_vol = pur_recs.merge(tbl, on="Product Code")

    # conditional expression to select volume
    conds = [
        (pur_rec_vol["Reported Volume"] == "Litres")
        | (pur_rec_vol["Reported Volume"] == "Kilos"),
        (
            (pur_rec_vol["Reported Volume"] == "Servings")
            | (pur_rec_vol["Reported Volume"] == "Units")
        )
        & ~np.isnan(pur_rec_vol["Grams"]),
        (
            (pur_rec_vol["Reported Volume"] == "Servings")
            | (pur_rec_vol["Reported Volume"] == "Units")
        )
        & np.isnan(pur_rec_vol["Grams"])
        & ~np.isnan(pur_rec_vol["Litres"]),
    ]

    choices = [
        pur_rec_vol["Reported Volume"],
        "Kilos",
        "Litres",
    ]

    choice_volume = [
        pur_rec_vol["Volume"],
        pur_rec_vol["Quantity"] * pur_rec_vol["Grams"] / 1000,
        pur_rec_vol["Quantity"] * pur_rec_vol["Litres"],
    ]

    # Updated volume label
    pur_rec_vol["reported_volume_up"] = np.select(conds, choices, "missing")

    # Updated volume
    pur_rec_vol["volume_up"] = np.select(conds, choice_volume, pur_rec_vol["Volume"])

    # scaled gross up weight - this converts the wegith from quantities to volumes (either kg or l)

    pur_rec_vol["scaled_factor"] = (
        pur_rec_vol["Gross Up Weight"] * pur_rec_vol["volume_up"]
    )

    # create unique list of products with total sales
    pur_recs_agg = (
        pur_rec_vol.groupby(["Product Code", "reported_volume_up"])["scaled_factor"]
        .sum()
        .reset_index(name="total_sale")
    )

    # # merge with product metadata
    pur_recs_meta = pur_recs_agg.merge(
        prod_meta, left_on="Product Code", right_on="product_code", how="left"
    )

    # check distribution of reported volume within category
    level = (
        pur_recs_meta.groupby([cat, "reported_volume_up"])
        .size()
        .copy()
        .reset_index(name="count")
    )
    level_pivot = (
        pd.pivot(level, index=[cat], columns="reported_volume_up", values="count")
        .fillna(0)
        .reset_index()
    )

    # Add missing column if no missing values for volume
    if "missing" not in level_pivot.columns:
        level_pivot["missing"] = 0

    # determine which measurement is used to generate average
    level_pivot["tot"] = (
        level_pivot["Kilos"] + level_pivot["Litres"] + level_pivot["missing"]
    )
    level_pivot["kilo_share"] = level_pivot["Kilos"] / level_pivot["tot"]
    level_pivot["litre_share"] = level_pivot["Litres"] / level_pivot["tot"]
    level_pivot["chosen_unit"] = np.where(
        level_pivot["litre_share"] >= 0.75,
        "Litres",
        np.where(level_pivot["kilo_share"] >= 0.75, "Kilos", "none"),
    )

    # merge with product metadata
    pur_rec_conv = pur_recs_agg.merge(
        prod_meta, left_on="Product Code", right_on="product_code", how="left"
    ).merge(level_pivot[[cat, "chosen_unit"]], on=cat)

    # subset to products where the reported volume is equal to the chosen unit based on 75% rule
    pur_rec_select = pur_rec_conv[
        pur_rec_conv["reported_volume_up"] == pur_rec_conv["chosen_unit"]
    ]

    # generate nutritional info to merge into the aggregate data
    # Convert to datetime format
    pur_rec_vol["Purchase Date"] = pd.to_datetime(
        pur_rec_vol["Purchase Date"], format="%d/%m/%Y"
    )

    # Get unique and most recent products
    pur_recs_latest = (
        pur_rec_vol.sort_values(by=["Purchase Date"], ascending=False)
        .drop_duplicates(subset="Product Code", keep="first")
        .merge(
            nut_recs[["Purchase Number", "Purchase Period", "Energy KCal"]],
            how="left",
            left_on=["PurchaseId", "Period"],
            right_on=["Purchase Number", "Purchase Period"],
        )
        .drop(["Purchase Number", "Purchase Period"], axis=1)
    )

    # Handle litres and kilos seperately
    pur_recs_latest_litres = (
        pur_recs_latest[pur_recs_latest.reported_volume_up == "Litres"]
        .copy()
        .merge(
            prod_meta[["product_code", "rst_4_extended"]].copy(),
            how="left",
            left_on="Product Code",
            right_on="product_code",
        )
        .drop("product_code", axis=1)
    )
    pur_recs_latest_litres = hfss.specific_gravity(
        pur_recs_latest_litres.copy(), gravity
    )
    pur_recs_latest_kilos = pur_recs_latest[
        pur_recs_latest.reported_volume_up == "Kilos"
    ].copy()

    # generate value of kcal per 100g
    pur_recs_latest_kilos["kcal_100g"] = pur_recs_latest_kilos["Energy KCal"] / (
        pur_recs_latest_kilos["volume_up"] * 10
    )
    pur_recs_latest_kilos = pur_recs_latest_kilos[
        pur_recs_latest_kilos["kcal_100g"] <= 900
    ].copy()
    pur_recs_latest_litres["kcal_100g"] = hfss.kcal_per_100g_drinks(
        pur_recs_latest_litres
    )
    # Join dfs back together
    pur_recs_latest_grams = pd.concat(
        [pur_recs_latest_kilos, pur_recs_latest_litres], axis=0
    )
    # unique dataframe of product with kcal info
    pur_recs_latest_grams.drop_duplicates(subset="Product Code", inplace=True)

    # merge kcal info with sales
    return pur_rec_select.merge(
        pur_recs_latest_grams[["Product Code", "kcal_100g"]].copy(), on="Product Code"
    )


def cat_energy_100(
    cat: str,
    pur_final: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return simple and weighted kcal/100ml(g) aggregate by product category
    Args:
        cat (str): one product category
        pur_final (pd.DataFrame): Pandas dataframe products with energy density, sales and category information
    Returns:
        pd.DataFrame: Dataframe with average kcal/100ml(gr) simple and weighted by sales (for the year) and reported volume
    """

    # simple mean
    s_mean = (
        pur_final.groupby([cat, "chosen_unit"])["kcal_100g"]
        .mean()
        .reset_index(name="kcal_100_s")
    )

    # weighted mean
    pur_final["cross_prd"] = (
        pur_final["kcal_100g"] * pur_final["total_sale"]
    )  # cross product
    kcal = (
        pur_final.groupby([cat])["cross_prd"].sum().reset_index(name="sum_cross_prd")
    )  # sum of cross product by cat
    sale = (
        pur_final.groupby([cat])["total_sale"].sum().reset_index(name="sum_total_sale")
    )  # sum of sales by cat
    w_mean = kcal.merge(sale, on=cat)  # merge sum of cross and total
    w_mean["kcal_100_w"] = (
        w_mean["sum_cross_prd"] / w_mean["sum_total_sale"]
    )  # weighted mean

    # generate final output
    return s_mean.merge(w_mean[[cat, "kcal_100_w"]], on=cat)


def score(df_col: pd.Series) -> pd.DataFrame:
    """
    Generate energy density category variable based on standard thresholds.
    Args:
        df_col (pd.Series): Energy density scores
    Returns:
        pd.DataFrame: Dataframe with average kcal/100ml(gr) simple and weighted by sales (for the year) and reported volume
    """
    # generate energy density category variable based on standard thresholds
    return pd.cut(
        df_col,
        bins=[0, 60, 150, 400, float("Inf")],
        labels=["very_low", "low", "medium", "high"],
    )


def decile(df_col: pd.Series) -> pd.Series:
    """Generate decile of energy density

    Args:
        df_col (pd.Series): Enerfy density scores

    Returns:
        pd.Series: Series with deciles of ED
    """
    return pd.qcut(df_col, q=10, labels=False)
