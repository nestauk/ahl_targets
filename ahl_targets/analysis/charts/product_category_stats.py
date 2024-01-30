# %%
#
from unicodedata import category
from ahl_targets.pipeline import model_data
from ahl_targets.pipeline import stores_transformation as stores
from ahl_targets.getters import get_data
from functools import reduce
import pandas as pd
from ahl_targets import PROJECT_DIR
import logging

# %%
prod_table = get_data.product_metadata()
# pur_rec_vol = get_data.purchase_records_volume()
# nut_recs = get_data.nutrition()
# %%
store_coding = get_data.store_itemisation_coding()
store_lines = get_data.store_itemisation_lines()

# %%
# gravity = get_data.get_gravity()
# fvn = get_data.get_fvn()
# store_coding = get_data.store_itemisation_coding()
# store_lines = get_data.store_itemisation_lines()

# %%


def spend_share(
    prod_purch_df: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    return (
        (prod_purch_df["Gross Up Weight"] * prod_purch_df["Spend"])
        .groupby(prod_purch_df[category])
        .sum()
        / (prod_purch_df["Gross Up Weight"] * prod_purch_df["Spend"]).sum()
    ).reset_index(name="spend_share")


def volume_share(
    cat_data: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    return (
        (cat_data["Gross Up Weight"] * cat_data["volume_up"])
        .groupby(cat_data[category])
        .sum()
        / (cat_data["Gross Up Weight"] * cat_data["volume_up"]).sum()
    ).reset_index(name="volume_share")


def product_share(
    cat_data: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    return (
        (cat_data["Gross Up Weight"]).groupby(cat_data[category]).sum()
        / (cat_data["Gross Up Weight"]).sum()
    ).reset_index(name="product_share")


def kcal_share(
    cat_data: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    return (
        (cat_data["Gross Up Weight"] * cat_data["Energy KCal"])
        .groupby(cat_data[category])
        .sum()
        / (cat_data["Gross Up Weight"] * cat_data["Energy KCal"]).sum()
    ).reset_index(name="kcal_share")


def ed_average(
    cat_data: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    return (
        (cat_data["Gross Up Weight"] * cat_data["ed"]).groupby(cat_data[category]).sum()
        / (cat_data["Gross Up Weight"]).groupby(cat_data[category]).sum()
    ).reset_index(name="ed_average")


def npm_average(
    cat_data: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    return (
        (cat_data["Gross Up Weight"] * cat_data["npm_score"])
        .groupby(cat_data[category])
        .sum()
        / (cat_data["Gross Up Weight"]).groupby(cat_data[category]).sum()
    ).reset_index(name="npm_average")


def hfss_average(
    cat_data: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    return (
        (cat_data["Gross Up Weight"] * cat_data["in_scope"])
        .groupby(cat_data[category])
        .sum()
        / (cat_data["Gross Up Weight"]).groupby(cat_data[category]).sum()
    ).reset_index(name="hfss_average")


def spend_share_subset(
    prod_purch_df: pd.DataFrame,
    category: str,
    threshold: float,
) -> pd.DataFrame:
    """
    Subset purchase/product info by categories with a greater than threshold % spend share.

    Args:
        prod_purch_df (pd.DataFrame): DataFrame with purchase/product information.
        category (str): Name of the column containing the categories.
        threshold (float): Minimum threshold for the spend share.

    Returns:
        pd.DataFrame: Subset of prod_purch_df containing only the categories with
                      a spend share greater than threshold.
    """
    # Calculate the spend share for each category
    spend_share_df = spend_share(prod_purch_df, category)
    # Get a list of the categories with spend share greater than the threshold
    spend_list = list(spend_share_df[spend_share_df.spend_share > threshold][category])
    # Return a subset of the original DataFrame containing only the selected categories
    return prod_purch_df[prod_purch_df[category].isin(spend_list)].copy()


def product_category_report(
    cat_data: pd.DataFrame,
    file_name: str,
    category: str,
) -> pd.DataFrame:
    """Creates table of metrics based on products per category group and saves as csv file.

    Args:
        cat_data (pd.DataFrame): combined dataset of purchase and category info
        file_name (str): Name to save file
        category (str): Name of category to groupby
    """
    datasets = [
        spend_share(cat_data, category),
        volume_share(cat_data, category),
        product_share(cat_data, category),
        kcal_share(cat_data, category),
        ed_average(cat_data, category),
        npm_average(cat_data, category),
        hfss_average(cat_data, category),
    ]

    prod_info = reduce(
        lambda left, right: pd.merge(left, right, on=category, how="inner"), datasets
    )

    file_path = f"outputs/reports/{file_name}.csv"
    prod_info.to_csv(PROJECT_DIR / file_path, index=False)

    logging.info(f"Saved {file_name}.csv in outputs/reports")
    return prod_info


# %%

# if __name__ == "__main__":


# %%

# Create dataset with complete purchase level info
prod_purch_df = get_data.model_data_full().compute()

# %%


store_levels = stores.custom_taxonomy(
    stores.taxonomy(
        store_coding,
        store_lines,
    )
)

# %%

prod_purch_df = prod_purch_df.merge(
    store_levels, left_on="Store Code", right_on="store_id"
)
# %%

# Subset dataset based on stores + manufacturers with >1.5% spend share
store_sub_prods = stores.store_subset(
    prod_purch_df,
)

# %%

manuf_sub_prods = spend_share_subset(prod_purch_df, "manufacturer", 0.005)
manuf_store_sub_prods = prod_purch_df.pipe(
    spend_share_subset,
    category="manufacturer",
    threshold=0.005,
).pipe(
    stores.store_subset,
)
# %%

logging.info("Saving files")
# store subset products - store
store_sub_df = product_category_report(
    store_sub_prods,
    "store_subset_store_info",
    "store_cat",
)
# %%

# store subset products - manufacturer
store_sub_manuf_df = product_category_report(
    manuf_store_sub_prods,
    "store_subset_manuf_info",
    "manufacturer",
)
# %%

# store
store_df = product_category_report(
    prod_purch_df,
    "store_info",
    "store_cat",
)
# manufacturer
manuf_df = product_category_report(manuf_sub_prods, "manuf_info", "manufacturer")
# %%
