from ahl_targets.pipeline import hfss
from ahl_targets.pipeline import stores_transformation as stores
from ahl_targets.pipeline import product_transformation as product
import logging


def make_data(
    prod_table,
    pur_rec_vol,
    gravity,
    nut_recs,
    fvn,
    store_coding,
    store_lines,
):
    """Merge datasets and restrict to selected retailers

    Args:
        prod_table (pd.DataFrame): product metafata
        pur_rec_vol (pd.DataFrame): purchase record
        gravity (pd.DataFrame): gravity info for liquids
        nut_recs (pd.DataFrame): nutritional record
        fvn (pd.DataFrame): fruit veg and nuts content
        store_coding (pd.DataFrame): store codes
        store_lines (pd.DataFrame): store taxonomy

    Returns:
        pd.DataFrame: merged data
    """
    logging.info("The function make_data takes about 10 minutes to run")

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

    store_levels = stores.custom_taxonomy(
        stores.taxonomy(
            store_coding,
            store_lines,
        )
    )

    stores_sub = store_levels[store_levels["store_cat"].isin(keep_stores)]

    logging.info("merge purchase record with product infro")
    dat1 = hfss.clean_tbl(prod_table, pur_rec_vol)

    logging.info("filter to kg only")
    # filter to kg only
    dat2 = hfss.prod_kilos(dat1, nut_recs)

    # generate ED
    dat2["ed"] = hfss.kcal_per_100g_foods(dat2)

    # remove implausible values
    prod_kg_nut = dat2[dat2["ed"] < 900].copy()

    # merge with purchase record
    dat3 = prod_kg_nut.merge(
        pur_rec_vol[
            [
                "PurchaseId",
                "Period",
                "Gross Up Weight",
                "Quantity",
                "Store Code",
                "Spend",
            ]
        ],
        on=["PurchaseId", "Period"],
    )

    logging.info("Run NPM function")

    npm = hfss.npm_score(
        prod_table,
        pur_rec_vol,
        gravity,
        nut_recs,
        fvn,
        hfss.a_points_cols(),
        "fiber_score",
        "protein_score",
        "Score",
    )

    # Merge with npm and store names
    store_data = dat3.merge(
        stores_sub, left_on="Store Code", right_on="store_id"
    ).merge(npm[["PurchaseId", "Period", "npm_score"]], on=["PurchaseId", "Period"])

    store_data = store_data.pipe(product.in_scope)

    return store_data


def weighted_kg(store_data):
    """aggregated df with volume weights

    Args:
        store_data (pd.DataFrame): merged data
    """
    out = (
        (store_data["volume_up"] * store_data["Gross Up Weight"])
        .groupby(store_data["product_code"])
        .sum()
        .reset_index(name="total_kg")
    )
    out["kg_w"] = out["total_kg"] / out["total_kg"].sum()

    return out


def weighted_kcal(store_data):
    """aggregated df with kcal weights

    Args:
        store_data (pd.DataFrame): merged data
    """
    out = (
        (store_data["Energy KCal"] * store_data["Gross Up Weight"])
        .groupby(store_data["product_code"])
        .sum()
        .reset_index(name="total_kcal")
    )
    out["kcal_w"] = out["total_kcal"] / out["total_kcal"].sum()

    return out


def weighted_prod(store_data):
    """aggregated df with product weights

    Args:
        store_data (pd.DataFrame): merged data
    """
    out = (
        store_data["Gross Up Weight"]
        .groupby(store_data["product_code"])
        .sum()
        .reset_index(name="total_prod")
    )
    out["prod_w"] = out["total_prod"] / out["total_prod"].sum()

    return out


def unique_ed(store_data):
    """Unique ED by prodyuct

    Args:
        store_data (pd.DataFrame): marged data

    Returns:
        pd.DataFrame: aggregate df by product code with unique ED
    """
    return store_data.groupby(["product_code"])["ed"].mean().reset_index(name="ed")


def unique_kcal(store_data):
    """Unique KCAL by prodyuct

    Args:
        store_data (pd.DataFrame): marged data

    Returns:
        pd.DataFrame: aggregate df by product code with unique KCAL
    """
    return (
        (store_data["Energy KCal"] / store_data["Quantity"])
        .groupby(store_data["product_code"])
        .mean()
        .reset_index(name="kcal_unit")
    )


def weighted(store_data):
    """merged aggregated weight data

    Args:
        store_data (pd.DataFrame): merged data

    Returns:
        pd.DataFrame: merged aggregated weight data
    """
    return (
        weighted_kg(store_data)
        .merge(weighted_kcal(store_data), on="product_code")
        .merge(weighted_prod(store_data), on="product_code")
        .merge(unique_ed(store_data), on="product_code")
        .merge(unique_kcal(store_data), on="product_code")
    )


def prod_weight_g(w_store_data):
    """average weight of product in grames

    Args:
        w_store_data (pd.DataFrame): merged data

    Returns:
        pd.Series: average weight of product
    """
    return w_store_data["total_kg"] / w_store_data["total_prod"] * 1000
