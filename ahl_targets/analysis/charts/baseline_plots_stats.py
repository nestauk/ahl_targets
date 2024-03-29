# %%
from ahl_targets.utils.plotting import configure_plots
from ahl_targets import PROJECT_DIR
from ahl_targets.getters import get_data
from ahl_targets.pipeline import stores_transformation as stores
from ahl_targets.pipeline import model_data
from ahl_targets.pipeline import hfss
from ahl_targets.pipeline import product_transformation as product

# from ahl_targets.analysis import product_category_stats as cat_stats
# %%

from ahl_targets.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
import altair as alt
import pandas as pd
import logging
from functools import reduce

# %%


def ed_average_weighted(
    cat_data: pd.DataFrame,
    category: str,
    weight: str,
) -> pd.DataFrame:
    return (
        (cat_data[weight] * cat_data["ed"]).groupby(cat_data[category]).sum()
        / (cat_data[weight]).groupby(cat_data[category]).sum()
    ).reset_index(name="ed_average" + weight)


def npm_average_weighted(
    cat_data: pd.DataFrame,
    category: str,
    weight: str,
) -> pd.DataFrame:
    return (
        (cat_data[weight] * cat_data["npm_score"]).groupby(cat_data[category]).sum()
        / (cat_data[weight]).groupby(cat_data[category]).sum()
    ).reset_index(name="npm_average" + weight)


def hfss_average_weighted(
    cat_data: pd.DataFrame,
    category: str,
    weight: str,
) -> pd.DataFrame:
    return (
        (cat_data[weight] * cat_data["in_scope"]).groupby(cat_data[category]).sum()
        / (cat_data[weight]).groupby(cat_data[category]).sum()
    ).reset_index(name="hfss_average" + weight)


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


def baseline_category_report(
    cat_data: pd.DataFrame,
    file_name: str,
    category: str,
    weight_kcal: str,
    weight_vol: str,
) -> pd.DataFrame:
    """Creates table of metrics based on products per category group and saves as csv file.

    Args:
        cat_data (pd.DataFrame): combined dataset of purchase and category info
        file_name (str): Name to save file
        category (str): Name of category to groupby
    """
    datasets = [
        cat_data.groupby([category])["ed"]
        .min()
        .reset_index()
        .rename(columns={"ed": "ed_min"}),
        cat_data.groupby([category])["ed"]
        .max()
        .reset_index()
        .rename(columns={"ed": "ed_max"}),
        ed_average(cat_data, category),
        npm_average(cat_data, category),
        hfss_average(cat_data, category),
        npm_average_weighted(cat_data, category, weight_kcal),
        hfss_average_weighted(cat_data, category, weight_kcal),
        ed_average_weighted(cat_data, category, weight_kcal),
        npm_average_weighted(cat_data, category, weight_vol),
        hfss_average_weighted(cat_data, category, weight_vol),
        ed_average_weighted(cat_data, category, weight_vol),
    ]

    baseline_info = reduce(
        lambda left, right: pd.merge(left, right, on=category, how="inner"), datasets
    )

    file_path = f"outputs/reports/{file_name}.csv"
    baseline_info.to_csv(PROJECT_DIR / file_path, index=False)

    logging.info(f"Saved {file_name}.csv in outputs/reports")
    return baseline_info


def plot_shares_bar(
    metric_df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    x_title: str,
    y_title: str,
):
    """Plot bar chart of shares of selected metric by category."""
    chart = (
        alt.Chart(metric_df[metric_df.spend_share > 0.015].copy())
        .mark_bar()
        .encode(
            x=alt.X(value_col, title=x_title),
            y=alt.Y(category_col, sort="-x", title=y_title),
            color=alt.ColorValue("#0000FF"),
        )
    )
    return configure_plots(
        chart,
        title,
        "",
        16,
        14,
        14,
    )


def plot_shares_bar_hfss(
    metric_df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    x_title: str,
    y_title: str,
):
    """Plot bar chart of shares of selected metric by category."""
    chart = (
        alt.Chart(metric_df)
        .mark_bar()
        .encode(
            x=alt.X(value_col, title=x_title),
            y=alt.Y(category_col, sort="-x", title=y_title),
            color=category_col,
        )
    )
    return configure_plots(
        chart,
        title,
        "",
        16,
        14,
        14,
    )


def plot_npm_component(
    prod_100_npm: pd.DataFrame,
):
    """Plot line chart of average NPM per component."""
    # Line chart
    line_chart = (
        alt.Chart(prod_100_npm)
        .mark_line()
        .encode(x="npm_score:N", y="per 100g", color="component")
        .properties(width=600, height=350)
    )
    return configure_plots(
        line_chart,
        "Average NPM per component",
        "",
        16,
        14,
        14,
    )


def plot_npm_component_facet(
    prod_100_npm: pd.DataFrame,
):
    """Plot line chart of average NPM per component as a facet plot."""
    # Line chart
    line_chart = (
        alt.Chart(prod_100_npm)
        .mark_line()
        .encode(
            x="npm_score:N",
            y="per 100g",
            color="component",
            tooltip=["npm_score", "per 100g", "component"],
        )
        .properties(width=600, height=350)
        .facet(
            facet="component",
            columns=3,
        )
    ).resolve_scale(y="independent")
    return configure_plots(
        line_chart,
        "Average NPM per component",
        "",
        16,
        14,
        14,
    )


# %%

# if __name__ == "__main__":


# %%

import logging

# Get data
logging.info("Getting data")
prod_table = get_data.product_metadata()
pur_rec_vol = get_data.purchase_records_volume()
# nut_recs = get_data.nutrition()

# %%
store_coding = get_data.store_itemisation_coding()
store_lines = get_data.store_itemisation_lines()

# %%
# gravity = get_data.get_gravity()
# fvn = get_data.get_fvn()

# %%

# Get repor† files
store_df = pd.read_csv(PROJECT_DIR / f"outputs/reports/store_info.csv")
manuf_df = pd.read_csv(PROJECT_DIR / f"outputs/reports/manuf_info.csv")

# %%

store_levels = stores.custom_taxonomy(
    stores.taxonomy(
        store_coding,
        store_lines,
    )
)
# %%

# Get product / purchase info
logging.info("Getting product / purchase info")
prod_purch_df = get_data.model_data_full().compute()

# %%

prod_purch_df = prod_purch_df.merge(
    store_levels, left_on="Store Code", right_on="store_id"
)

# %%

# Per store / manufacturer:
base_stats = prod_purch_df[
    [
        "Quantity",
        "product_code",
        "ed",
        "Gross Up Weight",
        "Energy KCal",
        "volume_up",
        "manufacturer",
        "store_cat",
        "npm_score",
        "in_scope",
    ]
].copy()

# %%

base_stats["weight_kcal"] = base_stats["Gross Up Weight"] * base_stats["Energy KCal"]
base_stats["weight_vol"] = base_stats["Gross Up Weight"] * base_stats["volume_up"]
base_stats["weight_prod"] = base_stats["Gross Up Weight"]
base_stats["weight_none"] = 1

# HFSS volume weighted shares
hfss_shares_volume = (
    base_stats.groupby(["in_scope"])["weight_vol"].sum()
    / base_stats["weight_vol"].sum()
)
# HFSS product weighted shares
hfss_shares_prod = (
    base_stats.groupby(["in_scope"])["weight_prod"].sum()
    / base_stats["weight_prod"].sum()
)

hfss_shares_none = (
    base_stats.groupby(["in_scope"])["weight_none"].sum()
    / base_stats["weight_none"].sum()
)

# Store subset
store_subset_shares = stores.store_subset(base_stats)

hfss_shares_volume = (
    store_subset_shares.groupby(["in_scope"])["weight_vol"].sum()
    / store_subset_shares["weight_vol"].sum()
)
# HFSS product weighted shares
hfss_shares_prod = (
    store_subset_shares.groupby(["in_scope"])["weight_prod"].sum()
    / store_subset_shares["weight_prod"].sum()
)

# Create new column high NPM >= 4 (1 else 0)
base_stats["high_npm"] = base_stats["npm_score"].apply(lambda x: 1 if x >= 4 else 0)
# High NPM volume weighted shares
high_npm_shares_volume = (
    base_stats.groupby(["high_npm"])["weight_vol"].sum()
    / base_stats["weight_vol"].sum()
)
# High NPM product weighted shares
high_npm_shares_prod = (
    base_stats.groupby(["high_npm"])["weight_prod"].sum()
    / base_stats["weight_prod"].sum()
)
# High NPM kcal weighted shares
high_npm_shares_kcal = (
    base_stats.groupby(["high_npm"])["weight_kcal"].sum()
    / base_stats["weight_kcal"].sum()
)

# Save as csv (for chart D)
hfss_shares_volume.reset_index().to_csv(
    PROJECT_DIR / f"outputs/data/hfss_shares_volume.csv", index=False
)

store_baseline = baseline_category_report(
    base_stats,
    "store_baseline",
    "store_cat",
    "weight_kcal",
    "weight_vol",
)
manuf_baseline = baseline_category_report(
    base_stats,
    "manuf_baseline",
    "manufacturer",
    "weight_kcal",
    "weight_vol",
)

# Subset by chosen stores
hff_shares_vol_weighted_subset = stores.store_subset(store_baseline)[
    ["store_cat", "hfss_averageweight_vol"]
].copy()
hff_shares_vol_weighted_subset.columns = ["stores", "hfss shares (volume weighted)"]
# Manually add a row for total (see above df for calculation)
hff_shares_vol_weighted_subset = pd.concat(
    [
        hff_shares_vol_weighted_subset,
        pd.DataFrame(
            {
                "stores": ["Overall average"],
                "hfss shares (volume weighted)": [hfss_shares_volume[1]],
            }
        ),
    ],
    ignore_index=True,
)
# Save as csv (for chart E3)
hff_shares_vol_weighted_subset.to_csv(
    PROJECT_DIR / f"outputs/data/hfss_shares_volume_weighted_subset.csv", index=False
)
# Plot as bar chart
webdr = google_chrome_driver_setup()
save_altair(
    plot_shares_bar_hfss(
        hff_shares_vol_weighted_subset,
        "stores",
        "hfss shares (volume weighted)",
        "Share of volume HFSS by store",
        "Share of volume",
        "Store",
    ),
    "hfss_shares_volume_weighted_subset",
    driver=webdr,
)


# 100-200 kcal/100g: an avocado (ED of 150 kcal/100g, NPM is X, not HFSS)
# Kcal per 100g, NPM score, HFSS (Y/N)
prod_weighted_data = model_data.weighted(
    prod_purch_df,
)

# %%

pur_rec_vol["Quantity_w"] = pur_rec_vol["Gross Up Weight"] * pur_rec_vol["Quantity"]
# %%

npm_prods = prod_purch_df.groupby(["product_code"])["npm_score"].max().reset_index()

# %%

prod_data = (
    prod_weighted_data[["product_code", "ed"]]
    .copy()
    .merge(
        prod_table[
            [
                "product_code",
                "rst_4_market_sector",
                "rst_4_market",
                "rst_4_extended",
            ]
        ].copy(),
        how="inner",
        left_on="product_code",
        right_on="product_code",
    )
    .merge(npm_prods, left_on="product_code", right_on="product_code")
    .merge(
        pur_rec_vol.groupby(["Product Code"])[["Quantity_w"]].sum().reset_index(),
        left_on="product_code",
        right_on="Product Code",
    )
)

# %%

prod_data = prod_data.pipe(product.is_food).pipe(product.type).pipe(product.in_scope)

# Creates sub groups
ind_prod_examples = [
    900182,
    900065,
    900147,
    27890,
    197326,
    950789,
    950553,
    998397,
    955708,
    902341,
    210096,
    88375,
    522637,
    204397,
    406225,
    175220,
    884930,
    232627,
    171891,
    541024,
    64828,
    101817,
    358788,
]

# %%

prod_data[(prod_data.ed >= 0) & (prod_data.ed < 100)].sort_values(
    by="Quantity_w", ascending=False
).head(20)[["product_code", "rst_4_market_sector", "rst_4_extended"]]
prod_data_sub_ed = prod_data[prod_data["product_code"].isin(ind_prod_examples)][
    [
        "product_code",
        "ed",
        "rst_4_market_sector",
        "rst_4_extended",
        "npm_score",
        "in_scope",
    ]
]
# Save sub groups
prod_data_sub_ed.to_csv(
    PROJECT_DIR / f"outputs/data/example_products_ed.csv", index=False
)
# %%

# Check products after liquid exclusion
(
    prod_purch_df.groupby(["rst_4_market"])["rst_4_extended"].unique().reset_index()
).to_csv(PROJECT_DIR / f"outputs/data/unique_categories_purchase_data.csv", index=False)

# %%

prod_purch_df_sub = stores.store_subset(prod_purch_df).copy()

# Products grouped by NPM score to get avg: sugar, salt...ect per 100g
prod_per_100 = (
    hfss.food_per_100g(prod_purch_df_sub)
    .groupby(["product_code"])[
        [
            "kcal_per_100g",
            "sat_per_100g",
            "prot_per_100g",
            "sug_per_100g",
            "sod_per_100g",
            "fibre_per_100g",
        ]
    ]
    .mean()
    .reset_index()
)

# %%

prod_100_npm = prod_per_100.merge(
    prod_data[["product_code", "npm_score"]],
    left_on="product_code",
    right_on="product_code",
).drop(["product_code"], axis=1)

# %%

prod_100_npm = (
    prod_100_npm.groupby(["npm_score"])
    .mean()
    .reset_index()
    .melt(
        id_vars=["npm_score"],
        var_name="component",
        value_name="per 100g",
    )
)
# Saving CSV file (for chartB)
prod_100_npm.to_csv(PROJECT_DIR / f"outputs/data/npm_components_mean.csv", index=False)

# %%

# Using high ED definition as 400kcal/100g and above:
# What % of high ED products are HFSS?
# What is the average NPM for high ED products?
high_ed_prods = prod_data[prod_data.ed >= 400].copy()
# Save as csv
print(
    "Percent of high ED products as HFSS: "
    + str(high_ed_prods["in_scope"].value_counts(normalize=True)[1])
)
print("Average NPM for high ED products: " + str(high_ed_prods["npm_score"].mean()))

# %%

# Plotting
webdr = google_chrome_driver_setup()

save_altair(
    plot_npm_component(prod_100_npm),
    "npm_components_mean",
    driver=webdr,
)

save_altair(
    plot_npm_component_facet(prod_100_npm),
    "npm_components_mean_facet",
    driver=webdr,
)

save_altair(
    plot_shares_bar(
        store_df,
        "store_cat",
        "kcal_share",
        "Share of kcal stores",
        "Share of kcal",
        "Store",
    ),
    "store_kcal_bar",
    driver=webdr,
)
save_altair(
    plot_shares_bar(
        manuf_df,
        "manufacturer",
        "kcal_share",
        "Share of kcal manufacturer",
        "Share of kcal",
        "manufacturer",
    ),
    "manuf_kcal_bar",
    driver=webdr,
)
save_altair(
    plot_shares_bar(
        store_df,
        "store_cat",
        "volume_share",
        "Share of volume stores",
        "Share of volume",
        "Store",
    ),
    "store_volume_bar",
    driver=webdr,
)
save_altair(
    plot_shares_bar(
        manuf_df,
        "manufacturer",
        "volume_share",
        "Share of volume manufacturer",
        "Share of volume",
        "manufacturer",
    ),
    "manuf_volume_bar",
    driver=webdr,
)
# %%
