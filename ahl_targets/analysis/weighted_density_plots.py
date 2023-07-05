# Description: This script creates plots of the weighted energy density distribution across stores.
from ahl_targets.utils.plotting import configure_plots
from ahl_targets import PROJECT_DIR
from ahl_targets.getters import get_data
from ahl_targets.pipeline import model_data
from ahl_targets.pipeline import stores_transformation as stores
import seaborn as sns
import matplotlib.pyplot as plt
from ahl_targets.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
import altair as alt
import pandas as pd
import logging
from functools import reduce


def plot_weighted_distribution(store_df: pd.DataFrame, store: str, file_name: str):
    """Plot weighted distribution of energy density across products.
    Args:
        store_df (pd.DataFrame): dataframe with energy density average per store
        store (str): store name
        file_name (str): name of the file
    Returns: None
    """
    ax = sns.histplot(
        x=store_df.ed,
        weights=list(store_df.vol_w_perc),
        bins=50,
        kde=True,
        kde_kws={"cut": 3},
        label="Volume weighted energy density",
        color="#0000FF",
    )
    sns.histplot(
        x=store_df.ed,
        weights=list(store_df.kcal_w_perc),
        bins=50,
        kde=True,
        kde_kws={"cut": 3},
        label="Kcal weighted energy density",
        color="#FDB633",
        ax=ax,
    )

    # Set plot labels and legend
    plt.xlabel("Energy density")
    plt.ylabel("Weighted density")
    plt.legend()
    plt.title(store + ": Distribution of energy density across products")

    # Display the plot
    file_path = f"outputs/figures/png/{file_name}.png"
    plt.savefig(PROJECT_DIR / file_path, bbox_inches="tight")
    plt.close()


def plot_boxplots_ed(data: pd.DataFrame, title: str) -> alt.Chart:
    """Plot boxplots of energy density across stores.
    Args:
        data (pd.DataFrame): dataframe with energy density per store
        title (str): title of the plot
    Returns:
        boxplot (altair.Chart): boxplot of energy density across stores
    """
    boxplot = (
        alt.Chart(data)
        .mark_boxplot(size=20)
        .encode(
            x=alt.X("ed:Q", title="Energy density"),
            y=alt.Y("store_cat:N", title="Store"),
            color=alt.Color("store_cat:N"),
        )
        .properties(width=600, height=400)
    )

    return configure_plots(
        boxplot,
        title,
        "",
        16,
        14,
        14,
    )


if __name__ == "__main__":
    # Get data
    logging.info("Getting data")
    prod_table = get_data.product_metadata()
    pur_rec_vol = get_data.purchase_records_volume()
    nut_recs = get_data.nutrition()
    store_coding = get_data.store_itemisation_coding()
    store_lines = get_data.store_itemisation_lines()
    gravity = get_data.get_gravity()
    fvn = get_data.get_fvn()
    store_coding = get_data.store_itemisation_coding()
    store_lines = get_data.store_itemisation_lines()
    # Get repor† files
    store_df = pd.read_csv(PROJECT_DIR / f"outputs/reports/store_info.csv")
    manuf_df = pd.read_csv(PROJECT_DIR / f"outputs/reports/manuf_info.csv")
    store_baseline_df = pd.read_csv(PROJECT_DIR / f"outputs/reports/store_baseline.csv")

    # Get product / purchase info
    logging.info("Getting product / purchase info")
    prod_purch_df = model_data.purchase_complete(
        prod_table,
        pur_rec_vol,
        gravity,
        nut_recs,
        fvn,
        store_coding,
        store_lines,
    )

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

    base_stats["weight_kcal"] = (
        base_stats["Gross Up Weight"] * base_stats["Energy KCal"]
    )
    base_stats["weight_vol"] = base_stats["Gross Up Weight"] * base_stats["volume_up"]

    # create dfa grouped by store and product
    prod_base_stats_w = (
        base_stats.groupby(["store_cat", "product_code"])[["weight_kcal", "weight_vol"]]
        .sum()
        .reset_index()
    )
    prod_base_stats_ed = (
        base_stats.groupby(["store_cat", "product_code"])[["ed"]].mean().reset_index()
    )
    # merge the two dataframes
    prod_base_stats = prod_base_stats_w.merge(
        prod_base_stats_ed[["product_code", "store_cat", "ed"]],
        how="left",
        left_on=["product_code", "store_cat"],
        right_on=["product_code", "store_cat"],
    )

    # Subset stores
    prod_base_stats_sub = stores.store_subset(prod_base_stats.copy())

    # Transform weights into % of total of store
    prod_base_stats_sub["kcal_w_perc"] = (
        prod_base_stats_sub["weight_kcal"]
        / prod_base_stats_sub.groupby("store_cat")["weight_kcal"].transform("sum")
        * 100
    )
    prod_base_stats_sub["vol_w_perc"] = (
        prod_base_stats_sub["weight_vol"]
        / prod_base_stats_sub.groupby("store_cat")["weight_vol"].transform("sum")
        * 100
    )

    # Weights as fraction of total store multiplied by 100000 to get a whole number
    prod_base_stats_sub["kcal_w"] = (
        prod_base_stats_sub["weight_kcal"]
        / prod_base_stats_sub.groupby("store_cat")["weight_kcal"].transform("sum")
        * 100000
    ).round(0)
    prod_base_stats_sub["vol_w"] = (
        prod_base_stats_sub["weight_vol"]
        / prod_base_stats_sub.groupby("store_cat")["weight_vol"].transform("sum")
        * 100000
    ).round(0)

    # Repetition of rows based on weight
    weighted_data_vol = prod_base_stats_sub.loc[
        prod_base_stats_sub.index.repeat(prod_base_stats_sub.vol_w)
    ]
    weighted_data_kcal = prod_base_stats_sub.loc[
        prod_base_stats_sub.index.repeat(prod_base_stats_sub.kcal_w)
    ]

    # Save plots
    # Disable max rows
    alt.data_transformers.disable_max_rows()
    webdr = google_chrome_driver_setup()
    # Boxplots
    save_altair(
        plot_boxplots_ed(
            weighted_data_vol, "Distribution of weighted energy density - volume"
        ),
        "dist_vol_ed_store",
        driver=webdr,
    )

    save_altair(
        plot_boxplots_ed(
            weighted_data_kcal, "Distribution of weighted energy density - kcal"
        ),
        "dist_kcal_ed_store",
        driver=webdr,
    )
    # Histograms
    for store in prod_base_stats_sub["store_cat"].unique():
        store_df = prod_base_stats_sub[prod_base_stats_sub["store_cat"] == store].copy()
        file_name = store + "weighted_ed_dist"
        plot_weighted_distribution(store_df, store, file_name)
        plt.clf()
