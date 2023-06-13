from ahl_targets.utils.plotting import configure_plots
from ahl_targets import PROJECT_DIR
from ahl_targets.getters import get_data
from ahl_targets.pipeline import model_data
from ahl_targets.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
import altair as alt
import pandas as pd


def plot_shares_bar(metric_df, category_col, value_col, title, x_title, y_title):
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

# Get product / purchase info
prod_purch_df = model_data.purchase_complete(
    prod_table,
    pur_rec_vol,
    gravity,
    nut_recs,
    fvn,
    store_coding,
    store_lines,
)

# Check products after liquid exclusion
category_unqiue_df = (
    prod_purch_df.groupby(["rst_4_market"])["rst_4_extended"].unique().reset_index()
)
category_unqiue_df.to_csv(
    PROJECT_DIR / f"outputs/data/unique_categories_purchase_data.csv", index=False
)

# Plotting
webdr = google_chrome_driver_setup()
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
