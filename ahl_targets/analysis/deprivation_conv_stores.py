from ahl_targets.pipeline import (
    stores_transformation as stores,
)
from ahl_targets.getters import get_data
from ahl_targets.utils import create_tables
import pandas as pd
from ahl_targets.utils.io import load_with_encoding
from ahl_targets import BUCKET_NAME
from ahl_targets import PROJECT_DIR
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

pur_rec_vol = get_data.purchase_records_volume()
nut_recs = get_data.nutrition()
store_lines = get_data.store_itemisation_lines()
store_coding = get_data.store_itemisation_coding()
store_lines = get_data.store_itemisation_lines()
demog_coding = get_data.demog_coding()
demog_val = get_data.demog_val()
pan_mast = get_data.household_master()
# Add in weights (from s3)
household_weights = pd.read_csv(
    load_with_encoding(BUCKET_NAME, "weights/rake_weights_in_home.csv")
)
imd_table = pd.read_csv(f"{PROJECT_DIR}/outputs/data/hh_table_imd.csv")


weights_info = household_weights[["Panel.Id", "weight"]].copy()
weights_info.rename(columns={"Panel.Id": "Panel Id"}, inplace=True)

demog_table = create_tables.hh_demographic_table(demog_coding, demog_val, pan_mast)
store_taxonomy = stores.taxonomy(store_coding, store_lines)
store_taxonomy = stores.custom_taxonomy(store_taxonomy)
store_taxonomy["chosen_store"] = stores.chosen_stores(store_taxonomy)


def household_purchases_table(store_variable):
    hh_info = (
        demog_table[["Household Income", "Postocde District"]].reset_index().copy()
    )
    store_info = store_taxonomy[["store_id", store_variable]].copy()
    store_info.rename(
        columns={"store_id": "Store Code", store_variable: "Store Category"},
        inplace=True,
    )

    hh_purchases = (
        pur_rec_vol[["Panel Id", "Store Code", "volume_up"]]
        .copy()
        .merge(
            store_info,
            on="Store Code",
            how="left",
        )
        .drop(columns=["Store Code"])
        .groupby(["Panel Id", "Store Category"])
        .sum()
        .reset_index()
    )

    hh_purchases_depr = hh_purchases.merge(hh_info, on="Panel Id", how="left").merge(
        imd_table[["Panel Id", "Index of Multiple Deprivation (IMD) Decile"]],
        on="Panel Id",
        how="left",
    )
    # Pivot hh_purchases to show percent of purchases in each store category
    return (
        hh_purchases_depr.pivot(
            index="Panel Id", columns="Store Category", values="volume_up"
        )
        .reset_index()
        .fillna(0)
        .merge(hh_info, on="Panel Id", how="left")
        .merge(
            imd_table[["Panel Id", "Index of Multiple Deprivation (IMD) Decile"]],
            on="Panel Id",
            how="left",
        )
        .merge(weights_info, on="Panel Id", how="left")
        .drop(columns=["Postocde District"])
    )


hh_purchases_total_2 = household_purchases_table("itemisation_level_2")
hh_purchases_total_cs = household_purchases_table("chosen_store")

stores_2 = [
    "Total Butchers",
    "Total Chemists/Drugstores",
    "Total Independents & Symbols",
    "Total Multiples",
    "Total Other Non-Grocers",
]
stores_cs = ["Chosen store", "Not chosen store"]


def get_store_purchases(hh_purchases_total, stores):
    for store in stores:
        hh_purchases_total["Weighted " + store] = (
            hh_purchases_total[store] * hh_purchases_total["weight"]
        )
    # get the percent of store purchases grouped by imd decile
    imd_store_purch = (
        hh_purchases_total[
            ["Weighted " + store for store in stores]
            + ["Index of Multiple Deprivation (IMD) Decile"]
        ]
        .dropna(subset="Index of Multiple Deprivation (IMD) Decile")
        .fillna(0)
        .groupby("Index of Multiple Deprivation (IMD) Decile")
        .sum()
    )
    # Get percent of store purchases by imd decile
    imd_store_purch_perc = (
        imd_store_purch.div(imd_store_purch.sum(axis=1), axis=0) * 100
    )
    # get the percent of store purchases grouped by income
    income_store_purch = (
        hh_purchases_total[
            ["Weighted " + store for store in stores] + ["Household Income"]
        ]
        .fillna(0)
        .groupby("Household Income")
        .sum()
    )
    # Get percent of store purchases by income
    income_store_purch_perc = (
        income_store_purch.div(income_store_purch.sum(axis=1), axis=0) * 100
    )
    return (
        imd_store_purch_perc,
        income_store_purch_perc,
        imd_store_purch,
        income_store_purch,
    )


(
    imd_store_purch_perc_2,
    income_store_purch_perc_2,
    imd_store_purch_2,
    income_store_purch_2,
) = get_store_purchases(hh_purchases_total_2, stores_2)
(
    imd_store_purch_perc_cs,
    income_store_purch_perc_cs,
    imd_store_purch_cs,
    income_store_purch_cs,
) = get_store_purchases(hh_purchases_total_cs, stores_cs)

# plot income_store_purch_perc as a heatmap
plt.figure(figsize=(10, 5))
plt.title("Percent of store purchases by income - level 2 stores")
income_heat = sns.heatmap(
    income_store_purch_perc_2, annot=True, fmt=".1f", cmap="Blues"
)
# Save the figure
plt.savefig(
    f"{PROJECT_DIR}/outputs/figures/png/heatmap_income_level2.png",
    dpi=300,
    bbox_inches="tight",
)

# plot imd_store_purch_perc as a heatmap
plt.figure(figsize=(10, 5))
plt.title("Percent of store purchases by IMD decile - level 2 stores")
ax = sns.heatmap(imd_store_purch_perc_2, annot=True, fmt=".1f", cmap="Blues")
# Save the figure
plt.savefig(
    f"{PROJECT_DIR}/outputs/figures/png/heatmap_imd_level2.png",
    dpi=300,
    bbox_inches="tight",
)

# plot income_store_purch_perc as a heatmap
plt.figure(figsize=(10, 5))
plt.title("Percent of store purchases by income - chosen stores")
income_heat = sns.heatmap(
    income_store_purch_perc_cs, annot=True, fmt=".1f", cmap="Blues"
)
# Save the figure
plt.savefig(
    f"{PROJECT_DIR}/outputs/figures/png/heatmap_income_cs.png",
    dpi=300,
    bbox_inches="tight",
)

# plot imd_store_purch_perc as a heatmap
plt.figure(figsize=(10, 5))
plt.title("Percent of store purchases by IMD decile - chosen stores")
ax = sns.heatmap(imd_store_purch_perc_cs, annot=True, fmt=".1f", cmap="Blues")
# Save the figure
plt.savefig(
    f"{PROJECT_DIR}/outputs/figures/png/heatmap_imd_cs.png",
    dpi=300,
    bbox_inches="tight",
)

chi2, p, _, _ = chi2_contingency(income_store_purch_cs)
print(f"Chi2 value: {chi2}")
print(f"P-value: {p}")

chi2, p, _, _ = chi2_contingency(imd_store_purch_cs)
print(f"Chi2 value: {chi2}")
print(f"P-value: {p}")
