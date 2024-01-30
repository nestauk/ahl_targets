from ahl_targets.getters import simulated_outcomes as get_sim_data
from ahl_targets.getters import get_data
from ahl_targets import PROJECT_DIR
import pandas as pd
from ahl_targets.utils import simulation_utils as su
import matplotlib.pyplot as plt
import os
from ahl_targets.pipeline import product_transformation as pt


path = PROJECT_DIR / "outputs/reports/chart_csv"

# check whether directory already exists
if not os.path.exists(path):
    os.mkdir(path)


results_df = get_sim_data.hfss_agg()
results_data_df = get_sim_data.hfss_full()


store_data = get_data.model_data()
store_data = store_data.pipe(pt.type).pipe(pt.is_food).pipe(pt.in_scope)
store_weight_hfss = su.weighted_hfss(store_data)
store_weight_hfss["prod_weight_g"] = store_weight_hfss.pipe(su.prod_weight_g)


# create aggregated file by store
grouped_mean = (
    results_df.groupby(
        [
            "store",
            "product_share_reform_low",
            "product_share_reform_medium",
            "product_share_reform_high",
            "product_share_sale",
            "sales_change_high",
            "sales_change_low",
        ]
    )
    .mean()
    .reset_index()
)


# create scenario file

all_sum = (
    grouped_mean.groupby(
        [
            "product_share_reform_low",
            "product_share_reform_medium",
            "product_share_reform_high",
            "product_share_sale",
            "sales_change_high",
            "sales_change_low",
        ]
    )
    .sum()
    .reset_index()
)

all_sum["hfss_share_kg_baseline"] = (
    all_sum["hfss_kg_baseline"] / all_sum["total_kg_baseline"]
)
all_sum["hfss_share_kg_new"] = all_sum["hfss_kg_new"] / all_sum["total_kg_new"]

all_sum["kcal_pp_baseline"] = all_sum["total_kcal_baseline"] / 65121700 / 365
all_sum["kcal_pp_new"] = all_sum["total_kcal_new"] / 65121700 / 365

all_sum["spend_baseline"] = all_sum["total_spend_baseline"] / 65121700 / 365
all_sum["spend_new"] = all_sum["total_spend_new"] / 65121700 / 365

# Add before after variables
baseline_columns = all_sum.filter(like="_baseline")
new_columns = all_sum.filter(like="_new")

baseline_columns.columns = baseline_columns.columns.str.replace("_baseline", "")
new_columns.columns = new_columns.columns.str.replace("_new", "")

result = 100 * (new_columns - baseline_columns) / baseline_columns
df = pd.concat([all_sum, result.add_suffix("_diff_percentage")], axis=1)
df["kcal_diff"] = df["kcal_pp_new"] - df["kcal_pp_baseline"]


# baseline HFSS share

# sales weighted distribution across retailers
bar_df = (
    store_weight_hfss.query("hfss == 1").groupby(["store_cat"])["total_kg"].sum()
    / store_weight_hfss.groupby(["store_cat"])["total_kg"].sum()
).reset_index()


total = {
    "store_cat": "total",
    "total_kg": store_weight_hfss.query("hfss == 1")["total_kg"].sum()
    / store_weight_hfss["total_kg"].sum(),
    "store_letter": "total",
}


bar_df_sort = bar_df.sort_values(by="total_kg", ascending=False).copy()

keys = bar_df_sort["store_cat"]

values = [f"Store {chr(65 + i)}" for i in range(len(keys))]

store_letters = dict(zip(keys, values))


bar_df_sort["store_letter"] = bar_df_sort["store_cat"].map(store_letters)


bar_df_tot = bar_df_sort.append(total, ignore_index=True)

bar_df_tot = bar_df_tot.sort_values(by="total_kg", ascending=True).copy()


def barh_chart(bar_df_tot):
    categories = bar_df_tot["store_letter"]
    means = bar_df_tot["total_kg"]

    # Colors for the bars
    colors = [
        "tab:blue",
        "tab:blue",
        "tab:blue",
        "tab:blue",
        "tab:blue",
        "tab:blue",
        "tab:blue",
        "tab:orange",
        "tab:blue",
        "tab:blue",
        "tab:blue",
        "tab:blue",
    ]

    # Create the horizontal bar chart
    plt.figure(figsize=(8, 5))
    plt.barh(categories, means, capsize=5, color=colors)

    # Customize the plot
    plt.xlabel("Sales weighted HFSS share")
    plt.ylabel("")
    plt.title("HFSS share")
    plt.grid(axis="x")
    plt.tight_layout()

    plt.savefig(PROJECT_DIR / "outputs/figures/png/annex/hfss_barh.png")

    return plt


barh_chart(bar_df_tot)

# product share
prod_hfss = (
    store_data["Gross Up Weight"] * store_data["Quantity"] * store_data["in_scope"]
).sum() / (store_data["Gross Up Weight"] * store_data["Quantity"]).sum()

prod_not_hfss = 1 - prod_hfss

print(prod_hfss, prod_not_hfss)

# calories share

(
    store_data["Gross Up Weight"]
    * store_data["Quantity"]
    * store_data["in_scope"]
    * store_data["Energy KCal"]
).sum() / (
    store_data["Gross Up Weight"] * store_data["Quantity"] * store_data["Energy KCal"]
).sum()


# target

target = df[(df["sales_change_high"] == 12.5) & (df["sales_change_low"] == 2.5)]

target["hfss_share_kg_new"]
