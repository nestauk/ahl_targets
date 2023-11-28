from ahl_targets.getters import simulated_outcomes as get_sim_data
from ahl_targets.getters import get_data
from ahl_targets.pipeline import product_transformation as pt
from ahl_targets.utils.plotting import configure_plots
from ahl_targets.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
import altair as alt
from ahl_targets import PROJECT_DIR
import pandas as pd
from ahl_targets.utils import simulation_utils as su
import matplotlib.pyplot as plt
import os

path = PROJECT_DIR / "outputs/reports/chart_csv"

# check whether directory already exists
if not os.path.exists(path):
    os.mkdir(path)


def npm_density_plot(plt_df_sub):
    chart = (
        alt.Chart(plt_df_sub)
        .transform_density(
            "npm_w", as_=["size", "density"], groupby=["when"], bandwidth=2
        )
        .mark_line()
        .encode(
            x=alt.X(
                "size:Q",
                axis=alt.Axis(
                    title="Sales weighted average NPM score",
                ),
            ),
            y=alt.Y("density:Q", axis=alt.Axis(title="Weighted sales (%)", format="%")),
            color=alt.Color("when:N", legend=alt.Legend(title="")),
        )
    )
    return configure_plots(
        chart,
        "",
        "",
        16,
        14,
        14,
    )


# read data
store_data = get_data.model_data()
results_df = get_sim_data.npm_agg()
npm_data = get_data.get_npm()

# Get product categories
prod_cats = (
    store_data[["Product Code", "store_cat", "rst_4_market_sector"]]
    .drop_duplicates()
    .copy()
)

# create aggregate data with weights
store_weight_npm = su.weighted_npm(store_data)
store_weight_npm["prod_weight_g"] = store_weight_npm.pipe(su.prod_weight_g)


# HFSS info
store_data_hfss = pt.type(store_data)
store_data_hfss = pt.in_scope(store_data_hfss)

store_data_hfss["weight_kcal"] = (
    store_data_hfss["Gross Up Weight"] * store_data_hfss["Energy KCal"]
)
store_data_hfss["weight_vol"] = (
    store_data_hfss["Gross Up Weight"] * store_data_hfss["volume_up"]
)
store_data_hfss["unweighted"] = store_data_hfss["Gross Up Weight"]

# HFSS volume weighted shares
hfss_shares_volume = (
    store_data_hfss.groupby(["in_scope"])["weight_vol"].sum()
    / store_data_hfss["weight_vol"].sum()
)
# HFSS product weighted shares
hfss_shares_unweighted = (
    store_data_hfss.groupby(["in_scope"])["unweighted"].sum()
    / store_data_hfss["unweighted"].sum()
)

hfss_shares_kcal = (
    store_data_hfss.groupby(["in_scope"])["weight_kcal"].sum()
    / store_data_hfss["weight_kcal"].sum()
)

# Shares of unique products sold (sort by purchase date)
unique_prods = store_data_hfss.sort_values(
    by=["Purchase Date"], ascending=False
).drop_duplicates(subset=["product_code"], keep="first")
unique_prods_sold = (
    unique_prods.groupby(["in_scope"])["product_code"].nunique()
    / unique_prods["product_code"].nunique()
)


# Create new column high NPM >= 4 (1 else 0)
store_data_hfss["high_npm"] = store_data_hfss["npm_score"].apply(
    lambda x: 1 if x >= 4 else 0
)

hfss_high_volume = (
    store_data_hfss.groupby(["high_npm"])["weight_vol"].sum()
    / store_data_hfss["weight_vol"].sum()
)

hfss_high_unweighted = (
    store_data_hfss.groupby(["high_npm"])["unweighted"].sum()
    / store_data_hfss["unweighted"].sum()
)

hfss_high_kcal = (
    store_data_hfss.groupby(["high_npm"])["weight_kcal"].sum()
    / store_data_hfss["weight_kcal"].sum()
)


# average across all iterations
avg = (
    results_df.groupby(
        [
            "product_share_reform",
            "product_share_sale",
            "sales_change_high",
            "sales_change_low",
            "npm_reduction",
        ]
    )[
        [
            "mean_npm_kg_new",
            "mean_npm_kcal_new",
            "mean_npm_kg_baseline",
            "mean_npm_kcal_baseline",
            "kcal_pp_baseline",
            "kcal_pp_new",
            "total_prod_baseline",
            "total_prod_new",
            "spend_baseline",
            "spend_new",
        ]
    ]
    .mean()
    .reset_index()
)

# Create df of average npm by store
avg_retailer = (
    (store_weight_npm["kg_w"] * store_weight_npm["npm_score"])
    .groupby(store_weight_npm["store_cat"])
    .sum()
    / store_weight_npm["kg_w"].groupby(store_weight_npm["store_cat"]).sum()
).reset_index(name="npm")

avg_retailer = (
    (store_weight_npm["kg_w"] * store_weight_npm["npm_score"])
    .groupby(store_weight_npm["store_cat"])
    .sum()
    / store_weight_npm["kg_w"].sum()
).reset_index(name="npm")


# Add in row manually for where store == 'Target' and npm == avg['mean_npm_kg_new'] where npm_reduction == 3, sales_change_low == 5 and sales_change_high == 10
avg_retailer = pd.concat(
    [
        avg_retailer,
        pd.DataFrame(
            {
                "store_cat": "Target",
                "npm": [
                    avg[
                        (avg["npm_reduction"] == 3)
                        & (avg["sales_change_low"] == 5)
                        & (avg["sales_change_high"] == 10)
                    ]["mean_npm_kg_new"].values[0]
                ],
            }
        ),
    ],
    ignore_index=True,
)
# Save as csv (for use in chart Y)
avg_retailer.to_csv(
    PROJECT_DIR / "outputs/reports/chart_csv/chartY_updated.csv", index=False
)

# Category level NPM - for specific retailer
store_weight_npm_cat = (
    store_weight_npm.copy()
    .merge(
        prod_cats,
        left_on=["product_code", "store_cat"],
        right_on=["Product Code", "store_cat"],
        how="left",
    )
    .drop(columns=["Product Code"])
)

store_weight_npm_cat = store_weight_npm_cat[
    store_weight_npm_cat["store_cat"] == "Total Asda"
].copy()

retailer_npm_cat = (
    (store_weight_npm_cat["npm_score"] * store_weight_npm_cat["kg_w"])
    .groupby(store_weight_npm_cat["rst_4_market_sector"])
    .sum()
    / store_weight_npm_cat.groupby(["rst_4_market_sector"])["kg_w"].sum()
).reset_index(name="npm_w")

# Plot NPM by category (horizontal bar chart) sorted by npm
retailer_npm_cat.sort_values(by="npm_w", ascending=True).plot.barh(
    x="rst_4_market_sector",
    y="npm_w",
    figsize=(7, 5),
    legend=False,
)


# Generate before-after variables
baseline_columns = avg.filter(like="_baseline")
new_columns = avg.filter(like="_new")

baseline_columns.columns = baseline_columns.columns.str.replace("_baseline", "")
new_columns.columns = new_columns.columns.str.replace("_new", "")

# Calculate percentage difference
result = (new_columns - baseline_columns) / baseline_columns.abs() * 100

# Concatenate the result with the original DataFrame
df = pd.concat([avg, result.add_suffix("_diff_percentage")], axis=1)
df["kcal_diff"] = df["kcal_pp_new"] - df["kcal_pp_baseline"]

# weighted npm average by product
baseline_prod = (
    (store_weight_npm["npm_score"] * store_weight_npm["kg_w"])
    .groupby(store_weight_npm["product_code"])
    .sum()
    / store_weight_npm.groupby(["product_code"])["kg_w"].sum()
).reset_index(name="npm_w")

alt.data_transformers.disable_max_rows()
### Creating data for chart C (with updated NPM data) ###
chart_c_df = baseline_prod.copy()
chart_c_df["npm_w"] = ((-2) * chart_c_df["npm_w"]) + 70
npm_density_plot(chart_c_df)
chart_c_df.to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartC_updated.csv")

chart_c_df["npm_rounded"] = chart_c_df["npm_w"].round(0)
# Percent of products with each NPM score
npm_share = (
    (chart_c_df["npm_rounded"].value_counts(normalize=True) * 100)
    .reset_index()
    .rename(columns={"index": "npm", "npm_rounded": "Percent Share"})
)
npm_share.to_csv(
    PROJECT_DIR / "outputs/reports/chart_csv/chartC2_alternative_npm_share.csv",
    index=False,
)


### Chart B - nutrient distribution ###
npm_store_df = npm_data.merge(
    store_data[
        [
            "PurchaseId",
            "Period",
            "store_cat",
            "is_food",
            "itemisation_level_3",
            "rst_4_extended",
            "rst_4_market",
            "rst_4_market_sector",
            "rst_4_sub_market",
            "rst_4_trading_area",
        ]
    ],
    left_on=["purchase_id", "period"],
    right_on=["PurchaseId", "Period"],
    how="inner",
).drop(columns=["PurchaseId", "Period"])

# Products grouped by NPM score to get avg: sugar, salt...ect per 100g
prod_per_100 = (
    npm_store_df.groupby(["product_code", "rst_4_market"])[
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

# remove salt

prod_per_100 = prod_per_100[prod_per_100["rst_4_market"] != "Salt"]

prod_100_npm = prod_per_100.merge(
    chart_c_df[["product_code", "npm_w"]],
    left_on="product_code",
    right_on="product_code",
).drop(["product_code"], axis=1)

prod_100_npm["npm_w"] = prod_100_npm["npm_w"].round(0)

prod_100_npm.rename(
    columns={
        "npm_w": "npm_score",
    },
    inplace=True,
)


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
prod_100_npm.to_csv(
    PROJECT_DIR / f"outputs/reports/chart_csv/chartB_updated.csv", index=False
)


#### Previous code (for reference) ####

# Data for chart C
baseline_prod["npm_rounded"] = baseline_prod["npm_w"].round(0)
# Percent of products with each NPM score
npm_share = (
    (baseline_prod["npm_rounded"].value_counts(normalize=True) * 100)
    .reset_index()
    .rename(columns={"index": "npm", "npm_rounded": "Percent Share"})
)


# Updated version of Chart C with new NPM data
npm_prod_df = pd.read_csv(PROJECT_DIR / "outputs/data/chartc_updated_npm.csv")
npm_density_plot(npm_prod_df)

# Save plot
webdr = google_chrome_driver_setup()
save_altair(
    npm_density_plot(baseline_prod),
    "annex/npm_share_sales",
    driver=webdr,
)

# Save as csv (for use in chart C)
npm_share.to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartC2_v2.csv", index=False)


# Plot percent share as a line chart in altair (where the line is smoothed)
npm_share_chart = (
    alt.Chart(npm_share)
    .mark_line()
    .encode(
        x=alt.X("npm:Q", title="Sales weighted average NPM score"),
        y=alt.Y("Percent Share:Q", title="Weighted sales (%)"),
    )
    .properties(width=400, height=300)
    .interactive()
)
npm_share_chart_conf = configure_plots(
    npm_share_chart,
    "",
    "",
    16,
    14,
    14,
)

# generate summary chart
webdr = google_chrome_driver_setup()

save_altair(
    npm_share_chart_conf,
    "annex/npm_share_sales",
    driver=webdr,
)


# %%


# weighted npm average by product and store
baseline_prod_store = (
    store_weight_npm["npm_score"] * store_weight_npm["kg_w"]
).groupby(
    [store_weight_npm["store_cat"], store_weight_npm["product_code"]]
).sum() / store_weight_npm.groupby(
    ["product_code", "store_cat"]
)[
    "kg_w"
].sum()

baseline_prod_store = baseline_prod_store.reset_index(name="share")
baseline_prod_store.to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartE2.csv")


# %%


# functions to create weighted standard deviation (chatgpt helped here)


def weighted_stddev_by_group(data, weights, group_column):
    # Create a DataFrame from the data and weights
    df = pd.DataFrame({"data": data, "weights": weights})

    # Calculate the weighted mean for each group
    grouped_mean = df.groupby(group_column).apply(
        lambda x: sum(x["data"] * x["weights"]) / sum(x["weights"])
    )

    # Calculate the squared differences for each group
    df["squared_diff"] = (
        df["data"] - df.groupby(group_column)["data"].transform("mean")
    ) ** 2

    # Calculate the weighted variance for each group
    grouped_var = df.groupby(group_column).apply(
        lambda x: sum(x["squared_diff"] * x["weights"]) / sum(x["weights"])
    )

    # Calculate the weighted standard deviation for each group
    grouped_stddev = grouped_var**0.5

    return grouped_stddev, grouped_mean


# dataframe with mean and standard deviation of NPM by store

result = (
    pd.concat(
        [
            weighted_stddev_by_group(
                store_weight_npm["npm_score"],
                store_weight_npm["kg_w"],
                store_weight_npm["store_cat"],
            )[0],
            weighted_stddev_by_group(
                store_weight_npm["npm_score"],
                store_weight_npm["kg_w"],
                store_weight_npm["store_cat"],
            )[1],
        ],
        axis=1,
    )
    .reset_index()
    .rename(columns={0: "std", 1: "mean"})
)


# %%


def weighted_stddev(data, weights):
    # Check if data and weights have the same length
    if len(data) != len(weights):
        raise ValueError("Data and weights must have the same length.")

    # Calculate the weighted mean
    weighted_mean = sum(w * x for w, x in zip(weights, data)) / sum(weights)

    # Calculate the weighted variance
    weighted_var = sum(
        w * (x - weighted_mean) ** 2 for w, x in zip(weights, data)
    ) / sum(weights)

    # Calculate the weighted standard deviation
    weighted_stddev = weighted_var**0.5

    return weighted_stddev, weighted_mean


# %%


# dictionary with overall mean and standard deviation
total = {
    "store_cat": "total",
    "mean": weighted_stddev(
        store_data.groupby(["product_code"])["npm_score"].mean(),
        (store_data["volume_up"] * store_data["Gross Up Weight"])
        .groupby(store_data["product_code"])
        .sum(),
    )[1],
    "std": weighted_stddev(
        store_data.groupby(["product_code"])["npm_score"].mean(),
        (store_data["volume_up"] * store_data["Gross Up Weight"])
        .groupby(store_data["product_code"])
        .sum(),
    )[0],
    "store_letter": "total",
}

# sort by mean
result_sort = result.sort_values(by="mean", ascending=False).copy()

# anonimise stores by assigning letters
keys = result["store_cat"]

values = [f"Store {chr(65 + i)}" for i in range(len(keys))]

store_letters = dict(zip(keys, values))

result_sort["store_letter"] = result_sort["store_cat"].map(store_letters)


# %%


# append total to store df
bar_df_tot = result_sort.append(total, ignore_index=True)

bar_df_tot = bar_df_tot.sort_values(by="mean", ascending=True).copy()

bar_df_tot.to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartE2_bar.csv")


# %%


# simulation file with spend
results_df[
    [
        "sales_change_high",
        "sales_change_low",
        "npm_reduction",
        "spend_new",
        "spend_baseline",
    ]
].to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartG2.csv", index=False)


# %%


# simulation file with kcal
results_df[
    [
        "sales_change_high",
        "sales_change_low",
        "npm_reduction",
        "kcal_pp_baseline",
        "kcal_pp_new",
    ]
].to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartF2.csv", index=False)


# %%
