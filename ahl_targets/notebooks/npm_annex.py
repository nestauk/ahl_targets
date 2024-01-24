from ahl_targets.getters import simulated_outcomes as get_sim_data
from ahl_targets.getters import get_data
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


# read data
store_data = get_data.model_data()
results_df = get_sim_data.npm_agg()

# create aggregate data with weights
store_weight_npm = su.weighted_npm(store_data)
store_weight_npm["prod_weight_g"] = store_weight_npm.pipe(su.prod_weight_g)

# sales of NPM >=4

store_data["high_npm"] = store_data["npm_score"] >= 4

(
    store_data["Gross Up Weight"] * store_data["volume_up"] * store_data["high_npm"]
).sum() / (store_data["Gross Up Weight"] * store_data["volume_up"]).sum()

(
    store_data["Gross Up Weight"]
    * store_data["Quantity"]
    * store_data["Energy KCal"]
    * store_data["high_npm"]
).sum() / (
    store_data["Gross Up Weight"] * store_data["Quantity"] * store_data["Energy KCal"]
).sum()

# figure 2

store_data["npm_r"] = store_data["npm_score"].round(0)

store_data["volume_w"] = store_data["Gross Up Weight"] * store_data["volume_up"]

npm_dist = (
    store_data.groupby("npm_r")["volume_w"].sum() / store_data["volume_w"].sum() * 100
)

npm_df = pd.DataFrame(npm_dist).reset_index()

npm_df.to_csv(PROJECT_DIR / "outputs/reports/appendix_charts/chart2.csv", index=False)


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
avg_retailer.to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartY.csv", index=False)


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

baseline_prod.to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartC.csv")


# Data for chart C
baseline_prod["npm_rounded"] = baseline_prod["npm_w"].round(0)
# Percent of products with each NPM score
npm_share = (
    (baseline_prod["npm_rounded"].value_counts(normalize=True) * 100)
    .reset_index()
    .rename(columns={"index": "npm", "npm_rounded": "Percent Share"})
)

alt.data_transformers.disable_max_rows()


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
