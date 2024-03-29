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
import numpy as np


path = PROJECT_DIR / "outputs/reports/chart_csv"

# check whether directory already exists
if not os.path.exists(path):
    os.mkdir(path)

pd.set_option("display.max_columns", None)


def annex_chart_kcal(df):
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            y=alt.X("kcal_pp_new:Q", scale=alt.Scale(domain=[1400, 1700])),
            x="ed_reduction",
            color="sales_change_low",
        )
    )
    rule = alt.Chart(df).mark_rule(color="red").encode(y="mean(kcal_pp_baseline):Q")

    return configure_plots(
        chart + rule,
        "",
        "",
        16,
        14,
        14,
    )


def annex_chart_spend(df):
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            y=alt.X("spend_new:Q", scale=alt.Scale(domain=[15, 20])),
            x="ed_reduction",
            color="sales_change_low",
        )
    )
    rule = alt.Chart(df).mark_rule(color="red").encode(y="mean(spend_baseline):Q")

    return configure_plots(
        chart + rule,
        "",
        "",
        16,
        14,
        14,
    )


def density_plot(plt_df_sub, thres, band_df, band_df_2):
    chart = (
        alt.Chart(plt_df_sub)
        .transform_density(
            "share", as_=["size", "density"], groupby=["when"], bandwidth=50
        )
        .mark_line()
        .encode(
            x=alt.X(
                "size:Q",
                axis=alt.Axis(
                    title="Sales weighted average calorie density (kcal/100g)"
                ),
            ),
            y=alt.Y("density:Q", axis=alt.Axis(title="Weighted sales (%)", format="%")),
            color=alt.Color("when:N", legend=alt.Legend(title="")),
        )
    )
    line = alt.Chart(thres).mark_line().encode(x="x", y="y")

    areas1 = (
        alt.Chart(band_df)
        .mark_rect(opacity=0.2)
        .encode(
            x="start",
            x2="stop",
            y=alt.value(0),  # pixels from top
            y2=alt.value(200),  # pixels from top
            color="index:N",
        )
    )

    areas2 = (
        alt.Chart(band_df_2)
        .mark_rect(opacity=0.2)
        .encode(
            x="start",
            x2="stop",
            y=alt.value(0),  # pixels from top
            y2=alt.value(200),  # pixels from top
        )
    )

    layered_chart = (chart + line + areas1 + areas2).properties(height=300, width=600)

    return configure_plots(
        layered_chart,
        "",
        "",
        16,
        14,
        14,
    )


# read data
store_data = get_data.model_data()
results_df = get_sim_data.energy_density_agg()
excluded_data = get_data.excluded_data()

# create aggregate data with weights

store_weight = su.weighted_ed(store_data)
store_weight["prod_weight_g"] = store_weight.pipe(su.prod_weight_g)

# share of sales

store_data["high_ed"] = store_data["ed"] >= 400

# share of sales accounted for by high ed products
(
    store_data["Gross Up Weight"] * store_data["volume_up"] * store_data["high_ed"]
).sum() / (store_data["Gross Up Weight"] * store_data["volume_up"]).sum()


# share of kcal accounted for by high ed products

(
    store_data["Gross Up Weight"]
    * store_data["Quantity"]
    * store_data["Energy KCal"]
    * store_data["high_ed"]
).sum() / (
    store_data["Gross Up Weight"] * store_data["Quantity"] * store_data["Energy KCal"]
).sum()


# unweighted average

store_data["w"] = store_data["Gross Up Weight"] * store_data["Quantity"]
excluded_data["w"] = excluded_data["Gross Up Weight"] * excluded_data["Quantity"]


def weighted_mean(values, weights):
    # Check if the lengths of values and weights match
    if len(values) != len(weights):
        raise ValueError("Lengths of values and weights must be the same.")

    # Calculate the weighted mean
    return np.average(values, weights=weights)


weighted_mean(store_data["ed"], store_data["w"])
weighted_mean(excluded_data["ed"], excluded_data["w"])

store_data.groupby("store_cat").apply(lambda x: weighted_mean(x["ed"], x["w"])).std()

# manufacturers
manuf = pd.concat([store_data, excluded_data], ignore_index=True)

weighted_mean(manuf["ed"], manuf["w"])

manuf.groupby("manufacturer").apply(lambda x: weighted_mean(x["ed"], x["w"])).std()


# generate average df over all iterations

avg = (
    results_df.groupby(
        [
            "product_share_reform",
            "product_share_sale",
            "sales_change_high",
            "sales_change_low",
            "ed_reduction",
        ]
    )[
        [
            "mean_ed_kg_new",
            "mean_ed_kcal_new",
            "mean_ed_kg_baseline",
            "mean_ed_kcal_baseline",
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

# Create df of average ed by store
avg_retailer = (
    (store_weight["kg_w"] * store_weight["ed"]).groupby(store_weight["store_cat"]).sum()
    / store_weight["kg_w"].groupby(store_weight["store_cat"]).sum()
).reset_index(name="ed")
# Add in row manually for where store == 'Target' and ed == avg['mean_ed_kg_new'] where ed_reduction == 10, sales_change_low == 5 and sales_change_high == 15
avg_retailer = pd.concat(
    [
        avg_retailer,
        pd.DataFrame(
            {
                "store_cat": ["Target"],
                "ed": [
                    avg[
                        (avg["ed_reduction"] == 10)
                        & (avg["sales_change_low"] == 5)
                        & (avg["sales_change_high"] == 15)
                    ]["mean_ed_kg_new"].values[0]
                ],
            }
        ),
    ],
    ignore_index=True,
)
# Save to csv (for use in chartX)
avg_retailer.to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartX.csv", index=False)


# Generate before-after variables
baseline_columns = avg.filter(like="_baseline")
new_columns = avg.filter(like="_new")

baseline_columns.columns = baseline_columns.columns.str.replace("_baseline", "")
new_columns.columns = new_columns.columns.str.replace("_new", "")

# Calculate percentage difference
result = (new_columns - baseline_columns) / baseline_columns * 100
df = pd.concat([avg, result.add_suffix("_diff_percentage")], axis=1)
df["kcal_diff"] = df["kcal_pp_new"] - df["kcal_pp_baseline"]


# generate summary chart
webdr = google_chrome_driver_setup()

save_altair(
    annex_chart_kcal(df),
    "annex/energy_density_kcal",
    driver=webdr,
)

save_altair(
    annex_chart_spend(df),
    "annex/energy_density_spend",
    driver=webdr,
)


baseline_prod = (
    (store_weight["ed"] * store_weight["kg_w"])
    .groupby(store_weight["product_code"])
    .sum()
    / store_weight.groupby(["product_code"])["kg_w"].sum()
).reset_index(name="share")
plt_df_sub = baseline_prod.sample(5000)

thres = pd.DataFrame({"x": [400, 400], "y": [0, 0.003]})

band_df = pd.DataFrame({"start": [400, 400], "stop": [0, 0]})

band_df_2 = pd.DataFrame({"start": [900, 900], "stop": [0, 0]})


# weighted npm average by product
baseline_prod.to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartA.csv")


# generate summary chart
webdr = google_chrome_driver_setup()

save_altair(
    density_plot(plt_df_sub, thres, band_df, band_df_2),
    "annex/density_plot",
    driver=webdr,
)


# weighted npm average by product and store
baseline_prod_store = (store_weight["ed"] * store_weight["kg_w"]).groupby(
    [store_weight["store_cat"], store_weight["product_code"]]
).sum() / store_weight.groupby(["product_code", "store_cat"])["kg_w"].sum()

baseline_prod_store = baseline_prod_store.reset_index(name="share")

baseline_prod_store.to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartE1.csv")


def violin_plot(baseline):
    chart = (
        alt.Chart(baseline.sample(5000))
        .transform_density(
            "share", as_=["share", "density"], extent=[-100, 900], groupby=["store_cat"]
        )
        .mark_area(orient="horizontal")
        .encode(
            y="share:Q",
            color="store_cat:N",
            x=alt.X(
                "density:Q",
                stack="center",
                impute=None,
                title=None,
                axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
            ),
            column=alt.Column(
                "store_cat:N",
                header=alt.Header(
                    titleOrient="bottom",
                    labelOrient="bottom",
                    labelPadding=0,
                ),
            ),
        )
        .properties(width=100)
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
    )

    return configure_plots(
        chart,
        "",
        "",
        16,
        14,
        14,
    )


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


result = (
    pd.concat(
        [
            weighted_stddev_by_group(
                store_weight["ed"], store_weight["kg_w"], store_weight["store_cat"]
            )[0],
            weighted_stddev_by_group(
                store_weight["ed"], store_weight["kg_w"], store_weight["store_cat"]
            )[1],
        ],
        axis=1,
    )
    .reset_index()
    .rename(columns={0: "std", 1: "mean"})
)


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


# dictionary with overall mean and standard deviation
total = {
    "store_cat": "total",
    "mean": weighted_stddev(
        store_data.groupby(["product_code"])["ed"].mean(),
        (store_data["volume_up"] * store_data["Gross Up Weight"])
        .groupby(store_data["product_code"])
        .sum(),
    )[1],
    "std": weighted_stddev(
        store_data.groupby(["product_code"])["ed"].mean(),
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


# append total to store df
bar_df_tot = result_sort.append(total, ignore_index=True)

bar_df_tot = bar_df_tot.sort_values(by="mean", ascending=True).copy()

bar_df_tot.to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartE1_bar.csv")


def ridge_plot(source, step=30, overlap=0.5):
    chart = (
        alt.Chart(source, height=step)
        .transform_joinaggregate(mean_temp="mean(share)", groupby=["store_cat"])
        .transform_bin(["bin_max", "bin_min"], "share")
        .transform_aggregate(
            value="count()", groupby=["store_cat", "mean_temp", "bin_min", "bin_max"]
        )
        .transform_impute(
            impute="value", groupby=["store_cat", "mean_temp"], key="bin_min", value=0
        )
        .mark_area(
            interpolate="monotone", fillOpacity=0.8, stroke="lightgray", strokeWidth=0.5
        )
        .encode(
            alt.X(
                "bin_min:Q",
                bin="binned",
                title="Sales weighted Calorie Density",
                scale=alt.Scale(domain=[0, 800]),
            ),
            alt.Y("value:Q", scale=alt.Scale(range=[step, -step * overlap]), axis=None),
            alt.Fill(
                "mean_temp:Q",
                legend=None,
                scale=alt.Scale(domain=[270, 290], scheme="redyellowblue"),
            ),
            alt.Row("store_cat:N", title=None, header=alt.Header(labelAngle=0)),
        )
        .properties(title="", bounds="flush")
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
        .configure_title(anchor="end")
        .configure_axisY(
            labelAngle=0  # Set the labelAngle property to 0 for horizontal labels on the y-axis
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


def barh_chart(bar_df_tot):
    categories = bar_df_tot["store_letter"]
    means = bar_df_tot["mean"]
    std_devs = bar_df_tot["std"]

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
    plt.barh(categories, means, xerr=std_devs, capsize=5, color=colors)

    # Customize the plot
    plt.xlabel("Sales weighted calorie density")
    plt.ylabel("")
    plt.title("Average and standard deviation")
    plt.grid(axis="x")
    plt.tight_layout()

    plt.savefig(PROJECT_DIR / "outputs/figures/png/annex/energy_density_barh.png")

    return plt


# generate summary chart
webdr = google_chrome_driver_setup()

save_altair(
    violin_plot(baseline_prod_store),
    "annex/violin_plot",
    driver=webdr,
)


source = baseline_prod_store.sample(5000)[["store_cat", "share"]]

# generate summary chart
webdr = google_chrome_driver_setup()

save_altair(
    ridge_plot(source),
    "annex/ridge_plot",
    driver=webdr,
)


barh_chart(bar_df_tot)


results_df[
    [
        "sales_change_high",
        "sales_change_low",
        "ed_reduction",
        "spend_new",
        "spend_baseline",
    ]
].to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartG1.csv", index=False)


results_df[
    [
        "sales_change_high",
        "sales_change_low",
        "ed_reduction",
        "kcal_pp_baseline",
        "kcal_pp_new",
    ]
].to_csv(PROJECT_DIR / "outputs/reports/chart_csv/chartF1.csv", index=False)


# distributionf of sales weighted (in kg) ed

store_data["ed_r"] = store_data["ed"].round(0)

store_data["volume_w"] = store_data["Gross Up Weight"] * store_data["volume_up"]

ed_dist = (
    store_data.groupby("ed_r")["volume_w"].sum() / store_data["volume_w"].sum() * 100
)

ed_df = pd.DataFrame(ed_dist).reset_index()

ed_df.to_csv(PROJECT_DIR / "outputs/reports/appendix_charts/chart1.csv", index=False)
