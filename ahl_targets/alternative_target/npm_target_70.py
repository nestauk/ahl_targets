from ahl_targets.getters import simulated_outcomes as get_sim_data
import pandas as pd


# read modelled data
results_df = get_sim_data.npm_agg()


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


# remove scenarions with kcal_diff > 0 and spend <0
df = df[(df["kcal_diff"] < 0) & (df["spend_diff_percentage"] > 0)]

# convert npm
df["npm_conv"] = -2 * df["mean_npm_kg_new"] + 70

# create a scatter plot with npm_conv and kcal_diff
df.plot.scatter(x="npm_conv", y="kcal_diff")
df.plot.scatter(x="kcal_diff", y="npm_conv")

df[(df["npm_conv"] >= 70) & (df["npm_conv"] < 71)]

df[(df["npm_conv"] >= 70) & (df["npm_conv"] < 71)]["kcal_diff"].mean()
