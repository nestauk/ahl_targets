from ahl_targets.getters import get_data
from ahl_targets import PROJECT_DIR
import pandas as pd

npm = get_data.full_npm()
prd_table = get_data.product_metadata()
prd_name = get_data.product_master()

# select minimum NPM score for each product
min_npm_score_rows = npm.groupby("product_code")["npm_score"].idxmin()
selected_row = npm.loc[min_npm_score_rows]

# merge npm and prd_table
df = selected_row.merge(prd_table, on="product_code", how="left").merge(
    prd_name, left_on="product_code", right_on="Product Code", how="left"
)

df0 = df.copy()

# remove salt

exclude_market = [
    "Salt",
    "Herbs+Spices",
    "Meat Extract",
]

exclude_ext = ["Other Baking Ingredients"]

df = df[
    (df["rst_4_market"].isin(exclude_market) == False)
    & (df["rst_4_extended"].isin(exclude_ext) == False)
]

# convert NPM score
df["npm_score_r"] = -2 * df["npm_score"] + 70

# remove implausible values

df = df[df["npm_score_r"] >= 0]

# remove rows with values of nutrients larger than 100
df = df[
    (df["sat_per_100g"] <= 100)
    & (df["sug_per_100g"] <= 100)
    & (df["fibre_per_100g"] <= 100)
    & (df["prot_per_100g"] <= 100)
]

# average

avg_df = (
    df.groupby("npm_score_r")[
        "kcal_per_100g",
        "sat_per_100g",
        "sug_per_100g",
        "fibre_per_100g",
        "sod_per_100g",
        "prot_per_100g",
    ]
    .mean()
    .reset_index()
)

avg_df.to_csv(
    PROJECT_DIR / "outputs/reports/appendix_charts/chart3_mean.csv", index=False
)

# scatter

scatter_df = df[
    [
        "product_code",
        "npm_score_r",
        "kcal_per_100g",
        "sat_per_100g",
        "sug_per_100g",
        "fibre_per_100g",
        "sod_per_100g",
        "prot_per_100g",
    ]
].drop_duplicates()

scatter_df.to_csv(
    PROJECT_DIR / "outputs/reports/appendix_charts/chart3_scatter.csv", index=False
)
