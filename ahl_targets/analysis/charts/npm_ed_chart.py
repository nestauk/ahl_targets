from ahl_targets.getters import get_data
from ahl_targets import PROJECT_DIR

npm = get_data.full_npm()
prd_table = get_data.product_metadata()
prd_name = get_data.product_master()

# merge npm and prd_table
df = npm.merge(prd_table, on="product_code", how="left").merge(
    prd_name, left_on="product_code", right_on="Product Code", how="left"
)

# remove salt

df = df[df["rst_4_market"] != "Salt"]

# convert NPM score
df["npm_score_r"] = -2 * df["npm_score"] + 70

# check implausible values

imp = df[df["npm_score_r"] < 0]

# remove implausible values

df = df[df["npm_score_r"] >= 0]

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
