import pandas as pd
import numpy as np
from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME, PROJECT_DIR
from ahl_targets.utils import simulation_utils as su
from ahl_targets.getters import get_data
from ahl_targets.getters import simulated_outcomes as get_sim_data
import random

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    num_iterations = 5000
    npm_reduction_values = 3
    unhealthy_sales_change_values = 10.5
    healthy_sales_change_values = 9

    # read data

    store_data = get_data.model_data()
    prod_table = get_data.product_metadata()

    store_weight_npm = su.weighted_npm(store_data)
    store_weight_npm["prod_weight_g"] = store_weight_npm.pipe(su.prod_weight_g)

    unique_prd = store_weight_npm[["product_code", "npm_score"]].drop_duplicates()

    # Print the regression table
    regression_df = get_sim_data.regression_df()

    results = []

    # Nested loop to iterate through different values of product_share and ed_reduction
    # Repeat the code num_iterations times
    for _ in range(num_iterations):
        gb_pop = random.randint(65121700 * 0.98, 65121700 * 1.02)
        product_share_reform = random.uniform(0.45, 0.55)

        # Generate random numbers from limits in coefficients_df
        random_numbers = np.random.uniform(
            regression_df["Low CI"], regression_df["High CI"]
        )

        # Create a new DataFrame with category values and random numbers
        coef_df = pd.DataFrame(
            {
                "rst_4_market_sector": regression_df["rst_4_market_sector"],
                "coef": random_numbers,
            }
        )

        npm_cut = store_weight_npm["npm_score"] >= 4
        high_npm = store_weight_npm[npm_cut].copy()
        low_npm = store_weight_npm[~npm_cut].copy()

        unique_products = pd.DataFrame(
            store_weight_npm[(store_weight_npm["npm_score"] >= 4)][
                "product_code"
            ].unique(),
            columns=["product_code"],
        )

        unique_products["indicator_reform"] = np.random.choice(
            [0, 1],
            size=len(unique_products),
            p=[1 - product_share_reform, product_share_reform],
        )

        high_npm = high_npm.merge(unique_products, on="product_code", how="left")
        high_npm["indicator_sale"] = 1
        high_npm["new_total_prod"] = np.where(
            high_npm["indicator_sale"] == 1,
            high_npm["total_prod"] * (1 - unhealthy_sales_change_values / 100),
            high_npm["total_prod"],
        )

        low_npm["indicator_reform"] = 0
        low_npm["indicator_sale"] = 1
        low_npm["new_total_prod"] = np.where(
            low_npm["indicator_sale"] == 1,
            low_npm["total_prod"] * (1 + healthy_sales_change_values / 100),
            low_npm["total_prod"],
        )

        randomised = pd.concat([high_npm, low_npm], ignore_index=True)
        randomised["new_total_kg"] = (
            randomised["new_total_prod"] * randomised["prod_weight_g"] / 1000
        )
        randomised["new_npm"] = randomised.pipe(
            su.apply_reduction_npm, npm_reduction_values
        )

        randomised = randomised.merge(
            prod_table[["rst_4_market_sector", "product_code"]],
            on="product_code",
        ).merge(coef_df, on="rst_4_market_sector", how="left")

        randomised["ed_pred"] = (
            randomised["ed"] - randomised["coef"] * npm_reduction_values
        )

        randomised["new_ed"] = np.where(
            randomised["indicator_reform"] == 1,
            randomised["ed_pred"],
            randomised["ed"],
        )
        randomised["new_kcal_tot"] = (
            randomised["new_ed"]
            / 100
            * randomised["prod_weight_g"]
            * randomised["new_total_prod"]
        )
        randomised["kcal_w_new"] = (
            randomised["new_kcal_tot"] / randomised["new_kcal_tot"].sum()
        )
        randomised["kg_w_new"] = (
            randomised["new_total_kg"] / randomised["new_total_kg"].sum()
        )

        mean_npm_kg_new = (randomised["kg_w_new"] * randomised["new_npm"]).sum()
        mean_npm_kcal_new = (randomised["kcal_w_new"] * randomised["new_npm"]).sum()

        mean_npm_kg_baseline = (randomised["kg_w"] * randomised["npm_score"]).sum()
        mean_npm_kcal_baseline = (randomised["kcal_w"] * randomised["npm_score"]).sum()

        kcal_pp_baseline = randomised["total_kcal"].sum() / gb_pop / 365
        kcal_pp_new = randomised["new_kcal_tot"].sum() / gb_pop / 365

        total_prod_baseline = randomised["total_prod"].sum()
        total_prod_new = randomised["new_total_prod"].sum()

        spend_baseline = (
            ((randomised["total_prod"] * randomised["spend"]).sum()) / gb_pop / 52
        )
        spend_new = (
            ((randomised["new_total_prod"] * randomised["spend"]).sum()) / gb_pop / 52
        )

        # Append the results to the list
        results.append(
            {
                "product_share_reform": product_share_reform,
                "mean_npm_kg_new": mean_npm_kg_new,
                "mean_npm_kcal_new": mean_npm_kcal_new,
                "mean_npm_kg_baseline": mean_npm_kg_baseline,
                "mean_npm_kcal_baseline": mean_npm_kcal_baseline,
                "kcal_pp_baseline": kcal_pp_baseline,
                "kcal_pp_new": kcal_pp_new,
                "total_prod_baseline": total_prod_baseline,
                "total_prod_new": total_prod_new,
                "spend_baseline": spend_baseline,
                "spend_new": spend_new,
                "gb_pop": gb_pop,
                "product_share_reform": product_share_reform,
            }
        )

    results_df = pd.DataFrame(results)

    # upload to S3
    upload_obj(
        results_df,
        BUCKET_NAME,
        "in_home/processed/targets/npm_robustness.csv",
        kwargs_writing={"index": False},
    )
