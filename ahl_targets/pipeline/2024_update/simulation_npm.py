"""
### 2024 Update ###

This is a copy of the original npm simulation model (`ahl_targets/pipeline/simulation_npm_legacy.py`) with the following adjustments:
- The input data has been updated based on updated model assumptions.
- A few additional edits to improve model performance and readability.

See full details of the 2024 update to the npm simulation model in `ahl_targets/pipeline/2024_update/README.md`.

"""

import pandas as pd
import numpy as np
from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME, PROJECT_DIR
from ahl_targets.utils import simulation_utils as su
from ahl_targets.getters import get_data
import yaml
import logging
from ahl_targets.getters import get_data_v2 as g2


def simulation_npm(
    store_weight_npm: pd.DataFrame,
    num_iterations: list,
    product_share_reform_values: list,
    product_share_sales_values: list,
    npm_reduction_values: list,
    unhealthy_sales_change_values: list,
    healthy_sales_change_values: list,
    coefficients_df: pd.DataFrame,
    prod_table: pd.DataFrame,
) -> pd.DataFrame:

    results = []
    results_data = []

    # Nested loop to iterate through different values of product_share and ed_reduction
    for product_share_reform in product_share_reform_values:
        for product_share_sale in product_share_sales_values:
            for npm_reduction in npm_reduction_values:
                for sales_change_high in unhealthy_sales_change_values:
                    for sales_change_low in healthy_sales_change_values:
                        # Repeat the code num_iterations times
                        for _ in range(num_iterations[0]):
                            npm_cut = store_weight_npm["npm_score"] >= 0
                            high_npm = store_weight_npm[npm_cut].copy()
                            low_npm = store_weight_npm[~npm_cut].copy()

                            # Note there are a few products with NaN npm values that will be assigned to low_npm.

                            unique_products = pd.DataFrame(
                                store_weight_npm[(store_weight_npm["npm_score"] >= 0)][
                                    "product_code"
                                ].unique(),
                                columns=["product_code"],
                            )

                            unique_products["indicator_reform"] = np.random.choice(
                                [0, 1],
                                size=len(unique_products),
                                p=[1 - product_share_reform, product_share_reform],
                            )

                            high_npm = high_npm.merge(
                                unique_products, on="product_code", how="left"
                            )
                            high_npm["indicator_sale"] = np.random.choice(
                                [0, 1],
                                size=len(high_npm),
                                p=[1 - product_share_sale, product_share_sale],
                            )
                            high_npm["new_total_prod"] = np.where(
                                high_npm["indicator_sale"] == 1,
                                high_npm["total_prod"] * (1 - sales_change_high / 100),
                                high_npm["total_prod"],
                            )

                            low_npm["indicator_reform"] = 0
                            low_npm["indicator_sale"] = np.random.choice(
                                [0, 1],
                                size=len(low_npm),
                                p=[1 - product_share_sale, product_share_sale],
                            )
                            low_npm["new_total_prod"] = np.where(
                                low_npm["indicator_sale"] == 1,
                                low_npm["total_prod"] * (1 + sales_change_low / 100),
                                low_npm["total_prod"],
                            )

                            randomised = pd.concat(
                                [high_npm, low_npm], ignore_index=True
                            )
                            randomised["new_total_kg"] = (
                                randomised["new_total_prod"]
                                * randomised["prod_weight_g"]
                                / 1000
                            )
                            randomised["new_npm"] = randomised.pipe(
                                su.apply_reduction_npm, npm_reduction
                            )

                            randomised = randomised.merge(
                                prod_table[["rst_4_market_sector", "product_code"]],
                                on="product_code",
                            ).merge(
                                coefficients_df, on="rst_4_market_sector", how="left"
                            )

                            randomised["ed_pred"] = np.where(
                                npm_reduction > 0,
                                randomised["ed"]
                                - randomised["Coefficient"] * npm_reduction,
                                randomised["ed"],
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
                                randomised["new_kcal_tot"]
                                / randomised["new_kcal_tot"].sum()
                            )
                            randomised["kg_w_new"] = (
                                randomised["new_total_kg"]
                                / randomised["new_total_kg"].sum()
                            )

                            mean_npm_kg_new = (
                                randomised["kg_w_new"] * randomised["new_npm"]
                            ).sum()
                            mean_npm_kcal_new = (
                                randomised["kcal_w_new"] * randomised["new_npm"]
                            ).sum()

                            mean_npm_kg_baseline = (
                                randomised["kg_w"] * randomised["npm_score"]
                            ).sum()

                            mean_npm_kcal_baseline = (
                                randomised["kcal_w"] * randomised["npm_score"]
                            ).sum()

                            kcal_pp_baseline = (
                                randomised["total_kcal"].sum() / 51718632 / 365
                            )
                            kcal_pp_new = (
                                randomised["new_kcal_tot"].sum() / 51718632 / 365
                            )

                            total_prod_baseline = randomised["total_prod"].sum()
                            total_prod_new = randomised["new_total_prod"].sum()

                            spend_baseline = (
                                ((randomised["total_prod"] * randomised["spend"]).sum())
                                / 51718632
                                / 52
                            )
                            spend_new = (
                                (
                                    (
                                        randomised["new_total_prod"]
                                        * randomised["spend"]
                                    ).sum()
                                )
                                / 51718632
                                / 52
                            )

                            randomised["iteration"] = _

                            # Append the results to the list
                            results.append(
                                {
                                    "product_share_reform": product_share_reform,
                                    "product_share_sale": product_share_sale,
                                    "sales_change_high": sales_change_high,
                                    "sales_change_low": sales_change_low,
                                    "npm_reduction": npm_reduction,
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
                                }
                            )

                            results_data.append(
                                randomised.assign(
                                    product_share_reform=product_share_reform,
                                    product_share_sale=product_share_sale,
                                    npm_reduction=npm_reduction,
                                    sales_change_high=sales_change_high,
                                    sales_change_low=sales_change_low,
                                )
                            )

    results_df = pd.DataFrame(results)  # Aggregate results for each iteration
    results_data_df = pd.concat(results_data, ignore_index=True)  # Detailed results

    return results_df, results_data_df


if __name__ == "__main__":
    # with open(
    #     f"{PROJECT_DIR}/ahl_targets/config/npm_model.yaml",
    #     "r",
    # ) as f:
    #     modeling_params = yaml.safe_load(f)

    # num_iterations = modeling_params["num_iterations"]
    # product_share_reform_values = modeling_params["product_share_reform_values"]
    # product_share_sales_values = modeling_params["product_share_sales_values"]
    # npm_reduction_values = modeling_params["npm_decrease_values"]
    # unhealthy_sales_change_values = modeling_params["unhealthy_sales_change_values"]
    # healthy_sales_change_values = modeling_params["healthy_sales_change_values"]

    # TEMP EDIT: SHARE ACTION PARAMS
    num_iterations = [500]
    product_share_reform_values = [0.5]
    product_share_sales_values = [1]
    npm_reduction_values = [2]
    unhealthy_sales_change_values = [0]
    healthy_sales_change_values = [0]

    # set seed for reproducibility

    # np.random.seed(42)

    # Read in main dataframe

    store_weight_npm = g2.get_agg_data_2024()

    logging.info(
        "kcal pp baseline: {}".format(
            store_weight_npm["total_kcal"].sum() / 51718632 / 365
        )
    )
    logging.info(
        "NPM SWA baseline: {}".format(
            (store_weight_npm["npm_score"] * store_weight_npm["kg_w"]).sum()
        )
    )

    # Read in product metadata
    prod_table = get_data.product_metadata()

    # Get coefficients for relationship between NPM and ED
    coefficients_df = g2.coefficients_2024()

    # Run simulation
    results_df, results_data_df = simulation_npm(
        store_weight_npm,
        num_iterations,
        product_share_reform_values,
        product_share_sales_values,
        npm_reduction_values,
        unhealthy_sales_change_values,
        healthy_sales_change_values,
        coefficients_df,
        prod_table,
    )

    kcal_diff = (
        (results_df["kcal_pp_baseline"] - results_df["kcal_pp_new"]).mean().round(2)
    )

    # Print new kcal pp baseline
    logging.info("Kcal pp new: {}".format(results_df["kcal_pp_new"].mean()))
    logging.info("NPM SWA new: {}".format(results_df["mean_npm_kg_new"].mean()))

    logging.info(
        "Difference in kcal pp: {}".format(
            (results_df["kcal_pp_baseline"] - results_df["kcal_pp_new"]).mean()
        )
    )

    save_prompt = input(
        "Would you like to save and overwrite the existing model on S3? (y/n)"
    )

    if save_prompt == "y":

        logging.info("Saving the model output to S3")

        upload_obj(
            results_df,
            BUCKET_NAME,
            f"in_home/processed/targets/share_action_custom_plots/model_results_{kcal_diff}.csv",
            kwargs_writing={"index": False},
        )

        # Only run this step if needed - takes about 30 minutes
        upload_obj(
            results_data_df,
            BUCKET_NAME,
            f"in_home/processed/targets/share_action_custom_plots/model_results_detailed_data_{kcal_diff}.parquet",
            kwargs_writing={"compression": "zstd", "engine": "pyarrow"},
        )
