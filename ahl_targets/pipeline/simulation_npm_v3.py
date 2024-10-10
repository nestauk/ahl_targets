"""
This is a copy of the original retailer targets script. The difference is that the input data is now built here in `create_diets_input.py` with files taken from the diets repository.
"""

import pandas as pd
import numpy as np
from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME, PROJECT_DIR
from ahl_targets.utils import simulation_utils as su
from ahl_targets.getters import get_data
from ahl_targets.getters import simulated_outcomes as get_sim_data
import yaml
import logging

from ahl_targets.utils import diets
from ahl_targets.getters import get_data_v2 as g2


if __name__ == "__main__":
    with open(
        f"{PROJECT_DIR}/ahl_targets/config/npm_model.yaml",
        "r",
    ) as f:
        modeling_params = yaml.safe_load(f)

    num_iterations = modeling_params["num_iterations"]
    product_share_reform_values = modeling_params["product_share_reform_values"]
    product_share_sales_values = modeling_params["product_share_sales_values"]
    npm_reduction_values = modeling_params["npm_decrease_values"]
    unhealthy_sales_change_values = modeling_params["unhealthy_sales_change_values"]
    healthy_sales_change_values = modeling_params["healthy_sales_change_values"]

    # set seed for reproducibility

    np.random.seed(42)

    # read data

    ##########v3 changes##########
    # Build the new model file
    # df = pd.read_csv(PROJECT_DIR / "inputs/processed/pur_nut_prod_food.csv")
    df = g2.new_model_data()

    # Merge on NPM score
    npm = get_data.full_npm()

    df_npm = df.merge(
        npm[["purchase_id", "period", "npm_score", "kcal_per_100g"]],
        on=["purchase_id", "period"],
        how="left",
    )

    # Rename variables the equivalent in the old model
    df_npm = df_npm.rename(
        columns={
            "panel_id": "Panel Id",
            "gross_up_weight": "Gross Up Weight",
            "volume": "volume_up",
            "store_level_3": "store_cat",
            "energy_kcal": "Energy KCal",
            "quantity": "Quantity",
            "spend": "Spend",
        }
    )

    # #Save df_npm locally
    # df_npm.to_csv(PROJECT_DIR / "outputs/df_npm.csv", index=False)

    ##############################

    ########## v2 edit ##########
    # Get adult intake
    demog_df = g2.get_demographics_data()
    adult_intake = diets.adult_intake(demog_df)

    # Get total prop intake for each household
    adult_intake = adult_intake.groupby("Panel Id")["prop_intake"].sum().reset_index()

    # Merge adult intake to store_data
    store_data = df_npm.merge(
        adult_intake[["Panel Id", "prop_intake"]], on="Panel Id", how="left"
    )

    ##Adjust kcal and volume values to reflect adult intake

    # Set original names to "old"
    store_data = store_data.rename(
        columns={"Energy KCal": "old_kcal", "volume_up": "old_volume_up"}
    )

    # Adjust kcal and volume by adult prop intake
    store_data["Energy KCal"] = store_data["old_kcal"] * store_data["prop_intake"]
    store_data["volume_up"] = store_data["old_volume_up"] * store_data["prop_intake"]

    #############################

    ##v3 edit##
    # Add some common sense exclusions: volume <20kg /<10L
    store_data = store_data[
        (store_data["volume_up"] <= 10) & (store_data["reported_volume"] == "Litres")
        | (store_data["volume_up"] <= 20)
    ]

    # store_data = get_data.model_data()
    prod_table = get_data.product_metadata()

    store_weight_npm = su.weighted_npm(store_data)
    store_weight_npm["prod_weight_g"] = store_weight_npm.pipe(su.prod_weight_g)

    # Print the coefficients table
    # coefficients_df = get_sim_data.coefficients_df()
    coefficients_df = g2.new_coefficients()

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
                            npm_cut = store_weight_npm["npm_score"] >= 4
                            high_npm = store_weight_npm[npm_cut].copy()
                            low_npm = store_weight_npm[~npm_cut].copy()

                            # Note there are a few products with NaN npm values that will be assigned to low_npm.

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

    # Create the DataFrame from the list of results
    results_df = pd.DataFrame(results)

    # Print kcal pp baseline
    logging.info("Kcal pp baseline: {}".format(results_df["kcal_pp_baseline"].mean()))
    logging.info(
        "Difference in kcal pp: {}".format(
            (results_df["kcal_pp_baseline"] - results_df["kcal_pp_new"]).mean()
        )
    )

    # results_data_df = pd.concat(results_data, ignore_index=True)

    # #Local save
    # results_df.to_csv(PROJECT_DIR / "outputs/npm_agg_v3.csv", index=False)

    # # upload to S3
    # upload_obj(
    #     results_df,
    #     BUCKET_NAME,
    #     "in_home/processed/targets/npm_agg_v3.csv",
    #     kwargs_writing={"index": False},
    # )
