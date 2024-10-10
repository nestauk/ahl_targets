"""
This is a quick run of testing the predicted kcal per person per day reductions on the **adult population** only.
Approach:

- Merge on the proportion of a household's intake that is consumed by adults (from functions pulled from diets work) to the input data.
- Otherwise leave the inputs and script unchanged.

__________
Results:
New baseline: 1787.8
New adjusted: 1732.4
Diff: 55.3

The old baseline was 1629.2 calories per person per day.

Results v2:
This uses the input generated in `pipeline/model_data_diets.py` which should have the same assumptions as the diets work.
Baseline: 1549

How is it even lower! There seem to be some problems with the file, including a different set of column names coming out of the model_data.py script to those currently in the S3 bucket. Other issues are:
- The 'Period' and 'Purchase Date' columns do not match, so unsure when the date of purchases are

__________
Causes for potential concern:
- The new baseline is much lower than on diets (2393.3). This is surprising as it's not obvious where there would large differences between the purchase data.
- Have new data deliveries from Kantar meant that the Panel ID merge is no longer matching to the right people? (I don't think so)
- ...
"""

import pandas as pd
import numpy as np
from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME, PROJECT_DIR
from ahl_targets.utils import simulation_utils as su
from ahl_targets.getters import get_data
from ahl_targets.getters import simulated_outcomes as get_sim_data
import yaml


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

    # store_data = get_data.model_data()

    store_data = pd.read_csv(
        PROJECT_DIR / "outputs/model_data_unfiltered.csv"
    )  # The data with diets specs applied

    ########## v2 edit ##########
    from ahl_targets.utils import diets
    from ahl_targets.getters import get_data_v2 as g2

    # Get adult intake
    demog_df = g2.get_demographics_data()
    adult_intake = diets.adult_intake(demog_df)

    # Get total prop intake for each household
    adult_intake = adult_intake.groupby("Panel Id")["prop_intake"].sum().reset_index()

    # Merge adult intake to store_data
    store_data = store_data.merge(
        adult_intake[["Panel Id", "prop_intake"]], on="Panel Id", how="left"
    )

    ##Adjust kcal and volume values to reflect adult intake

    # Temp edit: Volume = volume up, is this right?
    store_data["volume_up"] = store_data["Volume"].astype(float)
    store_data["Energy KCal"] = store_data["Energy KCal"].astype(float)

    # Set original names to "old"
    store_data = store_data.rename(
        columns={"Energy KCal": "old_kcal", "volume_up": "old_volume_up"}
    )

    # Adjust kcal and volume by adult prop intake
    store_data["Energy KCal"] = store_data["old_kcal"] * store_data["prop_intake"]
    store_data["volume_up"] = store_data["old_volume_up"] * store_data["prop_intake"]

    #############################

    prod_table = get_data.product_metadata()

    store_weight_npm = su.weighted_npm(store_data)
    store_weight_npm["prod_weight_g"] = store_weight_npm.pipe(su.prod_weight_g)

    # Print the coefficients table
    coefficients_df = get_sim_data.coefficients_df()

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

    results_data_df = pd.concat(results_data, ignore_index=True)

    # upload to S3
    # upload_obj(
    #     results_df,
    #     BUCKET_NAME,
    #     "in_home/processed/targets/npm_agg.csv",
    #     kwargs_writing={"index": False},
    # )

    # Local save
    results_df.to_csv(PROJECT_DIR / "outputs/npm_sim_results.csv", index=False)
