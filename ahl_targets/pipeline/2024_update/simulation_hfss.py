"""
### 2024 Update ###

This is a copy of the original hfss simulation model (`ahl_targets/pipeline/simulation_hfss_legacy.py`) with the following adjustments:
- The input data has been updated based on updated model assumptions.
- A few additional edits to improve model performance and readability.

### FLAG ###
This produces sensible/similar results to as expected so for our purposes (just producing some figures) I think it's satisfactory for now. For more confidence a more detailed comparison of this output and the output of original hfss model output would be needed, as has been done for the update to the npm simulation.
"""

# Imports
import pandas as pd
import numpy as np
from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME, PROJECT_DIR
from ahl_targets.utils import simulation_utils as su
from ahl_targets.getters import get_data
import yaml
import logging
from ahl_targets.getters import get_data_v2 as g2


def simulation_hfss(
    store_weight_hfss: pd.DataFrame,
    num_iterations: list,
    product_share_reform_values_low: list,
    product_share_reform_values_medium: list,
    product_share_reform_values_high: list,
    hfss_high_sales_change_values: list,
    hfss_low_sales_change_values: list,
    product_share_sale_values: list,
    hfss_cutoff: list,
) -> pd.DataFrame:
    """
    This is mostly copied code from `pipeline/simulation_hfss_legacy.py`.
    """

    # Initialize results
    results = []
    results_data = []

    # Flag which products should be more impacted by the reformulation (the lower the NPM score (in the HFSS category), the higher % of products get reformulated, as they are closer to the HFSS threshold)
    unique_hfss_products_low = pd.DataFrame(
        store_weight_hfss[
            (store_weight_hfss["hfss"] == 1)
            & (store_weight_hfss["npm_score"] <= hfss_cutoff[0])
        ]["product_code"].unique(),
        columns=["product_code"],
    )
    unique_hfss_products_medium = pd.DataFrame(
        store_weight_hfss[
            (store_weight_hfss["hfss"] == 1)
            & (store_weight_hfss["npm_score"] > hfss_cutoff[0])
            & (store_weight_hfss["npm_score"] <= hfss_cutoff[1])
        ]["product_code"].unique(),
        columns=["product_code"],
    )
    unique_hfss_products_high = pd.DataFrame(
        store_weight_hfss[
            (store_weight_hfss["hfss"] == 1)
            & (store_weight_hfss["npm_score"] > hfss_cutoff[1])
            & (store_weight_hfss["npm_score"] <= hfss_cutoff[2])
        ]["product_code"].unique(),
        columns=["product_code"],
    )

    for product_share_reform_low in product_share_reform_values_low:
        for product_share_reform_medium in product_share_reform_values_medium:
            for product_share_reform_high in product_share_reform_values_high:
                for product_share_sale in product_share_sale_values:
                    for sales_change_high in hfss_high_sales_change_values:
                        for sales_change_low in hfss_low_sales_change_values:
                            for _ in range(num_iterations[0]):

                                # generate list of products to reformulate
                                unique_products_low = unique_hfss_products_low.copy()
                                unique_products_low["indicator_reform"] = (
                                    np.random.choice(
                                        [0, 1],
                                        size=len(unique_products_low),
                                        p=[
                                            1 - product_share_reform_low,
                                            product_share_reform_low,
                                        ],
                                    )
                                )

                                unique_products_medium = (
                                    unique_hfss_products_medium.copy()
                                )
                                unique_products_medium["indicator_reform"] = (
                                    np.random.choice(
                                        [0, 1],
                                        size=len(unique_products_medium),
                                        p=[
                                            1 - product_share_reform_medium,
                                            product_share_reform_medium,
                                        ],
                                    )
                                )

                                unique_products_high = unique_hfss_products_high.copy()
                                unique_products_high["indicator_reform"] = (
                                    np.random.choice(
                                        [0, 1],
                                        size=len(unique_products_high),
                                        p=[
                                            1 - product_share_reform_high,
                                            product_share_reform_high,
                                        ],
                                    )
                                )

                                unique_products = pd.concat(
                                    [
                                        unique_products_low,
                                        unique_products_medium,
                                        unique_products_high,
                                    ]
                                )

                                # Merge the unique_products with the store_weight_hfss to get the full dataset
                                data_all = store_weight_hfss.merge(
                                    unique_products, on="product_code", how="left"
                                )

                                # split data into hfss and non-hfss where a product is considered non HFSS if it was HFSS in the original data but it has been reformulated
                                npm_cut = (data_all["hfss"] == 1) & (
                                    data_all["indicator_reform"] == 0
                                )
                                hfss = data_all[npm_cut].copy()
                                non_hfss = data_all[~npm_cut].copy()

                                hfss["indicator_sale"] = np.random.choice(
                                    [0, 1],
                                    size=len(hfss),
                                    p=[1 - product_share_sale, product_share_sale],
                                )
                                hfss["new_total_prod"] = np.where(
                                    hfss["indicator_sale"] == 1,
                                    hfss["total_prod"] * (1 - sales_change_high / 100),
                                    hfss["total_prod"],
                                )
                                hfss["new_npm"] = hfss["npm_score"]
                                hfss["new_hfss"] = hfss["hfss"]

                                non_hfss["indicator_sale"] = np.random.choice(
                                    [0, 1],
                                    size=len(non_hfss),
                                    p=[1 - product_share_sale, product_share_sale],
                                )
                                non_hfss["new_total_prod"] = np.where(
                                    non_hfss["indicator_sale"] == 1,
                                    non_hfss["total_prod"]
                                    * (1 + sales_change_low / 100),
                                    non_hfss["total_prod"],
                                )
                                non_hfss["new_npm"] = np.where(
                                    non_hfss["indicator_reform"] == 1,
                                    3,
                                    non_hfss["npm_score"],
                                )
                                non_hfss["new_hfss"] = np.where(
                                    non_hfss["indicator_reform"] == 1,
                                    0,
                                    non_hfss["hfss"],
                                )

                                randomised = pd.concat(
                                    [hfss, non_hfss], ignore_index=True
                                )

                                randomised["new_total_kg"] = (
                                    randomised["new_total_prod"]
                                    * randomised["prod_weight_g"]
                                    / 1000
                                )

                                randomised = randomised.merge(
                                    prod_table[["rst_4_market_sector", "product_code"]],
                                    on="product_code",
                                ).merge(
                                    coefficients_df,
                                    on="rst_4_market_sector",
                                    how="left",
                                )

                                randomised["ed_pred"] = np.where(
                                    randomised["indicator_reform"] == 1,
                                    randomised["ed"]
                                    - randomised["Coefficient"]
                                    * (randomised["npm_score"] - 3),
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

                                total_prod = randomised["new_total_prod"].sum()

                                hfss_baseline = (
                                    randomised["kg_w"] * randomised["hfss"]
                                ).sum()
                                hfss_kcal_baseline = (
                                    randomised["kcal_w"] * randomised["hfss"]
                                ).sum()
                                total_prod_baseline = randomised["total_prod"].sum()

                                store_name = data_all["store_cat"].unique()

                                total_kcal_baseline = randomised["total_kcal"].sum()
                                total_kcal_new = randomised["new_kcal_tot"].sum()

                                total_prod_baseline = randomised["total_prod"].sum()
                                total_prod_new = randomised["new_total_prod"].sum()

                                total_kg_baseline = randomised["total_kg"].sum()
                                total_kg_new = randomised["new_total_kg"].sum()

                                hfss_prod_baseline = (
                                    randomised["total_prod"] * randomised["hfss"]
                                ).sum()
                                hfss_prod_new = (
                                    randomised["new_total_prod"]
                                    * randomised["new_hfss"]
                                ).sum()

                                hfss_kg_baseline = (
                                    randomised["total_kg"] * randomised["hfss"]
                                ).sum()
                                hfss_kg_new = (
                                    randomised["new_total_kg"] * randomised["new_hfss"]
                                ).sum()

                                hfss_kcal_baseline = (
                                    randomised["total_kcal"] * randomised["hfss"]
                                ).sum()
                                hfss_kcal_new = (
                                    randomised["new_kcal_tot"] * randomised["new_hfss"]
                                ).sum()

                                kcal_pp_baseline = (
                                    randomised["total_kcal"].sum() / 51718632 / 365
                                )
                                kcal_pp_new = (
                                    randomised["new_kcal_tot"].sum() / 51718632 / 365
                                )

                                total_spend_baseline = (
                                    randomised["total_prod"] * randomised["spend"]
                                ).sum()
                                total_spend_new = (
                                    randomised["new_total_prod"] * randomised["spend"]
                                ).sum()

                                randomised["kg_w_new"] = (
                                    randomised["new_total_kg"]
                                    / randomised["new_total_kg"].sum()
                                )

                                randomised["iteration"] = _

                                # Append the results to the list
                                results.append(
                                    {
                                        "product_share_reform_low": product_share_reform_low,
                                        "product_share_reform_medium": product_share_reform_medium,
                                        "product_share_reform_high": product_share_reform_high,
                                        "product_share_sale": product_share_sale,
                                        "sales_change_high": sales_change_high,
                                        "sales_change_low": sales_change_low,
                                        "total_prod_baseline": total_prod_baseline,
                                        "total_prod_new": total_prod_new,
                                        "total_kg_baseline": total_kg_baseline,
                                        "total_kg_new": total_kg_new,
                                        "hfss_prod_baseline": hfss_prod_baseline,
                                        "hfss_prod_new": hfss_prod_new,
                                        "hfss_kg_baseline": hfss_kg_baseline,
                                        "hfss_kg_new": hfss_kg_new,
                                        "total_kcal_baseline": total_kcal_baseline,
                                        "total_kcal_new": total_kcal_new,
                                        "hfss_kcal_baseline": hfss_kcal_baseline,
                                        "hfss_kcal_new": hfss_kcal_new,
                                        "total_spend_baseline": total_spend_baseline,
                                        "total_spend_new": total_spend_new,
                                        "kcal_pp_baseline": kcal_pp_baseline,
                                        "kcal_pp_new": kcal_pp_new,
                                    }
                                )

                                results_data.append(
                                    randomised.assign(
                                        product_share_reform_low=product_share_reform_low,
                                        product_share_reform_medium=product_share_reform_medium,
                                        product_share_reform_high=product_share_reform_high,
                                        product_share_sale=product_share_sale,
                                        sales_change_high=sales_change_high,
                                        sales_change_low=sales_change_low,
                                    )
                                )

    # Create the DataFrame from the list of results
    results_df = pd.DataFrame(results)
    results_data_df = pd.concat(results_data, ignore_index=True)

    return results_df, results_data_df


if __name__ == "__main__":

    # Load params
    with open(
        f"{PROJECT_DIR}/ahl_targets/config/hfss_model.yaml",
        "r",
    ) as f:
        modeling_params = yaml.safe_load(f)

    # # Set params to variables
    # num_iterations = modeling_params["num_iterations"]
    # product_share_reform_values_low = modeling_params["product_share_reform_values_low"]
    # product_share_reform_values_medium = modeling_params["product_share_reform_values_medium"]
    # product_share_reform_values_high = modeling_params["product_share_reform_values_high"]
    # hfss_high_sales_change_values = modeling_params["hfss_high_sales_change_values"]
    # hfss_low_sales_change_values = modeling_params["hfss_low_sales_change_values"]
    # product_share_sale_values = modeling_params["product_share_sale_values"]
    # hfss_cutoff = modeling_params["hfss_cutoff"]

    # TEMP EDIT: SHARE ACTION PARAMS (reformulation only)
    num_iterations = [100]
    product_share_reform_values_low = [1]
    product_share_reform_values_medium = [0]
    product_share_reform_values_high = [0]
    hfss_high_sales_change_values = [0]  # Set sales shifts to 0
    hfss_low_sales_change_values = [0]
    product_share_sale_values = [1]
    hfss_cutoff = [12, 15, 20]

    # Read in data
    # Read in main dataframe
    store_weight = g2.get_agg_data_2024()

    # Add hfss or not
    store_weight_hfss = store_weight.copy()
    store_weight_hfss["hfss"] = store_weight_hfss["npm_score"] >= 4

    logging.info(
        "kcal pp baseline: {}".format(
            store_weight_hfss["total_kcal"].sum() / 51718632 / 365
        )
    )
    logging.info(
        "% HFSS baseline: {}".format(
            (store_weight_hfss["hfss"] * store_weight_hfss["kg_w"]).sum() * 100
        )
    )

    # Read in product metadata
    prod_table = get_data.product_metadata()

    # Get coefficients for relationship between NPM and ED
    coefficients_df = g2.coefficients_2024()

    # Run simulation
    results_df, results_data_df = simulation_hfss(
        store_weight_hfss,
        num_iterations,
        product_share_reform_values_low,
        product_share_reform_values_medium,
        product_share_reform_values_high,
        hfss_high_sales_change_values,
        hfss_low_sales_change_values,
        product_share_sale_values,
        hfss_cutoff,
    )
    kcal_diff = (
        (results_df["kcal_pp_baseline"] - results_df["kcal_pp_new"]).mean().round(2)
    )

    # Print new kcal pp baseline
    logging.info("Kcal pp new: {}".format(results_df["kcal_pp_new"].mean()))
    logging.info(
        "hfss % new: {}".format(
            results_df["hfss_kg_new"].sum() / results_df["total_kg_new"].sum() * 100
        )
    )

    logging.info(
        "Difference in kcal pp: {}".format(
            (results_df["kcal_pp_baseline"] - results_df["kcal_pp_new"]).mean()
        )
    )

    # Extra analysis needed for blog:
    # 1. How many products are reformulated?
    reformulated_products = (
        results_data_df[results_data_df["indicator_reform"] == 1]
        .groupby("iteration")
        .product_code.nunique()
    )  # Average number of reformulated products
    total_products = results_data_df.groupby(
        "iteration"
    ).product_code.nunique()  # Average number of total products (avg unnecessary but just to be consistent)
    reformulated_percentage = (reformulated_products / total_products).mean()

    logging.info(
        "Prct of products that are reformulated: {}".format(reformulated_percentage)
    )  # ~21% of products are reformulated on average.
    # N.B. This feels slightly weird (calculating the % of _unique products_ that are reformulated), but that's what's refered to in the product_share_reform_values_x parameters.

    # 2. What proportion of products with a (converted) NPM score of (70) 0 or (below) above are reformulated?
    total_products_below_70 = (
        results_data_df[results_data_df["npm_score"] >= 0]
        .groupby("iteration")
        .product_code.nunique()
    )
    reformulated_percentage_below_70 = (
        reformulated_products / total_products_below_70
    ).mean()  # ~30% of products with a NPM score of 0 or below are reformulated on average.

    # 3. What is the avg difference in NPM for reformulated products?
    avg_npm_diff_reformulated = (
        results_data_df[results_data_df["indicator_reform"] == 1]
        .groupby("iteration")
        .apply(lambda x: (x["new_npm"] - x["npm_score"]).mean())
        .mean()
    )

    logging.info(
        "Avg difference in converted NPM for reformulated products: {}".format(
            -2 * avg_npm_diff_reformulated
        )
    )  # ~11 absolute NPM difference

    save_prompt = input(
        "Would you like to save and overwrite the existing model on S3? (y/n)"
    )

    if save_prompt == "y":

        logging.info("Saving the model output to S3")

        upload_obj(
            results_df,
            BUCKET_NAME,
            f"in_home/processed/targets/share_action_custom_plots/feb/hfss_model_results_{kcal_diff}.csv",
            kwargs_writing={"index": False},
        )

        # Only run this step if needed - takes about 30 minutes
        upload_obj(
            results_data_df,
            BUCKET_NAME,
            f"in_home/processed/targets/share_action_custom_plots/feb/hfss_model_results_detailed_data_{kcal_diff}.parquet",
            kwargs_writing={"compression": "zstd", "engine": "pyarrow"},
        )
