import pandas as pd
import numpy as np
from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME
from ahl_targets.utils import simulation_utils as su
from ahl_targets.getters import get_data
from ahl_targets.pipeline import (
    num_iterations,
    product_share_reform_values,
    product_share_sale_values,
    ed_reduction_values,
    ed_high_sales_change_values,
    ed_low_sales_change_values,
    ed_cutoff,
)


if __name__ == "__main__":
    # set seed for reproducibility

    np.random.seed(42)

    # read data

    store_data = get_data.model_data().compute()
    prod_table = get_data.product_metadata()

    # generate aggregate data

    store_weight = su.weighted_ed(store_data)
    store_weight["prod_weight_g"] = store_weight.pipe(su.prod_weight_g)

    # initialise empty list

    results = []
    results_data = []

    # Nested loop to iterate through different values of product_share and ed_reduction
    for product_share_reform in product_share_reform_values:
        for product_share_sale in product_share_sale_values:
            for ed_reduction in ed_reduction_values:
                for sales_change_high in ed_high_sales_change_values:
                    for sales_change_low in ed_low_sales_change_values:
                        # Repeat the code num_iterations times
                        for _ in range(num_iterations):
                            # split into high and low ed
                            ed_cut = store_weight["ed"] >= ed_cutoff
                            high_ed = store_weight[ed_cut].copy()
                            low_ed = store_weight[~ed_cut].copy()

                            unique_products = pd.DataFrame(
                                store_weight[(store_weight["ed"] >= ed_cutoff)][
                                    "product_code"
                                ].unique(),
                                columns=["product_code"],
                            )
                            unique_products["indicator_reform"] = np.random.choice(
                                [0, 1],
                                size=len(unique_products),
                                p=[1 - product_share_reform, product_share_reform],
                            )

                            # calculation in high ed products
                            high_ed = high_ed.merge(
                                unique_products, on="product_code", how="left"
                            )
                            high_ed["indicator_sale"] = np.random.choice(
                                [0, 1],
                                size=len(high_ed),
                                p=[1 - product_share_sale, product_share_sale],
                            )
                            high_ed["new_total_prod"] = np.where(
                                high_ed["indicator_sale"] == 1,
                                high_ed["total_prod"] * (1 - sales_change_high / 100),
                                high_ed["total_prod"],
                            )

                            # calculations in low ed products
                            low_ed["indicator_reform"] = 0
                            low_ed["indicator_sale"] = np.random.choice(
                                [0, 1],
                                size=len(low_ed),
                                p=[1 - product_share_sale, product_share_sale],
                            )
                            low_ed["new_total_prod"] = np.where(
                                low_ed["indicator_sale"] == 1,
                                low_ed["total_prod"] * (1 + sales_change_low / 100),
                                low_ed["total_prod"],
                            )

                            randomised = pd.concat([high_ed, low_ed], ignore_index=True)
                            randomised["new_total_kg"] = (
                                randomised["new_total_prod"]
                                * randomised["prod_weight_g"]
                                / 1000
                            )
                            randomised["new_ed"] = randomised.pipe(
                                su.apply_reduction_ed, ed_reduction
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

                            mean_ed_kg_new = (
                                randomised["kg_w_new"] * randomised["new_ed"]
                            ).sum()
                            mean_ed_kcal_new = (
                                randomised["kcal_w_new"] * randomised["new_ed"]
                            ).sum()

                            mean_ed_kg_baseline = (
                                randomised["kg_w"] * randomised["ed"]
                            ).sum()
                            mean_ed_kcal_baseline = (
                                randomised["kcal_w"] * randomised["ed"]
                            ).sum()

                            kcal_pp_baseline = (
                                randomised["total_kcal"].sum() / 66000000 / 365
                            )
                            kcal_pp_new = (
                                randomised["new_kcal_tot"].sum() / 66000000 / 365
                            )

                            total_prod_baseline = randomised["total_prod"].sum()
                            total_prod_new = randomised["new_total_prod"].sum()

                            kcal_high_baseline = randomised[
                                randomised["ed"] >= ed_cutoff
                            ]["total_kcal"].sum()
                            kcal_high_new = randomised[
                                randomised["new_ed"] >= ed_cutoff
                            ]["new_kcal_tot"].sum()

                            kcal_low_baseline = randomised[
                                randomised["ed"] < ed_cutoff
                            ]["total_kcal"].sum()
                            kcal_low_new = randomised[randomised["new_ed"] < ed_cutoff][
                                "new_kcal_tot"
                            ].sum()

                            spend_baseline = (
                                ((randomised["total_prod"] * randomised["spend"]).sum())
                                / 66000000
                                / 52
                            )
                            spend_new = (
                                (
                                    (
                                        randomised["new_total_prod"]
                                        * randomised["spend"]
                                    ).sum()
                                )
                                / 66000000
                                / 52
                            )

                            iteration = _

                            # Append the results to the list
                            results.append(
                                {
                                    "product_share_reform": product_share_reform,
                                    "product_share_sale": product_share_sale,
                                    "sales_change_high": sales_change_high,
                                    "sales_change_low": sales_change_low,
                                    "ed_reduction": ed_reduction,
                                    "mean_ed_kg_new": mean_ed_kg_new,
                                    "mean_ed_kcal_new": mean_ed_kcal_new,
                                    "mean_ed_kg_baseline": mean_ed_kg_baseline,
                                    "mean_ed_kcal_baseline": mean_ed_kcal_baseline,
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
                                    ed_reduction=ed_reduction,
                                    sales_change_high=sales_change_high,
                                    sales_change_low=sales_change_low,
                                    iteration=iteration,
                                )
                            )

    # Create the DataFrame from the list of results
    results_df = pd.DataFrame(results)

    results_data_df = pd.concat(results_data, ignore_index=True)

    # upload to S3
    upload_obj(
        results_df,
        BUCKET_NAME,
        "in_home/data_outputs/targets_annex/ed_agg.csv",
        kwargs_writing={"index": False},
    )

    # upload_obj(
    #     results_data_df,
    #     BUCKET_NAME,
    #     "in_home/data_outputs/targets_annex/ed_full.csv",
    #     kwargs_writing={"index": False},
    # )
