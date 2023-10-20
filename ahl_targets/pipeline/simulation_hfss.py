import pandas as pd
import numpy as np
from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME
from ahl_targets.utils import simulation_utils as su
from ahl_targets.getters import get_data
from ahl_targets.getters import simulated_outcomes as get_sim_data
from ahl_targets import PROJECT_DIR
import yaml
from ahl_targets.pipeline import product_transformation as pt


if __name__ == "__main__":
    # set seed for reproducibility

    np.random.seed(42)

    with open(
        f"{PROJECT_DIR}/ahl_targets/config/hfss_model.yaml",
        "r",
    ) as f:
        modeling_params = yaml.safe_load(f)

    num_iterations = modeling_params["num_iterations"]
    product_share_reform_values_low = modeling_params["product_share_reform_values_low"]
    product_share_reform_values_medium = modeling_params[
        "product_share_reform_values_medium"
    ]
    product_share_reform_values_high = modeling_params[
        "product_share_reform_values_high"
    ]
    hfss_high_sales_change_values = modeling_params["hfss_high_sales_change_values"]
    hfss_low_sales_change_values = modeling_params["hfss_low_sales_change_values"]
    hfss_cutoff = modeling_params["hfss_cutoff"]
    product_share_sale_values = modeling_params["product_share_sale_values"]

    # read data

    store_data = get_data.model_data()
    prod_table = get_data.product_metadata()
    coefficients_df = get_sim_data.coefficients_df()

    # round NPM score
    store_data["npm_score"] = round(store_data["npm_score"], 0)

    # hfss indicator
    store_data = store_data.pipe(pt.type).pipe(pt.is_food).pipe(pt.in_scope)

    # generate aggregate data

    store_weight_hfss = su.weighted_hfss(store_data)
    store_weight_hfss["prod_weight_g"] = store_weight_hfss.pipe(su.prod_weight_g)

    # create list of stores

    store_list = []

    grouped_data = store_weight_hfss.groupby("store_cat")

    for group_name, group_df in grouped_data:
        # Create a copy of the group dataframe and append it to the list of datasets
        store_list.append(group_df.copy())

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

    results = []
    results_data = []

    for product_share_reform_low in product_share_reform_values_low:
        for product_share_reform_medium in product_share_reform_values_medium:
            for product_share_reform_high in product_share_reform_values_high:
                for product_share_sale in product_share_sale_values:
                    for sales_change_high in hfss_high_sales_change_values:
                        for sales_change_low in hfss_low_sales_change_values:
                            for _ in range(30):
                                # generate list of products to reformulate
                                unique_products_low = unique_hfss_products_low.copy()
                                unique_products_low[
                                    "indicator_reform"
                                ] = np.random.choice(
                                    [0, 1],
                                    size=len(unique_products_low),
                                    p=[
                                        1 - product_share_reform_low,
                                        product_share_reform_low,
                                    ],
                                )

                                unique_products_medium = (
                                    unique_hfss_products_medium.copy()
                                )
                                unique_products_medium[
                                    "indicator_reform"
                                ] = np.random.choice(
                                    [0, 1],
                                    size=len(unique_products_medium),
                                    p=[
                                        1 - product_share_reform_medium,
                                        product_share_reform_medium,
                                    ],
                                )

                                unique_products_high = unique_hfss_products_high.copy()
                                unique_products_high[
                                    "indicator_reform"
                                ] = np.random.choice(
                                    [0, 1],
                                    size=len(unique_products_high),
                                    p=[
                                        1 - product_share_reform_high,
                                        product_share_reform_high,
                                    ],
                                )

                                unique_products = pd.concat(
                                    [
                                        unique_products_low,
                                        unique_products_medium,
                                        unique_products_high,
                                    ]
                                )

                                for dataset in store_list:
                                    # merge store data with list of unique products
                                    data_all = dataset.merge(
                                        unique_products, on="product_code", how="left"
                                    ).fillna(0)

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
                                        hfss["total_prod"]
                                        * (1 - sales_change_high / 100),
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
                                        prod_table[
                                            ["rst_4_market_sector", "product_code"]
                                        ],
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

                                    store_name = dataset["store_cat"].unique()

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
                                        randomised["new_total_kg"]
                                        * randomised["new_hfss"]
                                    ).sum()

                                    hfss_kcal_baseline = (
                                        randomised["total_kcal"] * randomised["hfss"]
                                    ).sum()
                                    hfss_kcal_new = (
                                        randomised["new_kcal_tot"]
                                        * randomised["new_hfss"]
                                    ).sum()

                                    kcal_pp_baseline = (
                                        randomised["total_kcal"].sum() / 66000000 / 365
                                    )
                                    kcal_pp_new = (
                                        randomised["new_kcal_tot"].sum()
                                        / 66000000
                                        / 365
                                    )

                                    total_spend_baseline = (
                                        randomised["total_prod"] * randomised["spend"]
                                    ).sum()
                                    total_spend_new = (
                                        randomised["new_total_prod"]
                                        * randomised["spend"]
                                    ).sum()

                                    randomised["iteration"] = _

                                    # Append the results to the list
                                    results.append(
                                        {
                                            "store": ", ".join(map(str, store_name)),
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

    # upload to S3
    upload_obj(
        results_df,
        BUCKET_NAME,
        "in_home/processed/targets/hfss_agg.csv",
        kwargs_writing={"index": False},
    )
