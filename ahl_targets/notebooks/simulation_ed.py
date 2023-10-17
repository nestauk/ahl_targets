#!/usr/bin/env python
# coding: utf-8
# %%

# %%


from ahl_targets.pipeline import model_data
from ahl_targets.getters import get_data
from functools import reduce
import pandas as pd
from ahl_targets import PROJECT_DIR
import matplotlib.pyplot as plt
import numpy as np
import random
import statsmodels.formula.api as sm
from ahl_targets.pipeline import stores_transformation as stores
from ahl_targets.pipeline import hfss
from ahl_targets.pipeline import product_transformation as product
import pyarrow
import statsmodels.api as sm
from plotnine import (
    ggplot,
    geom_line,
    aes,
    stat_smooth,
    facet_wrap,
    geom_smooth,
    labs,
    geom_hline,
    geom_density,
    geom_vline,
    after_stat,
    geom_vline,
)


# %%


random.seed(42)


# %%


# read data
store_data = get_data.model_data().compute()
prod_table = get_data.product_metadata()


# %%


def weighted_kg_by_store(store_data):
    """aggregated df with volume weights

    Args:
        store_data (pd.DataFrame): merged data
    """
    store_data["cross"] = store_data["volume_up"] * store_data["Gross Up Weight"]
    out = (
        store_data.groupby(["product_code", "store_cat"])["cross"]
        .sum()
        .reset_index(name="total_kg")
    )
    out["kg_w"] = out["total_kg"] / out["total_kg"].sum()

    return out


def weighted_kcal_by_store(store_data):
    """aggregated df with volume weights

    Args:
        store_data (pd.DataFrame): merged data
    """
    store_data["cross"] = store_data["Energy KCal"] * store_data["Gross Up Weight"]
    out = (
        store_data.groupby(["product_code", "store_cat"])["cross"]
        .sum()
        .reset_index(name="total_kcal")
    )
    out["kcal_w"] = out["total_kcal"] / out["total_kcal"].sum()

    return out


def weighted_prod_by_store(store_data):
    """aggregated df with volume weights

    Args:
        store_data (pd.DataFrame): merged data
    """
    out = (
        store_data.groupby(["product_code", "store_cat"])["Gross Up Weight"]
        .sum()
        .reset_index(name="total_prod")
    )
    out["prod_w"] = out["total_prod"] / out["total_prod"].sum()

    return out


def unique_ed(store_data):
    """Unique ED by prodyuct

    Args:
        store_data (pd.DataFrame): marged data

    Returns:
        pd.DataFrame: aggregate df by product code with unique ED
    """
    return store_data.groupby(["product_code"])["ed"].mean().reset_index(name="ed")


def unique_kcal(store_data):
    """Unique KCAL by prodyuct

    Args:
        store_data (pd.DataFrame): marged data

    Returns:
        pd.DataFrame: aggregate df by product code with unique KCAL
    """
    return (
        (store_data["Energy KCal"] / store_data["Quantity"])
        .groupby(store_data["product_code"])
        .mean()
        .reset_index(name="kcal_unit")
    )


def unique_npm(store_data):
    """Unique NPM by prodyuct

    Args:
        store_data (pd.DataFrame): marged data

    Returns:
        pd.DataFrame: aggregate df by product code with unique NPM
    """

    return (
        store_data.groupby(["product_code"])["npm_score"]
        .mean()
        .reset_index(name="npm_score")
    )


def unique_spend(store_data):
    """Unique spend by prodyuct

    Args:
        store_data (pd.DataFrame): marged data

    Returns:
        pd.DataFrame: aggregate df by product code with unique NPM
    """
    return (
        (store_data["Spend"] / store_data["Quantity"])
        .groupby(store_data["product_code"])
        .mean()
        .reset_index(name="spend")
    )


def unique_hfss(store_data):
    """Unique HFSS by prodyuct

    Args:
        store_data (pd.DataFrame): marged data

    Returns:
        pd.DataFrame: aggregate df by product code with unique NPM
    """

    return (store_data.groupby(["product_code"])["in_scope"].max()).reset_index(
        name="hfss"
    )


def weighted_ed(store_data):
    """merged aggregated weight data

    Args:
        store_data (pd.DataFrame): merged data

    Returns:
        pd.DataFrame: merged aggregated weight data
    """
    return (
        weighted_kg_by_store(store_data)
        .merge(weighted_kcal_by_store(store_data), on=["product_code", "store_cat"])
        .merge(weighted_prod_by_store(store_data), on=["product_code", "store_cat"])
        .merge(unique_ed(store_data), on="product_code")
        .merge(unique_kcal(store_data), on="product_code")
        .merge(unique_spend(store_data), on="product_code")
    )


def weighted_npm(store_data):
    """merged aggregated weight data

    Args:
        store_data (pd.DataFrame): merged data

    Returns:
        pd.DataFrame: merged aggregated weight data
    """
    return (
        weighted_kg_by_store(store_data)
        .merge(weighted_kcal_by_store(store_data), on=["product_code", "store_cat"])
        .merge(weighted_prod_by_store(store_data), on=["product_code", "store_cat"])
        .merge(unique_npm(store_data), on="product_code")
        .merge(unique_ed(store_data), on="product_code")
        .merge(unique_kcal(store_data), on="product_code")
        .merge(unique_spend(store_data), on="product_code")
    )


def weighted_hfss(store_data):
    """merged aggregated weight data

    Args:
        store_data (pd.DataFrame): merged data

    Returns:
        pd.DataFrame: merged aggregated weight data
    """
    return (
        weighted_kg_by_store(store_data)
        .merge(weighted_kcal_by_store(store_data), on=["product_code", "store_cat"])
        .merge(weighted_prod_by_store(store_data), on=["product_code", "store_cat"])
        .merge(unique_npm(store_data), on="product_code")
        .merge(unique_ed(store_data), on="product_code")
        .merge(unique_kcal(store_data), on="product_code")
        .merge(unique_hfss(store_data), on="product_code")
        .merge(unique_spend(store_data), on="product_code")
    )


def prod_weight_g(w_store_data):
    """average weight of product in grames

    Args:
        w_store_data (pd.DataFrame): merged data

    Returns:
        pd.Series: average weight of product
    """
    return w_store_data["total_kg"] / w_store_data["total_prod"] * 1000


def npm_ed_table(store_data):
    """
    Returns a table of the mean value of the 'ed' column in the given DataFrame, grouped by the unique values in the 'npm_score' column.

    Args:
    - store_data (pandas.DataFrame): The DataFrame to group by and aggregate.

    Returns:
    - pandas.DataFrame: A new DataFrame containing the mean value of the 'ed' column for each unique value in the 'npm_score' column.
    """
    # Group the DataFrame by the 'npm_score' column and calculate the mean value of the 'ed' column for each group
    table = store_data.groupby(["npm_score"])["ed"].mean()

    # Reset the index of the resulting DataFrame to make 'npm_score' a column again
    table = table.reset_index()

    # Return the resulting table
    return table


# Define a function to perform calculations on the column
def apply_reduction_ed(store_weight, ed_reduction):
    # Modify the values of the column
    modified_column = np.where(
        store_weight["indicator_reform"] == 1,
        store_weight["ed"] - ed_reduction * store_weight["ed"] / 100,
        store_weight["ed"],
    )

    return modified_column


# %%


store_weight = weighted_ed(store_data)

store_weight["prod_weight_g"] = store_weight.pipe(prod_weight_g)


# %%


# assumptions:
# 1 - reformulation (lowering ED) only happens in products that have at least 400 kcal/100g (high ED)
# 2 - sales changes reflect a switch between high ED to low ED (with cut off 400kcal/100g). The percentage is calculated as a proportion of current sales so whilst a nominal x% is taken for different products in absolute numbers this can mean different number of products


# %%


# version 3

num_iterations = 20
product_share_reform_values = [0.5]  # share of products that are reformulated
product_share_sale_values = [1]  # share of products whose sales are shifted
ed_reduction_values = [
    5,
    7.5,
    10,
]  # percentage reduction in energy density e.g. 1 indicates a 1% reduction in energy density
high_sales_change_values = [2.5, 5, 10]  # percentage shift in sales
low_sales_change_values = [2.5, 5, 10]  # percentage shift in sales
cutoff = 400  # cut off for high energy density


# %%


results = []
results_data = []


# Nested loop to iterate through different values of product_share and ed_reduction
for product_share_reform in product_share_reform_values:
    for product_share_sale in product_share_sale_values:
        for ed_reduction in ed_reduction_values:
            for sales_change_high in high_sales_change_values:
                for sales_change_low in low_sales_change_values:
                    # Repeat the code num_iterations times
                    for _ in range(num_iterations):
                        # split into high and low ed
                        ed_cut = store_weight["ed"] >= cutoff
                        high_ed = store_weight[ed_cut].copy()
                        low_ed = store_weight[~ed_cut].copy()

                        # generate list of products to reformulate
                        unique_products = pd.DataFrame(
                            store_weight["product_code"].unique(),
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
                            apply_reduction_ed, ed_reduction
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
                        kcal_pp_new = randomised["new_kcal_tot"].sum() / 66000000 / 365

                        total_prod_baseline = randomised["total_prod"].sum()
                        total_prod_new = randomised["new_total_prod"].sum()

                        kcal_high_baseline = randomised[randomised["ed"] >= cutoff][
                            "total_kcal"
                        ].sum()
                        kcal_high_new = randomised[randomised["new_ed"] >= cutoff][
                            "new_kcal_tot"
                        ].sum()

                        kcal_low_baseline = randomised[randomised["ed"] < cutoff][
                            "total_kcal"
                        ].sum()
                        kcal_low_new = randomised[randomised["new_ed"] < cutoff][
                            "new_kcal_tot"
                        ].sum()

                        spend_baseline = (
                            ((randomised["total_prod"] * randomised["spend"]).sum())
                            / 66000000
                            / 52
                        )
                        spend_new = (
                            ((randomised["new_total_prod"] * randomised["spend"]).sum())
                            / 66000000
                            / 52
                        )

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
                            )
                        )


# Create the DataFrame from the list of results
results_df = pd.DataFrame(results)

results_data_df = pd.concat(results_data, ignore_index=True)


# %%


# note that when all products are selected (e.g. shares are 1) - all iterations are the same
avg = (
    results_df.groupby(
        [
            "product_share_reform",
            "product_share_sale",
            "sales_change_high",
            "sales_change_low",
            "ed_reduction",
        ]
    )[
        [
            "mean_ed_kg_new",
            "mean_ed_kcal_new",
            "mean_ed_kg_baseline",
            "mean_ed_kcal_baseline",
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


# %%


# Extract columns with the suffix '_baseline'
baseline_columns = avg.filter(like="_baseline")

# Extract columns with the suffix '_new'
new_columns = avg.filter(like="_new")

baseline_columns.columns = baseline_columns.columns.str.replace("_baseline", "")
new_columns.columns = new_columns.columns.str.replace("_new", "")

result = (new_columns - baseline_columns) / baseline_columns
df = pd.concat([avg, result.add_suffix("_diff_percentage")], axis=1)
df["kcal_diff"] = df["kcal_pp_new"] - df["kcal_pp_baseline"]


# %%


df.to_csv(PROJECT_DIR / "outputs/reports/simulation_ED.csv", index=False)


# %%


(
    ggplot(
        df,
        aes(
            x="mean_ed_kg_diff_percentage",
            y="kcal_pp_new",
            color="factor(sales_change_low)",
        ),
    )
    + geom_line()
    + geom_hline(yintercept=[1607], linetype="dashed")
    + geom_hline(yintercept=[1607 - 80])
    + geom_vline(xintercept=[0], linetype="dashed")
    + facet_wrap("sales_change_high", labeller="label_both", nrow=4)
    + labs(
        x="Change in Volume Weigthed Average ED",
        y="Per Capita Daily Kcal",
        color="Sales Increase in Medium/Low ED Products",
    )
)


# %%


# suitable scnarios
suitable = df[
    (df["mean_ed_kg_diff_percentage"] < 0)
    & (df["kcal_pp_new"] <= 1607)
    & (df["spend_diff_percentage"] >= -5)
]


# %%


suitable


# %%


full_results = results_data_df[
    (results_data_df["ed_reduction"] == 10)
    & (results_data_df["sales_change_high"] == 10)
    & (results_data_df["sales_change_low"] == 2.5)
]

grouped = full_results.groupby(
    [
        "product_code",
        "store_cat",
        "product_share_reform",
        "product_share_sale",
        "ed_reduction",
        "sales_change_high",
        "sales_change_low",
    ]
)
density_plot = grouped.mean().reset_index()


# %%


density_plot["ed"].plot(kind="hist", weights=density_plot["kg_w"], alpha=0.2)
density_plot["new_ed"].plot(kind="hist", weights=density_plot["kg_w_new"], alpha=0.2)
plt.axvline(
    x=(density_plot["ed"] * density_plot["kg_w"]).sum() / density_plot["kg_w"].sum(),
    color="b",
    linestyle="--",
)
plt.axvline(
    x=(density_plot["new_ed"] * density_plot["kg_w_new"]).sum()
    / density_plot["kg_w_new"].sum(),
    color="r",
    linestyle="--",
)
plt.text(
    (density_plot["ed"] * density_plot["kg_w"]).sum() / density_plot["kg_w"].sum() + 10,
    0.35,
    "Mean before: "
    + str(
        round(
            (density_plot["ed"] * density_plot["kg_w"]).sum()
            / density_plot["kg_w"].sum(),
            1,
        )
    )
    + "kcal/day",
)
plt.text(
    (density_plot["new_ed"] * density_plot["kg_w_new"]).sum()
    / density_plot["kg_w_new"].sum()
    + 10,
    0.25,
    "Mean after: "
    + str(
        round(
            (density_plot["new_ed"] * density_plot["kg_w_new"]).sum()
            / density_plot["kg_w_new"].sum(),
            1,
        )
    )
    + "kcal/day",
)
plt.title("Volume weighted ED")
plt.legend(loc="upper right")
plt.show()


# %%


# absolute target
# under the scenario of an absolute target for average ED reduction, all retailers would be mandated to achieve the same reduction in average ED which is the one generated by our model
# for each retailer, we show how far they would be from achieveing that


# %%


# from store_weight, generate the average ED weighted by kg for each retailer
avg_retailer = (
    (store_weight["kg_w"] * store_weight["ed"]).groupby(store_weight["store_cat"]).sum()
    / store_weight["kg_w"].groupby(store_weight["store_cat"]).sum()
).reset_index(name="ed")
avg_retailer["weight"] = (
    store_weight.groupby(["store_cat"])["kg_w"].sum().reset_index()["kg_w"]
)
avg_retailer["target"] = (
    density_plot["new_ed"] * density_plot["kg_w_new"]
).sum() / density_plot["kg_w_new"].sum()
avg_retailer["diff"] = avg_retailer["ed"] - avg_retailer["target"]
avg_retailer["diff_percentage"] = 100 * (
    avg_retailer["target"] / avg_retailer["ed"] - 1
)


# %%


avg_retailer


# %%


# relative target
# a relative target is one that would allow the overall ED average to meet that specified by the target but


# %%


values = avg_retailer["ed"]
weights = avg_retailer["weight"]
target_average = avg_retailer["target"].mean()

current_average = np.average(values, weights=weights)
percentage_reduction = (current_average - target_average) / current_average * 100

reduced_values = values * (1 - percentage_reduction / 100)
reductions = values - reduced_values
reduction_percentages = reductions / values * 100
weighted_average = np.average(reduced_values, weights=weights)

print("Reduced values:", reduced_values)
print("Weighted average:", weighted_average)
print("Current average:", current_average)
print("Target average:", target_average)

print("Overall Percentage reduction needed: {:.2f}%".format(percentage_reduction))
for i in range(len(values)):
    print("Value {}: {:.2f}% reduction".format(i + 1, reduction_percentages[i]))

if np.isclose(weighted_average, target_average):
    print("The target average is the same as the calculated weighted average.")
else:
    print("The target average is different from the calculated weighted average.")


# %%

# %%
