import numpy as np


def round_to_nearest_5(number):
    """
    Round a given number to the nearest multiple of 5.

    Args:
        number (int or float): The number to be rounded.

    Returns:
        int or float: The rounded number.
    """
    return round(number / 5) * 5


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


def apply_reduction_ed(store_weight, ed_reduction):
    # Modify the values of the column
    modified_column = np.where(
        store_weight["indicator_reform"] == 1,
        store_weight["ed"] - ed_reduction * store_weight["ed"] / 100,
        store_weight["ed"],
    )

    return modified_column


def apply_reduction_npm(random_sample, npm_reduction):
    # Modify the values of the column
    modified_column = np.where(
        random_sample["indicator_reform"] == 1,
        random_sample["npm_score"] - npm_reduction,
        random_sample["npm_score"],
    )

    return modified_column


def apply_reduction_hfss(random_sample):
    # Modify the values of the column
    modified_column = np.where(
        random_sample["indicator_reform"] == 1, 3, random_sample["npm_score"]
    )

    return modified_column
