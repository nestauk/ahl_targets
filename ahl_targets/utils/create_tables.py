import pandas as pd


def map_codes(df: pd.DataFrame, col: str):
    """
    Maps code numbers to values

    Args:
        df (pd.DataFrame): Pandas dataframe
        col (str): Name of column to map

    Returns:
        pd.DateFrame:
    """
    di = {
        2: "Urban-Rural",
        3: "Social Class",
        4: "Council Tax Band",
        5: "Region",
        7: "Newspaper Read",
        8: "Life Stage",
        9: "Household Income",
        11: "Ethnicity",
        12: "Education Level",
    }
    return df[col].map(di)


def hh_demographic_table(
    demog_coding: pd.DataFrame, demog_val: pd.DataFrame, pan_mast: pd.DataFrame
):
    """
    Creates a dataframe with combined demographic information about each household

    Args:
        demog_coding (pd.DataFrame): Pandas dataframe codes per household that links to demographic information
        demog_val (pd.DataFrame): Pandas dataframe demographic values per code
        pan_mast (pd.DataFrame): Pandas dataframe of household information

    Returns:
        pd.DateFrame: Dataframe with demographic information for each household
    """
    demog_val.columns = ["Demog Id", "Demog Value", "Demog Description"]
    hh_demogs = demog_coding.merge(
        demog_val,
        left_on=["Demog Id", "Demog Value"],
        right_on=["Demog Id", "Demog Value"],
        how="left",
    )
    hh_demogs.drop(["Demog Value"], axis=1, inplace=True)
    hh_demogs["Demog Id"] = map_codes(hh_demogs, "Demog Id")
    hh_demogs.set_index("Panel Id", inplace=True)
    hh_demogs = hh_demogs.pivot_table(
        values="Demog Description",
        index=hh_demogs.index,
        columns="Demog Id",
        aggfunc="first",
    )
    return pd.merge(
        hh_demogs, pan_mast.set_index("Panel Id"), left_index=True, right_index=True
    )


def product_table(
    val_fields: pd.DataFrame,
    prod_mast: pd.DataFrame,
    uom: pd.DataFrame,
    prod_codes: pd.DataFrame,
    prod_vals: pd.DataFrame,
    prod_att: pd.DataFrame,
):
    """
    Creates a dataframe with information for each product

    Args:
        val_fields (pd.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
        prod_mast (pd.DataFrame): Panadas dataframe unique product list
        uom (pd.DataFrame): Panadas dataframe contains product measurement information
        prod_codes (pd.DataFrame): Panadas dataframe contains the codes to link products to category information
        prod_vals (pd.DataFrame): Panadas dataframe contains the product category information
        prod_att (pd.DataFrame): Panadas dataframe containing description for each attribute

    Returns:
        pd.DateFrame: Dataframe with information for each product
    """
    # Get volume
    val_fields.drop_duplicates(inplace=True)  # Remove duplicates
    prod_vol = (
        prod_mast[["Product Code", "Validation Field"]]
        .merge(
            val_fields[["VF", "UOM"]],
            left_on="Validation Field",
            right_on="VF",
            how="left",
        )
        .merge(uom[["UOM", "Reported Volume"]], on="UOM", how="left")
        .drop(["Validation Field", "VF", "UOM"], axis=1)
    )
    # Get product info (including categories)
    att_dict = dict()
    for ix, row in prod_att.iterrows():
        att_dict[row["Attribute Number"]] = row["Attribute Description"]
    prod_codes["Attribute"] = prod_codes["Attribute Number"].apply(
        lambda x: att_dict[x]
    )
    combined_prod_att = prod_codes.merge(
        prod_vals, left_on="Attribute Value", right_on="Attribute Value", how="left"
    )
    combined_prod_att.set_index("Product Code", inplace=True)
    combined_prod_att = combined_prod_att.pivot_table(
        values="Attribute Value Description",
        index=combined_prod_att.index,
        columns="Attribute",
        aggfunc="first",
    )
    return pd.merge(
        combined_prod_att,
        prod_vol.set_index("Product Code"),
        left_index=True,
        right_index=True,
    ).reset_index()


def nutrition_merge(nutrition: pd.DataFrame, purch_recs: pd.DataFrame, cols: list):
    """Merges the purchase records and nutrition file
    Args:
        nutrition (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        cols (list): List of columns names to merge from the nutrition dataset
    Returns:
        (pandas.DateFrame): Merged pandas dataframe
    """
    # Add unique purchase ID
    nutrition["pur_id"] = (
        nutrition["Purchase Number"].astype(str)
        + "_"
        + nutrition["Purchase Period"].astype(str)
    )
    purch_recs["pur_id"] = (
        purch_recs["PurchaseId"].astype(str) + "_" + purch_recs["Period"].astype(str)
    )
    # Merge datasets
    return purch_recs.merge(
        nutrition[["pur_id"] + cols].copy(), on="pur_id", how="left"
    )


def total_product_hh_purchase(purch_recs: pd.DataFrame, cols):
    """Groups by household, measurement and product and sums the volume and kcal content.
    Args:
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        cols (list): List of cols to group (different for kcal and volume representations)
    Returns:
        (pandas.DateFrame): groupby pandas dataframe
    """
    # Remove cases where volume is zero (8 cases)
    purch_recs = purch_recs[purch_recs["Volume"] != 0].copy()
    purch_recs["Gross_up_kcal"] = (
        purch_recs["Energy KCal"] * purch_recs["Gross Up Weight"]
    )
    return (
        purch_recs.groupby(["Panel Id"] + cols)[
            ["Volume", "Energy KCal", "Quantity", "Gross Up Weight", "Gross_up_kcal"]
        ]
        .sum()
        .reset_index()
    )


def make_purch_records(
    nutrition: pd.DataFrame, purchases_comb: pd.DataFrame, cols: list
):
    """
    Merges dataframes to create purchase records df with food category and nutrition information
    Args:
        nutrition (pd.DataFrame): Pandas dataframe of purchase level nutritional information
        purchases_comb (pd.DataFrame): Combined files to give product informaion to purchases
        cols (list): Columns to use for groupby
    Returns:
        pd.DateFrame: Household totals per food category
    """
    purchases_nutrition = nutrition_merge(nutrition, purchases_comb, ["Energy KCal"])
    return total_product_hh_purchase(purchases_nutrition, cols)


def hh_kcal_per_prod(purch_recs: pd.DataFrame, kcal_col: str):
    """
    Unstacks df to show total kcal per product per household
    Args:
        purch_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        kcal_col (str): Energy Kcal column (weighted or unweighted)
    Returns:
        (pd.DateFrame): Kcal totals per product per household
    """
    purch_recs = (
        purch_recs.set_index(["Panel Id", "att_vol"])[[kcal_col]]
        .unstack(["att_vol"])
        .fillna(0)
    )
    purch_recs.columns = purch_recs.columns.droplevel()
    return purch_recs


def hh_kcal_weight(
    prod_cat: int,
    pur_recs: pd.DataFrame,
    nut_recs: pd.DataFrame,
    prod_meta: pd.DataFrame,
):
    """
    Create weighted hh kcal per cat
    Args:
        prod_category (int): one product category
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut_recs (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        prod_meta (pd.DataFrame): Pandas dataframe with product descriptions
    Returns:
        pd.DataFrame: Table with metrics based on kcal contribution per category
    """
    comb_files = pur_recs.merge(
        prod_meta[["product_code", prod_cat]],
        left_on=["Product Code"],
        right_on="product_code",
        how="left",
    )
    comb_files = comb_files[
        comb_files["Reported Volume"].notna()
    ]  # Remove purchases with no volume
    comb_files["att_vol"] = comb_files[prod_cat]
    comb_files.drop("product_code", axis=1, inplace=True)
    # Make household representations
    purch_recs_comb = make_purch_records(nut_recs, comb_files, ["att_vol"])
    return hh_kcal_per_prod(purch_recs_comb, "Gross_up_kcal")


def combine_files(
    prod_cat: str,
    pur_recs: pd.DataFrame,
    prod_meta: pd.DataFrame,
):
    """
    Cleans purchase records file and combines product info
    Args:
        prod_category (str): one product category
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        prod_meta (pd.DataFrame): Pandas dataframe with product descriptions
    Returns:
        pd.DataFrame: Cleaned purchase records with prod info
    """
    comb_files = pur_recs.merge(
        prod_meta[["product_code", prod_cat]],
        left_on=["Product Code"],
        right_on="product_code",
        how="left",
    )
    comb_files = comb_files[
        comb_files["Reported Volume"].notna()
    ]  # Remove purchases with no volume
    comb_files["att_vol"] = comb_files[prod_cat]
    return comb_files.drop("product_code", axis=1)


def hh_kcal_unweighted(
    prod_cat: str,
    pur_recs: pd.DataFrame,
    nut_recs: pd.DataFrame,
    prod_meta: pd.DataFrame,
):
    """
    Create weighted hh kcal per cat
    Args:
        prod_category (str): one product category
        pur_recs (pd.DataFrame): Pandas dataframe contains the purchase records of specified data
        nut_recs (pd.DataFrame): Pandas dataframe with per purchase nutritional information
        prod_meta (pd.DataFrame): Pandas dataframe with product descriptions
    Returns:
        pd.DataFrame: Table with metrics based on kcal contribution per category
    """
    # Make household representations
    purch_recs_comb = make_purch_records(
        nut_recs, combine_files(prod_cat, pur_recs, prod_meta), ["att_vol"]
    )
    return hh_kcal_per_prod(purch_recs_comb, "Energy KCal")


def measure_table(conv: pd.DataFrame):
    """
    Creates a table of products and measurements
    Args:
        conv (pd.DataFrame): Pandas dataframe with product measurements
    Returns:
        pd.DateFrame: Dataframe with table of products and measurements
    """
    conv_update = conv.copy()
    conv_update.set_index("PRODUCT", inplace=True)
    conv_meas = (
        conv_update.groupby([conv_update.index, "VOLUME TYPE"])["VALUE"]
        .first()
        .unstack()
        .reset_index()
    )
    conv_meas.columns = ["Product Code", "Grams", "Millilitres", "Servings", "Units"]
    conv_meas["Litres"] = conv_meas["Millilitres"] / 1000
    return conv_meas


def get_store1_table(store_coding: pd.DataFrame, store_lines: pd.DataFrame):
    """
    Creates a table with store 1 (kantar provide two store taxonomies) information.
    Args:
        store_coding (pd.DataFrame): Pandas dataframe with store codes
        store_lines (pd.DataFrame): Pandas dataframe with store category names
    Returns:
        pd.DateFrame: Dataframe with table of products and measurements
    """
    coding_1 = (
        store_coding[store_coding["itemisation_id"] == 1]
        .drop(["itemisation_id"], axis=1)
        .copy()
    )
    lines_1 = (
        store_lines[store_lines["itemisation_id"] == 1]
        .drop(["itemisation_id"], axis=1)
        .copy()
    )
    return coding_1.merge(
        lines_1,
        how="left",
        left_on="itemisation_line_id",
        right_on="itemisation_line_id",
    ).drop(["itemisation_line_id"], axis=1)
