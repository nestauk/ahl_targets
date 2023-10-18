import pandas as pd
from toolz import pipe
import re
from ahl_targets import PROJECT_DIR, BUCKET_NAME
from nesta_ds_utils.loading_saving.S3 import download_obj
from ahl_targets.utils.io import load_with_encoding
from ahl_targets.utils.create_tables import product_table
import os.path
import dask.dataframe as dd
from boto3.s3.transfer import TransferConfig


def purchase_records() -> pd.DataFrame:
    """
      Reads the purchase records file. As file too slow to load from s3, first save
      the file in inputs/data/.
    Args:
        None
    Returns:
        pd.DataFrame: purchase records dataframe
    """
    return pd.read_csv(PROJECT_DIR / "inputs/raw/purchase_records.csv")


def nutrition() -> pd.DataFrame:
    """
      Reads in dataset of purchase level nutritional information. As file too slow to load from s3, first save
      the file in inputs/data/.
    Args: None
    Returns: pd.DataFrame: nutrition dataframe
    """
    return pd.read_csv(
        PROJECT_DIR / "inputs/raw/nutrition_data.csv", encoding="ISO-8859-1"
    )


def purchase_subsets(date_period: str) -> pd.DataFrame:
    """
      Reads in purchase_records.csv and creates subset of purchase records file by defined month.
      First checks if files exists before creating.
    Args:
        date_period (int): Year and month to subset data (must match format in dataset - YYYYMM)
    Returns:
        subset (pd.DataFrame): Purchase records dataframe sliced by date_period.
    """
    file_path = PROJECT_DIR / "outputs/processed/pur_rec_" + str(date_period) + ".csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)
    else:
        subset_records = purchase_records().query(f"Period == {date_period}")
        subset_records.to_csv(file_path, index=False)
        return subset_records


def nutrition_subsets(date_period: str) -> pd.DataFrame:
    """
      Reads in the nutrition.csv and creates subset of by defined month.
      First checks if files exists before creating.
    Args:
        date_period (int): Year and month to subset data (must match format in dataset - YYYYMM)
    Returns:
        subset (pd.DataFrame): Nutrition dataframe sliced by date_period.
    """
    file_path = PROJECT_DIR / "outputs/processed/nut_" + str(date_period) + ".csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)
    else:
        pur_recs = pd.read_csv(PROJECT_DIR / "inputs/raw/nutrition_data.csv")
        subset = pur_recs[pur_recs["Purchase Period"] == date_period]
        subset.to_csv(file_path, index=False)
        return subset


def product_master() -> pd.DataFrame:
    """
      Reads the product master file.
      First checks if its saved locally and reads from s3 if it isn't.
    Args: None
    Returns: pd.DataFrame: product master dataframe
    """
    file_path = PROJECT_DIR / "inputs/raw/product_master.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data", "in_home/latest_data/product_master.csv"
            ),
            encoding="ISO-8859-1",
        )


def val_fields() -> pd.DataFrame:
    """
      Reads in dataset of codes to merge product master and uom information.
      First checks if its saved locally and reads from s3 if it isn't.
    Args: None
    Returns: pd.DataFrame: validation fields dataframe
    """
    file_path = PROJECT_DIR / "inputs/raw/validation_field.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data", "in_home/latest_data/validation_field.csv"
            ),
            encoding="ISO-8859-1",
        )


def uom() -> pd.DataFrame:
    """
      Reads in dataset of product measurement information.
      First checks if its saved locally and reads from s3 if it isn't.
    Args: None
    Returns: pd.DataFrame: uom dataframe
    """
    file_path = PROJECT_DIR / "inputs/raw/uom.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, header=0, names=["UOM", "Reported Volume"])
    else:
        return pd.read_csv(
            load_with_encoding("ahl-private-data", "in_home/latest_data/uom.csv"),
            encoding="ISO-8859-1",
            header=0,
            names=["UOM", "Reported Volume"],
        )


def product_codes() -> pd.DataFrame:
    """
      Reads in dataset which contains the codes to link products to category information.
      First checks if its saved locally and reads from s3 if it isn't.
    Args: None
    Returns: pd.DataFrame: product codes dataframe
    """
    file_path = PROJECT_DIR / "inputs/raw/product_attribute_coding.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data", "in_home/latest_data/product_attribute_coding.csv"
            ),
            encoding="ISO-8859-1",
        )


def product_values() -> pd.DataFrame:
    """
      Reads in dataset containing the product category information.
      First checks if its saved locally and reads from s3 if it isn't.
    Args: None
    Returns: pd.DataFrame: product values dataframe
    """
    file_path = PROJECT_DIR / "inputs/raw/product_attribute_values.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data", "in_home/latest_data/product_attribute_values.csv"
            ),
            encoding="ISO-8859-1",
        )


def product_attribute() -> pd.DataFrame:
    """
      Reads in dataset containing information on the product attributes.
      First checks if its saved locally and reads from s3 if it isn't.
    Args: None
    Returns: pd.DataFrame: product values dataframe
    """
    file_path = PROJECT_DIR / "inputs/raw/product_attribute.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data", "in_home/latest_data/product_attribute.csv"
            ),
            encoding="ISO-8859-1",
        )


def household_master() -> pd.DataFrame:
    """
      Reads in dataset of household information.
      First checks if its saved locally and reads from s3 if it isn't.
    Args: None
    Returns: pd.DataFrame: household master dataframe
    """
    file_path = PROJECT_DIR / "inputs/raw/panel_household_master.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data", "in_home/latest_data/panel_household_master.csv"
            ),
            encoding="ISO-8859-1",
        )


def household_ind() -> pd.DataFrame:
    """Reads in dataset of information about each household member.
       First checks if its saved locally and reads from s3 if it isn't.

    Args: None
    Returns: pd.DataFrame: household individual dataframe
    """
    file_path = PROJECT_DIR / "inputs/raw/panel_individual_master.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data", "in_home/latest_data/panel_individual_master.csv"
            ),
            encoding="ISO-8859-1",
        )


def demog_coding() -> pd.DataFrame:
    """
      Reads in dataset of codes per household that links to demographic information.
      First checks if its saved locally and reads from s3 if it isn't.
    Args: None
    Returns: pd.DataFrame: demographic coding dataframe
    """
    file_path = PROJECT_DIR / "inputs/raw/panel_demographic_coding.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data", "in_home/latest_data/panel_demographic_coding.csv"
            ),
            encoding="ISO-8859-1",
        )


def demog_val() -> pd.DataFrame:
    """
      Reads in dataset of demographic values per code.
      First checks if its saved locally and reads from s3 if it isn't.
    Args: None
    Returns: pd.DataFrame: demographic values dataframe
    """
    file_path = PROJECT_DIR / "inputs/raw/panel_demographic_values.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data", "in_home/latest_data/panel_demographic_values.csv"
            ),
            encoding="ISO-8859-1",
        )


def panel_weights() -> pd.DataFrame:
    """
    Reads the panel weights file.
    First checks if its saved locally and reads from s3 if it isn't.
    Args: None
    Returns: pd.DataFrame: demographic weights by period

    """
    file_path = PROJECT_DIR / "inputs/raw/panel_demographic_weights_period.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data",
                "in_home/latest_data/panel_demographic_weights_period.csv",
            ),
            encoding="ISO-8859-1",
        )


def panel_weights_year() -> pd.DataFrame:
    """
    Reads the panel weights file.
    First checks if its saved locally and reads from s3 if it isn't.
    Args: None
    Returns: pd.DataFrame: demographic weights for the year

    """
    file_path = PROJECT_DIR / "inputs/raw/panel_demographic_weights_year.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data",
                "in_home/latest_data/panel_demographic_weights_year.csv",
            ),
            encoding="ISO-8859-1",
        )


def product_metadata() -> pd.DataFrame:
    """Table combining all the product metadata
    Args: None
    Returns: pd.DataFrame: table with unique prduct codes and characteristics"""

    return pipe(
        product_table(
            val_fields(),
            product_master(),
            uom(),
            product_codes(),
            product_values(),
            product_attribute(),
        ),
        lambda df: df.rename(
            columns={c: re.sub(" ", "_", c.lower()) for c in df.columns}
        ),
    )


def purchase_records_volume() -> pd.DataFrame:
    """
    Getter for the copy of purchase record with imputed weights.
    Args:
        None
    Returns:
        df (pd.DataFrame): pruchase records with additional columns for volumes
    """

    return (
        pd.read_csv(PROJECT_DIR / "inputs/processed/pur_rec_volume.csv")
        .iloc[:, 1:]
        .query("Quantity > 0")
    )


def purchase_records_updated() -> pd.DataFrame:
    """
    Getter for the copy of purchase record with imputed weights.
    Cleaned to format of purchase records but with reported volume added.
    Args:
        None
    Returns:
        df (pd.DataFrame): purchase records with imputed volumes
    """
    pur_recs_updated = purchase_records_volume()
    return pur_recs_updated.drop(
        ["Reported Volume", "volume_per", "Volume"], axis=1
    ).rename({"reported_volume_up": "Reported Volume", "volume_up": "Volume"}, axis=1)


def product_measurement() -> pd.DataFrame:
    """File containing all available measurements per product
    Args: None
    Returns: pd.DataFrame: list of product measurements
    """
    file_path = (
        PROJECT_DIR
        / "inputs/raw/Nesta - Units, Grams, Millilitres, Servings All Products.txt"
    )
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data",
                "in_home/latest_data/Nesta - Units, Grams, Millilitres, Servings All Products.txt",
            ),
            encoding="ISO-8859-1",
        )


def store_itemisation_lines() -> pd.DataFrame:
    """
      Reads the store_itemisation_lines file.
    Args:
        None
    Returns:
        pd.DataFrame: store_itemisation_lines dataframe
    """
    return pd.read_csv(PROJECT_DIR / "inputs/raw/store_itemisation_lines.csv")


def store_itemisation_coding() -> pd.DataFrame:
    """
      Reads the store_itemisation_coding file.
    Args:
        None
    Returns:
        pd.DataFrame: store_itemisation_coding dataframe
    """
    return pd.read_csv(PROJECT_DIR / "inputs/raw/store_itemisation_coding.csv")


def panel_kcal_deciles() -> pd.DataFrame:
    """
      Reads the file containing the kcal deciles.
    Args:
        None
    Returns:
        pd.DataFrame: panel_kcal_deciles dataframe
    """
    return pd.read_csv(PROJECT_DIR / "outputs/processed/hh_kcal_groups.csv")


def get_imputed_bmi() -> pd.DataFrame:
    """Reads the file containing the imputed BMI values
    Args: None
    Returns: pd.DataFrame: file with imputed bmi values"""
    return pd.read_csv(PROJECT_DIR / "inputs/processed/imputed_bmi.csv")


def product_type() -> dict:
    """Read manually labeled file with product types (ingredient, staple, discretionary)
    Args: None
    Returns: pd.DataFrame: file with product taxonomy by type"""
    file_path = PROJECT_DIR / "inputs/processed/product_type.csv"

    # Check if file already exists:
    def read_file(file_path):
        if os.path.isfile(file_path):
            return pd.read_csv(file_path)
        else:
            return pd.read_csv(
                load_with_encoding(
                    "ahl-private-data", "in_home/latest_data/product_type.csv"
                )
            )

    dat = read_file(file_path)
    dat["key"] = dat["rst_4_market"] + dat["rst_4_market_sector"]
    return dat.set_index("key")["type"].to_dict()


def get_gravity() -> pd.DataFrame:
    """Returns a dataframe with specific gravities for drink products
    Args: None
    Returns: pd.DataFrame: specific gravity file manually compiled"""
    file_path = PROJECT_DIR / "inputs/processed/specific_gravity.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.read_csv(
            load_with_encoding(
                "ahl-private-data", "in_home/latest_data/specific_gravity.csv"
            )
        )


def get_fvn() -> pd.DataFrame:
    """Returns a dataframe with foot, veg, nuts points for NPM score
    Args: None
    Returns: pd.DataFrame: FVN scores by catgory as supplied by provider"""
    file_path = PROJECT_DIR / "inputs/processed/fvn_points.csv"
    # Check if file already exists:
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.read_csv(
            load_with_encoding("ahl-private-data", "in_home/latest_data/fvn_points.csv")
        )


def get_npm() -> pd.DataFrame:
    """
      Reads the npm_scores file.
    Args:
        None
    Returns:
        pd.DataFrame: npm_scores dataframe
    """
    return download_obj(
        BUCKET_NAME,
        "in_home/processed/npm_with_nut.csv",
        download_as="dataframe",
    )


def model_data():
    """
      Reads the npm_scores file.
    Args:
        None
    Returns:
        pd.DataFrame: npm_scores dataframe
    """
    return download_obj(
        BUCKET_NAME,
        "in_home/processed/targets/model_data.parquet",
        download_as="dataframe",
        kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
    )
