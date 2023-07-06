import pandas as pd
import numpy as np
import polars as pl
import logging
from typing import Tuple
import os
from typing import Any, Callable, List, Union
from concurrent.futures import ThreadPoolExecutor


def product_table(
    val_fields: pl.DataFrame,
    prod_mast: pl.DataFrame,
    uom: pl.DataFrame,
    prod_codes: pl.DataFrame,
    prod_vals: pl.DataFrame,
    prod_att: pl.DataFrame,
) -> pl.DataFrame:
    """
    Creates a dataframe with information for each product

    Args:
        val_fields (pl.DataFrame): Pandas dataframe with codes to merge product master and uom dfs
        prod_mast (pl.DataFrame): Panadas dataframe unique product list
        uom (pl.DataFrame): Panadas dataframe contains product measurement information
        prod_codes (pl.DataFrame): Panadas dataframe contains the codes to link products to category information
        prod_vals (pl.DataFrame): Panadas dataframe contains the product category information
        prod_att (pl.DataFrame): Panadas dataframe containing description for each attribute

    Returns:
        pl.DateFrame: Dataframe with information for each product
    """
    import polars as pl

def product_table(
    val_fields: pl.DataFrame,
    prod_mast: pl.DataFrame,
    uom: pl.DataFrame,
    prod_codes: pl.DataFrame,
    prod_vals: pl.DataFrame,
    prod_att: pl.DataFrame,
) -> pl.DataFrame:
    """
    Creates a dataframe with information for each product

    Args:
        val_fields (pl.DataFrame): Polars DataFrame with codes to merge product master and uom dfs
        prod_mast (pl.DataFrame): Polars DataFrame unique product list
        uom (pl.DataFrame): Polars DataFrame contains product measurement information
        prod_codes (pl.DataFrame): Polars DataFrame contains the codes to link products to category information
        prod_vals (pl.DataFrame): Polars DataFrame contains the product category information
        prod_att (pl.DataFrame): Polars DataFrame containing description for each attribute

    Returns:
        pl.DataFrame: DataFrame with information for each product
    """
    # Get volume
    val_fields = val_fields.unique()  # Remove duplicates
    prod_vol = (
        prod_mast.select(["Product Code", "Validation Field"])
        .join(val_fields.select(["VF", "UOM"]))
        .join(uom.select(["UOM", "Reported Volume"]))
        .drop(["Validation Field", "VF", "UOM"])
    )
    # Get product info (including categories)
    att_dict = prod_att.to_pandas().set_index("Attribute Number")["Attribute Description"].to_dict()
    prod_codes = prod_codes.with_columns(
        "Attribute", prod_codes["Attribute Number"].apply(lambda x: att_dict[x])
    )
    combined_prod_att = prod_codes.join(prod_vals, left_on="Attribute Value", right_on="Attribute Value")
    combined_prod_att = combined_prod_att.with_columns("Product Code", combined_prod_att.index())
    combined_prod_att = combined_prod_att.pivot(
        pivot_column="Attribute",
        value_column="Attribute Value Description",
        agg_function="first",
        columns=combined_prod_att.column_names(),
    )
    return combined_prod_att.join(prod_vol, left_on="Product Code", right_on="Product Code").drop_nulls().sort("Product Code")

def clean_tbl(prod_meta: pl.DataFrame, pur_rec: pl.DataFrame) -> pl.DataFrame:
    return (
        prod_meta
        .select(
            [
                "product_code",
                "rst_4_extended",
                "rst_4_market",
                "rst_4_market_sector",
                "rst_4_sub_market",
                "rst_4_trading_area",
            ]
        )
        .join(
            pur_rec,
            left_on="product_code",
            right_on="Product Code",
        )
        .select(
            [
                "product_code",
                "rst_4_extended",
                "rst_4_market",
                "rst_4_market_sector",
                "rst_4_sub_market",
                "rst_4_trading_area",
                "PurchaseId",
                "Period",
                "reported_volume_up",
                "volume_up",
                "volume_per",
            ]
        )
        .filter(pl.col("volume_up") > 0)
    )

def split_volumes(prod_tbl: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    prod_kg = prod_tbl.filter(pl.col('reported_volume_up') == "Kilos")
    prod_lt = prod_tbl.filter(pl.col('reported_volume_up') == "Litres")

    return prod_kg, prod_lt

def merge_nut(df: pl.DataFrame, nut_rec: pl.DataFrame) -> pl.DataFrame:
    return df.join(
        nut_rec,
        left_on=[pl.col("PurchaseId"), pl.col("Period")],
        right_on=[pl.col("Purchase Number"), pl.col("Purchase Period")],
        how="inner",
    )


def specific_gravity(prod_lt: pl.DataFrame, gravity: pl.DataFrame) -> pl.DataFrame:
    """Creates df with specific gravity values per category

    Args:
        prod_lt (pd.DataFrame): table of products with categories
        gravity (pd.DataFrame): lookup of specific gravity values per category

    Returns:
        pd.DataFrame: df with specific gravity values of products
    """
    return prod_lt.join(gravity, on="rst_4_extended")

def prod_litres(prod_tbl_liquids: pl.DataFrame, gravity: pl.DataFrame, nut_rec: pl.DataFrame):
    return merge_nut(
        specific_gravity(prod_tbl_liquids, gravity),
        nut_rec,
    )

def prod_kilos(prod_tbl_solids: pl.DataFrame, nut_rec: pl.DataFrame):
    """Returns dataframe with merged nutritional info for kg products (prod_kg_nut)"""
    return merge_nut(prod_tbl_solids, nut_rec)

def kcal_per_100g_drinks() -> pl.Expr:
    """Returns series with kcal per 100 g"""
    return pl.col("Energy KCal") / (10 * pl.col("volume_up") * pl.col("sg"))


def nut_per_100g_drinks(column: str) -> pl.Expr:
    """Returns series with macro per 100 g"""
    return 100 * (pl.col(column) / (pl.col("volume_up") * pl.col("sg")))

def drink_per_100g(prod_lt_nut: pl.DataFrame) -> pl.DataFrame:
    return (
        prod_lt_nut
        .with_columns(
            [
                kcal_per_100g_drinks().alias("kcal_per_100g"),
                nut_per_100g_drinks("Saturates KG").alias("sat_per_100g"),
                nut_per_100g_drinks("Protein KG").alias("prot_per_100g"),
                nut_per_100g_drinks("Sugar KG").alias("sug_per_100g"),
                nut_per_100g_drinks("Sodium KG").alias("sod_per_100g"),
                nut_per_100g_drinks("Fibre KG Flag").alias("fibre_per_100g"),
            ]
        )
    )

def kcal_per_100g_foods() -> pl.Expr:
    """Returns series with kcal per 100 g"""
    return pl.col("Energy KCal") / (pl.col("volume_up") * 10)


def nut_per_100g_foods(column: str) -> pl.Expr:
    """Returns series with macro per 100 g"""
    return 100 * (pl.col(column) / pl.col("volume_up"))


def food_per_100g(prod_kg_nut: pl.DataFrame) -> pl.DataFrame:
    """Assign new colums with standardised nutritional info and remove implausible values"""
    return (
        prod_kg_nut
        .with_columns(
            [
                kcal_per_100g_drinks().alias("kcal_per_100g"),
                nut_per_100g_drinks("Saturates KG").alias("sat_per_100g"),
                nut_per_100g_drinks("Protein KG").alias("prot_per_100g"),
                nut_per_100g_drinks("Sugar KG").alias("sug_per_100g"),
                nut_per_100g_drinks("Sodium KG").alias("sod_per_100g"),
                nut_per_100g_drinks("Fibre KG Flag").alias("fibre_per_100g"),
            ]
        )
        # remove implausible values
        .filter(pl.col("kcal_per_100g") < 900)
    )
# TODO: explore if these can be polarised
def assign_energy_score(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Calculates NPM energy density score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [
        -1,
        335,
        670,
        1005,
        1340,
        1675,
        2010,
        2345,
        2680,
        3015,
        3350,
        float("inf"),
    ]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name] * 4.184, bins=thresholds, labels=scores, right=True
    )
    return binned_cats

def assign_satf_score(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Calculates NPM saturated fats score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float("inf")]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats

def assign_sugars_score(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Calculates NPM sugars score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45, float("inf")]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats

def assign_sodium_score(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Calculates NPM sodium score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 90, 180, 270, 360, 450, 540, 630, 720, 810, 900, float("inf")]
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats

def assign_protein_score(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Calculates NPM proteins score using 2004-2005 thresholds."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [0, 1.6, 3.2, 4.8, 6.4, 8, float("inf")]
    scores = [0, 1, 2, 3, 4, 5]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats

def assign_fiber_score(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Use 2004-2005 NPM to assign C points for fiber (AOAC)."""
    if not (df[column_name].dtype == np.float64 or df[column_name].dtype == np.int64):
        raise TypeError("values should be in int64 or float64 format")
    thresholds = [
        0,
        0.9,
        1.9,
        2.8,
        3.7,
        4.7,
        float("inf"),
    ]  # NSP thresholds [0, 0.7, 1.4, 2.1, 2.8, 3.5]
    scores = [0, 1, 2, 3, 4, 5]
    binned_cats = pd.cut(
        df[column_name], bins=thresholds, labels=scores, right=True, include_lowest=True
    )
    return binned_cats

def assign_scores(df: pl.DataFrame) -> pl.DataFrame:
    """Assign columns with scores for each NPM factor"""
    prod_all = df.to_pandas()

    prod_all["energy_score"] = assign_energy_score(prod_all, "kcal_per_100g")
    prod_all["satf_score"] = assign_satf_score(prod_all, "sat_per_100g")
    prod_all["sug_score"] = assign_sugars_score(prod_all, "sug_per_100g")
    prod_all["sodium_score"] = assign_sodium_score(prod_all, "sod_per_100g")
    prod_all["protein_score"] = assign_protein_score(prod_all, "prot_per_100g")
    prod_all["fiber_score"] = assign_fiber_score(prod_all, "fibre_per_100g")

    return pl.from_pandas(prod_all)

def all_prod(prod_lt_nut: pl.DataFrame, prod_kg_nut: pl.DataFrame, fvn: pl.DataFrame) -> pl.DataFrame:
    """Concatenate kg and litre product dataframes and merge fruit veg and nut points"""
    return pl.concat([prod_lt_nut, prod_kg_nut]).join(
        fvn, on=["rst_4_trading_area", "rst_4_market_sector", "rst_4_market"]
    )

def scoring_df(#TODO: these are pipelines, explore putting this into a class to use initialise dataframes instead
    prod_meta_df: pl.DataFrame,
    pur_rec_df: pl.DataFrame,
    nut_rec_df: pl.DataFrame,
    gravity_df: pl.DataFrame,
    fvn_df: pl.DataFrame,
) -> pl.DataFrame:
    prod_tbl_df = clean_tbl(prod_meta_df, pur_rec_df)

    solids, liquids = split_volumes(prod_tbl_df)

    solids_nutrition = food_per_100g(
        prod_kilos(solids, nut_rec_df)
    )

    liquids_nutrition = drink_per_100g(
        prod_litres(liquids, gravity_df, nut_rec_df)
    )

    return assign_scores(
        all_prod(liquids_nutrition, solids_nutrition, fvn_df)
    ).select(
        [
            "product_code",
            "PurchaseId",
            "Period",
            "energy_score",
            "satf_score",
            "sug_score",
            "sodium_score",
            "protein_score",
            "fiber_score",
            "Score",
        ]
    )

def calculate_npm_score(#TODO: these are pipelines, explore putting this into a class to use initialise dataframes instead
    df: pl.DataFrame, a_points: str, fiber_col: str, protein_col: str, fvn_col: str
) -> pl.DataFrame:
    """Returns a df of purchases and NPM scores"""
    tbl = (
        df
        .with_columns(
            [
                pl.col(a_points).cast(int),
                pl.col(fiber_col).cast(float),
                pl.col(protein_col).cast(float),
                pl.col(fvn_col).cast(float),
            ]
        )
        .with_columns(
            pl.when(
                pl.col(a_points).sum().ge(11)
                & pl.col(fvn_col).ge(5)
            ).then(
                pl.col(a_points).sum()
                - pl.col(fiber_col)
                - pl.col(protein_col)
                - pl.col(fvn_col)
            ).otherwise(
                pl.col(a_points).sum()
                - pl.col(fiber_col)
                - pl.col(fvn_col)
            ).alias("npm_score")
        )
    )
    return tbl

def npm_score(#TODO: these are pipelines, explore putting this into a class to use initialise dataframes instead
    prod_meta_df: pl.DataFrame,
    pur_rec_df: pl.DataFrame,
    nut_rec_df: pl.DataFrame,
    gravity_df: pl.DataFrame,
    fvn_df: pl.DataFrame,
    a_points: str, fiber_col: str, protein_col: str, fvn_col: str
) -> pl.DataFrame:
    return calculate_npm_score(
        scoring_df(
            prod_meta_df,
            pur_rec_df,
            nut_rec_df,
            gravity_df,
            fvn_df
        ),
        a_points, fiber_col, protein_col, fvn_col
    )

def npm_score_unique(#TODO: these are pipelines, explore putting this into a class to use initialise dataframes instead
    prod_meta_df: pl.DataFrame,
    pur_rec_df: pl.DataFrame,
    nut_rec_df: pl.DataFrame,
    gravity_df: pl.DataFrame,
    fvn_df: pl.DataFrame,
    a_points: str, fiber_col: str, protein_col: str, fvn_col: str
) -> pl.DataFrame:
    return (
        npm_score(
            prod_meta_df,
            pur_rec_df,
            nut_rec_df,
            gravity_df,
            fvn_df,
            a_points, fiber_col, protein_col, fvn_col
        )
        .groupby("product_code")
        .agg(
            pl.col("npm_score").max()
        )
    )

class HFSS():
    def __init__(self, option: str) -> None:
        if not option:
            raise AttributeError("Missing parameter `option`: must be 'all' or 'prod_meta' or 'pur_rec' or 'nut_rec' or 'gravity' or 'fvn'")
        else:
            self.option = option
            functions = self._eval_option(self.option)
            if self.option == "all":
                with ThreadPoolExecutor() as executor:
                    results = executor.map(lambda func: func(), functions)
            else:
                setattr(self, f"{self.option}_df", functions)
            

    def _eval_option(self, option) -> Union[List[Callable], pl.DataFrame]:
        match self.option:
            case "all":
                return [
                    self._get_prod_meta_df,
                    self._get_pur_rec_df,
                    self._get_nut_rec_df,
                    self._get_gravity_df,
                    self._get_fvn_df,
                    self._get_scoring_df,
                ]
            case "prod_meta":
                return self._get_prod_meta_df()
            case "pur_rec":
                return self._get_pur_rec_df()
            case "nut_rec":
                return self._get_nut_rec_df()
            case "gravity":
                return self._get_gravity_df()
            case "fvn":
                return self._get_fvn_df()
            case _:
                raise AttributeError("Missing parameter `option`: must be 'all' or 'prod_meta' or 'pur_rec' or 'nut_rec' or 'gravity' or 'fvn'")
                
    #TODO: add local, and cloud getters
    def _get_prod_meta_df(self) -> pl.DataFrame:
        return pl.DataFrame()
    def _get_pur_rec_df(self) -> pl.DataFrame:
        return pl.DataFrame()
    def _get_nut_rec_df(self) -> pl.DataFrame:
        return pl.DataFrame()
    def _get_gravity_df(self) -> pl.DataFrame:
        return pl.DataFrame()
    def _get_fvn_df(self) -> pl.DataFrame:
        return pl.DataFrame()

    def _get_scoring_df(self) -> pl.DataFrame:
        prod_tbl_df = clean_tbl(self.prod_meta_df, self.pur_rec_df)

        solids, liquids = split_volumes(prod_tbl_df)

        solids_nutrition = food_per_100g(
            prod_kilos(solids, self.nut_rec_df)
        )

        liquids_nutrition = drink_per_100g(
            prod_litres(liquids, self.gravity_df, self.nut_rec_df)
        )

        return assign_scores(
            all_prod(liquids_nutrition, solids_nutrition, self.fvn_df)
        ).select(
            [
                "product_code",
                "PurchaseId",
                "Period",
                "energy_score",
                "satf_score",
                "sug_score",
                "sodium_score",
                "protein_score",
                "fiber_score",
                "Score",
            ]
        )

    def npm_score(self,
        a_points: str, fiber_col: str, protein_col: str, fvn_col: str,
    ) -> pl.DataFrame:
        """Returns a df of purchases and NPM scores"""
        return (
            self.scoring_df
            .with_columns(
                [
                    pl.col(a_points).cast(int),
                    pl.col(fiber_col).cast(float),
                    pl.col(protein_col).cast(float),
                    pl.col(fvn_col).cast(float),
                ]
            )
            .with_columns(
                pl.when(
                    pl.col(a_points).sum().ge(11)
                    & pl.col(fvn_col).ge(5)
                ).then(
                    pl.col(a_points).sum()
                    - pl.col(fiber_col)
                    - pl.col(protein_col)
                    - pl.col(fvn_col)
                ).otherwise(
                    pl.col(a_points).sum()
                    - pl.col(fiber_col)
                    - pl.col(fvn_col)
                ).alias("npm_score")
            )
        )

    def npm_score_unique(self,
        a_points: str, fiber_col: str, protein_col: str, fvn_col: str,
    ) -> pl.DataFrame:
        return (
            self.npm_score(
                a_points, fiber_col, protein_col, fvn_col
            )
            .groupby("product_code")
            .agg(
                pl.col("npm_score").max()
            )
        )