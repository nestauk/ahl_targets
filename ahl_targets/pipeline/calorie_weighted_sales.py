from ahl_targets import PROJECT_DIR
import pandas as pd


def gross_kcal(pur_nut: pd.DataFrame) -> pd.Series:
    """Returns pd Series with total kcal sold at population level per instance
    Args: pur_nut (pd.DataFrame): nutritional data
    Returns: pd.DataFrame: grossed up calorie dales"""
    return pur_nut["Energy KCal"] * pur_nut["Gross Up Weight"]


def kcal_sales(dat: pd.DataFrame, col: str) -> pd.DataFrame:
    """Returns df aggregated according to col with sum of kcal purchased
    Args:
        - dat (pd.DataFrame): dataframe with nutritional values and products
        - col (str): variables against which the aggregation happens
    Returns: pd.DataFrame: aggregated dataset by col returning sum of kcal sold"""
    dat["gross_kcal"] = gross_kcal(dat)
    return dat.groupby(col)["gross_kcal"].sum().reset_index()
