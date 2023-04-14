from ahl_targets import PROJECT_DIR
import pandas as pd


def gross_kcal(pur_nut) -> pd.Series:
    """Returns pd Series with total kcal sold at population level per instance"""
    return pur_nut["Energy KCal"] * pur_nut["Gross Up Weight"]


def kcal_sales(dat, col) -> pd.DataFrame:
    """Returns df aggregated according to col with sum of kcal purchased"""
    dat["gross_kcal"] = gross_kcal(dat)
    return dat.groupby(col)["gross_kcal"].sum().reset_index()
