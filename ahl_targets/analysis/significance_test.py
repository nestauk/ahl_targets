# %%
# Import modules
from ahl_targets.pipeline import (
    product_transformation as product,
    energy_density as energy,
    hfss,
    stores_transformation as stores,
)
from ahl_targets.utils import create_tables as tables
from ahl_targets.getters import get_data
from ahl_targets.utils.plotting import configure_plots
from ahl_targets.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
import altair as alt
import pandas as pd
import logging
import statsmodels.formula.api as smf
from patsy.contrasts import Sum
import numpy as np

# %%
nut_recs = get_data.nutrition()
pur_recs = get_data.purchase_records_updated()
pur_rec_vol = get_data.purchase_records_volume()
store_coding = get_data.store_itemisation_coding()
store_lines = get_data.store_itemisation_lines()
prod_table = get_data.product_metadata()
prod_meas = get_data.product_measurement()
gravity = get_data.get_gravity()
fvn = get_data.get_fvn()
# %%
npm = hfss.npm_score_unique(
    prod_table,
    pur_rec_vol,
    gravity,
    nut_recs,
    fvn,
    hfss.a_points_cols(),
    "fiber_score",
    "protein_score",
    "Score",
)
# %%
store_levels = stores.taxonomy(
    store_coding,
    store_lines,
)


# %%
# merge with nutritional file

tbl_nut = tables.nutrition_merge(nut_recs, pur_recs, ["Energy KCal"]).merge(
    store_levels, left_on=["Store Code"], right_on=["store_id"]
)
# %%

tbl_nut = tbl_nut[
    [
        "Panel Id",
        "Product Code",
        "Store Code",
        "PurchaseId",
        "Period",
        "Gross Up Weight",
        "itemisation_level_1",
        "itemisation_level_2",
        "itemisation_level_3",
        "itemisation_level_4",
        "Energy KCal",
    ]
].copy()
# %%

# calculate contribution to total kcal sold at retailer for each product

tbl_nut["gross_kcal"] = tbl_nut["Gross Up Weight"] * tbl_nut["Energy KCal"]

# %%

# subset for testing

tbl_sub = tbl_nut.sample(n=100000, replace=False, random_state=1)
# %%
tbl_sub = tbl_nut
# %%

num = (
    tbl_sub.groupby(["itemisation_level_3", "Product Code"])["gross_kcal"]
    .sum()
    .reset_index(name="sum_kcal")
)

# %%
den = (
    tbl_sub.groupby(["itemisation_level_3"])["gross_kcal"]
    .sum()
    .reset_index(name="total")
)

# %%
# -> table with product/store rows

dat = num.merge(den, on=["itemisation_level_3"]).merge(
    npm, left_on=["Product Code"], right_on=["product_code"]
)
# %%

dat["share"] = dat["sum_kcal"] / dat["total"] * 100
# %%

# test from one NPM

sub_dat = dat[dat["npm_score"] == 10]


# %%

# mod = smf.ols("share ~ C(itemisation_level_3, Sum)", data = sub_dat).fit(cov_type = "cluster", cov_kwds={'groups' : sub_dat["itemisation_level_3"]})

# %%

mod = smf.ols("share ~ C(itemisation_level_3, Sum)", data=sub_dat).fit(cov_type="HC0")

# %%
p_value = pd.DataFrame(mod.pvalues, columns=["pvalue"]).reset_index()
coefs = pd.DataFrame(mod.params, columns=["coef"]).reset_index()
# %%
p_value["sig"] = np.where(p_value["pvalue"] < 0.05, 1, 0)
coefs["pos"] = np.where(coefs["coef"] > 0, 1, 0)

# %%

result = coefs.merge(p_value, on=["index"])

# %%
result["pos_sig"] = np.where(((result["sig"] == 1) & (result["pos"] == 1)), 1, 0)
result["neg_sig"] = np.where(((result["sig"] == 1) & (result["pos"] == 0)), 1, 0)
# %%
result
# %%
result["rescale_coef"] = result["coef"] * 100
# %%

ref_line = result[result["index"] == "Intercept"]

# %%
result = result[result["index"] != "Intercept"].copy()

result["store_type"] = result["index"].str[30:]

# %%
points = (
    alt.Chart(result, title="NPM = 10")
    .mark_circle(size=60)
    .encode(
        alt.Y(
            "store_type:N",
            sort=alt.EncodingSortField(field="rescale_coef", order="descending"),
        ),
        x="rescale_coef",
        color="sig:N",
    )
)

rules = alt.Chart(ref_line).mark_rule().encode(x="rescale_coef")
# %%
points + rules


# %%
