from nesta_ds_utils.loading_saving.S3 import upload_obj, download_obj
from ahl_targets.utils import create_tables
from ahl_targets.getters import get_data
from ahl_targets import BUCKET_NAME

imd = download_obj(
    BUCKET_NAME,
    "in_home/data_outputs/hh_table_imd.csv",
    download_as="dataframe",
)

demog_coding = get_data.demog_coding()
demog_val = get_data.demog_val()
pan_mast = get_data.household_master()
demog_table = create_tables.hh_demographic_table(demog_coding, demog_val, pan_mast)

demog_table_imd = demog_table.reset_index().merge(
    imd[["Panel Id", "Index of Multiple Deprivation (IMD) Decile"]],
    on="Panel Id",
    how="left",
)

upload_obj(
    demog_table_imd,
    BUCKET_NAME,
    "in_home/processed/demog_with_imd.csv",
    kwargs_writing={"index": False},
)
