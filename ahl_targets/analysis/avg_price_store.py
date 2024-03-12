from ahl_targets.getters import get_data
from ahl_targets.pipeline import stores_transformation as stores
from ahl_targets import BUCKET_NAME
from nesta_ds_utils.loading_saving.S3 import upload_obj


if __name__ == "__main__":
    pur_rec_store = get_data.model_data()

    # Calculate the weighted average price per store (wieghted by volume)
    pur_rec_store["volume_weighted"] = (
        pur_rec_store["volume_up"] * pur_rec_store["Gross Up Weight"]
    )
    weighted_purchases_store_volume = (
        pur_rec_store.groupby("store_cat")
        .apply(
            lambda x: (x["Spend"] * x["volume_weighted"]).sum()
            / x["volume_weighted"].sum()
        )
        .reset_index()
    )
    weighted_purchases_store_volume.columns = ["store_cat", "Weighted_Avg_Price_Volume"]

    # Save to S3

    upload_obj(
        weighted_purchases_store_volume,
        BUCKET_NAME,
        "in_home/data_outputs/weighted_purchases_store_volume.csv",
        kwargs_writing={"index": False, "encoding": "utf-8"},
    )
