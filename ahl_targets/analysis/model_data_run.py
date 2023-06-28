from ahl_targets.pipeline import model_data
from ahl_targets.getters import get_data
from ahl_targets import PROJECT_DIR
from ahl_targets.pipeline import product_transformation as product

if __name__ == "__main__":
    pur_rec_vol = get_data.purchase_records_volume()
    nut_recs = get_data.nutrition()
    store_coding = get_data.store_itemisation_coding()
    store_lines = get_data.store_itemisation_lines()
    gravity = get_data.get_gravity()
    fvn = get_data.get_fvn()
    store_coding = get_data.store_itemisation_coding()
    store_lines = get_data.store_itemisation_lines()
    prod_table = get_data.product_metadata()
    npm = get_data.get_npm()

    store_data = model_data.make_data(
        prod_table, pur_rec_vol, gravity, nut_recs, fvn, store_coding, store_lines, npm
    )

    store_data = (
        store_data.pipe(product.type).pipe(product.in_scope).query("Quantity >0")
    )

    store_data.to_csv(
        PROJECT_DIR / "inputs/processed/target_model_data_full.csv", index=False
    )

    store_data[
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
            "is_food",
            "Energy KCal",
            "ed",
            "Gross Up Weight",
            "Quantity",
            "Store Code",
            "Spend",
            "store_id",
            "store_cat",
            "npm_score",
            "type",
            "in_scope",
        ]
    ].to_csv(PROJECT_DIR / "inputs/processed/target_model_data.csv", index=False)
