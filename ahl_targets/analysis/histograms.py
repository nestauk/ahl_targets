from ahl_targets.pipeline import model_data
from ahl_targets.getters import get_data
from functools import reduce
import pandas as pd
from ahl_targets import PROJECT_DIR
import matplotlib.pyplot as plt


prod_table = get_data.product_metadata()
pur_rec_vol = get_data.purchase_records_volume()
nut_recs = get_data.nutrition()
store_coding = get_data.store_itemisation_coding()
store_lines = get_data.store_itemisation_lines()
gravity = get_data.get_gravity()
fvn = get_data.get_fvn()
store_coding = get_data.store_itemisation_coding()
store_lines = get_data.store_itemisation_lines()


if __name__ == "__main__":
    store_data = model_data.make_data(
        prod_table, pur_rec_vol, gravity, nut_recs, fvn, store_coding, store_lines
    )

    weighted_data = model_data.weighted(store_data)

    # unweighted distribution
    weighted_data["ed"].plot(kind="hist")
    plt.axvline(
        x=weighted_data["ed"].mean(), color="r", linestyle="--"
    )  # Replace `value` with the desired position of the line
    plt.text(
        weighted_data["ed"].mean() + 10,
        16500,
        "Mean: " + str(round(weighted_data["ed"].mean(), 1)),
    )
    plt.xlabel("Energy (kcal/100g)")
    plt.title("Energy Density Distribution")
    plt.show(block=False)
    plt.savefig(
        PROJECT_DIR / "outputs/figures/png/histogram_unweighted.png",
        bbox_inches="tight",
    )

    # weighted by kcal
    weighted_data["ed"].plot(kind="hist", weights=weighted_data["kcal_w"])
    # Add a vertical line
    plt.axvline(
        x=(weighted_data["ed"] * weighted_data["kcal_w"]).sum()
        / weighted_data["kcal_w"].sum(),
        color="r",
        linestyle="--",
    )  # Replace `value` with the desired position of the line
    plt.axvline(
        x=0.9
        * (
            (weighted_data["ed"] * weighted_data["kcal_w"]).sum()
            / weighted_data["kcal_w"].sum()
        ),
        color="b",
        linestyle="--",
    )  # Replace `value` with the desired position of the line
    plt.show(block=False)
    plt.savefig(
        PROJECT_DIR / "outputs/figures/png/histogram_kcal_weight.png",
        bbox_inches="tight",
    )

    # weighted by kg
    weighted_data["ed"].plot(kind="hist", weights=weighted_data["kg_w"])
    # Add a vertical line
    plt.axvline(
        x=(weighted_data["ed"] * weighted_data["kg_w"]).sum()
        / weighted_data["kg_w"].sum(),
        color="r",
        linestyle="--",
    )  # Replace `value` with the desired position of the line
    plt.axvline(
        x=0.9
        * (
            (weighted_data["ed"] * weighted_data["kg_w"]).sum()
            / weighted_data["kg_w"].sum()
        ),
        color="b",
        linestyle="--",
    )  # Replace `value` with the desired position of the line
    plt.show(block=False)
    plt.savefig(
        PROJECT_DIR / "outputs/figures/png/histogram_kg_weight.png", bbox_inches="tight"
    )

    # weighted by prod
    weighted_data["ed"].plot(kind="hist", weights=weighted_data["prod_w"])
    # Add a vertical line
    plt.axvline(
        x=(weighted_data["ed"] * weighted_data["prod_w"]).sum()
        / weighted_data["prod_w"].sum(),
        color="r",
        linestyle="--",
    )  # Replace `value` with the desired position of the line
    plt.axvline(
        x=0.9
        * (
            (weighted_data["ed"] * weighted_data["prod_w"]).sum()
            / weighted_data["prod_w"].sum()
        ),
        color="b",
        linestyle="--",
    )  # Replace `value` with the desired position of the line
    plt.show(block=False)
    plt.savefig(
        PROJECT_DIR / "outputs/figures/png/histogram_prod_weight.png",
        bbox_inches="tight",
    )
