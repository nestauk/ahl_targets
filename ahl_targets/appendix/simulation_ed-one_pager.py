# %%

from ahl_targets.getters import get_data
from functools import reduce
import pandas as pd
from ahl_targets import PROJECT_DIR
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ahl_targets.utils.plotting import configure_plots
from ahl_targets.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
import altair as alt
from altair.expr import datum
from plotnine import (
    ggplot,
    geom_line,
    aes,
    # stat_smooth,
    facet_wrap,
    # geom_smooth,
    labs,
    geom_hline,
    # geom_density,
    geom_vline,
    # after_stat,
    geom_vline,
)


# %%
np.random.seed(42)

# %%

pd.set_option("display.max_columns", None)

# %%
# read data
store_data = get_data.model_data().compute()
prod_table = get_data.product_metadata()


# %%
def round_to_nearest_5(number):
    """
    Round a given number to the nearest multiple of 5.

    Args:
        number (int or float): The number to be rounded.

    Returns:
        int or float: The rounded number.
    """
    return round(number / 5) * 5


def weighted_kg_by_store(store_data):
    """aggregated df with volume weights

    Args:
        store_data (pd.DataFrame): merged data
    """
    store_data["cross"] = store_data["volume_up"] * store_data["Gross Up Weight"]
    out = (
        store_data.groupby(["product_code", "store_cat"])["cross"]
        .sum()
        .reset_index(name="total_kg")
    )
    out["kg_w"] = out["total_kg"] / out["total_kg"].sum()

    return out


def weighted_kcal_by_store(store_data):
    """aggregated df with volume weights

    Args:
        store_data (pd.DataFrame): merged data
    """
    store_data["cross"] = store_data["Energy KCal"] * store_data["Gross Up Weight"]
    out = (
        store_data.groupby(["product_code", "store_cat"])["cross"]
        .sum()
        .reset_index(name="total_kcal")
    )
    out["kcal_w"] = out["total_kcal"] / out["total_kcal"].sum()

    return out


def weighted_prod_by_store(store_data):
    """aggregated df with volume weights

    Args:
        store_data (pd.DataFrame): merged data
    """
    out = (
        store_data.groupby(["product_code", "store_cat"])["Gross Up Weight"]
        .sum()
        .reset_index(name="total_prod")
    )
    out["prod_w"] = out["total_prod"] / out["total_prod"].sum()

    return out


def unique_ed(store_data):
    """Unique ED by prodyuct

    Args:
        store_data (pd.DataFrame): marged data

    Returns:
        pd.DataFrame: aggregate df by product code with unique ED
    """
    return store_data.groupby(["product_code"])["ed"].mean().reset_index(name="ed")


def unique_kcal(store_data):
    """Unique KCAL by prodyuct

    Args:
        store_data (pd.DataFrame): marged data

    Returns:
        pd.DataFrame: aggregate df by product code with unique KCAL
    """
    return (
        (store_data["Energy KCal"] / store_data["Quantity"])
        .groupby(store_data["product_code"])
        .mean()
        .reset_index(name="kcal_unit")
    )


def unique_npm(store_data):
    """Unique NPM by prodyuct

    Args:
        store_data (pd.DataFrame): marged data

    Returns:
        pd.DataFrame: aggregate df by product code with unique NPM
    """

    return (
        store_data.groupby(["product_code"])["npm_score"]
        .mean()
        .reset_index(name="npm_score")
    )


def unique_spend(store_data):
    """Unique spend by prodyuct

    Args:
        store_data (pd.DataFrame): marged data

    Returns:
        pd.DataFrame: aggregate df by product code with unique NPM
    """
    return (
        (store_data["Spend"] / store_data["Quantity"])
        .groupby(store_data["product_code"])
        .mean()
        .reset_index(name="spend")
    )


def unique_hfss(store_data):
    """Unique HFSS by prodyuct

    Args:
        store_data (pd.DataFrame): marged data

    Returns:
        pd.DataFrame: aggregate df by product code with unique NPM
    """

    return (store_data.groupby(["product_code"])["in_scope"].max()).reset_index(
        name="hfss"
    )


def weighted_ed(store_data):
    """merged aggregated weight data

    Args:
        store_data (pd.DataFrame): merged data

    Returns:
        pd.DataFrame: merged aggregated weight data
    """
    return (
        weighted_kg_by_store(store_data)
        .merge(weighted_kcal_by_store(store_data), on=["product_code", "store_cat"])
        .merge(weighted_prod_by_store(store_data), on=["product_code", "store_cat"])
        .merge(unique_ed(store_data), on="product_code")
        .merge(unique_kcal(store_data), on="product_code")
        .merge(unique_spend(store_data), on="product_code")
    )


def weighted_npm(store_data):
    """merged aggregated weight data

    Args:
        store_data (pd.DataFrame): merged data

    Returns:
        pd.DataFrame: merged aggregated weight data
    """
    return (
        weighted_kg_by_store(store_data)
        .merge(weighted_kcal_by_store(store_data), on=["product_code", "store_cat"])
        .merge(weighted_prod_by_store(store_data), on=["product_code", "store_cat"])
        .merge(unique_npm(store_data), on="product_code")
        .merge(unique_ed(store_data), on="product_code")
        .merge(unique_kcal(store_data), on="product_code")
        .merge(unique_spend(store_data), on="product_code")
    )


def weighted_hfss(store_data):
    """merged aggregated weight data

    Args:
        store_data (pd.DataFrame): merged data

    Returns:
        pd.DataFrame: merged aggregated weight data
    """
    return (
        weighted_kg_by_store(store_data)
        .merge(weighted_kcal_by_store(store_data), on=["product_code", "store_cat"])
        .merge(weighted_prod_by_store(store_data), on=["product_code", "store_cat"])
        .merge(unique_npm(store_data), on="product_code")
        .merge(unique_ed(store_data), on="product_code")
        .merge(unique_kcal(store_data), on="product_code")
        .merge(unique_hfss(store_data), on="product_code")
        .merge(unique_spend(store_data), on="product_code")
    )


def prod_weight_g(w_store_data):
    """average weight of product in grames

    Args:
        w_store_data (pd.DataFrame): merged data

    Returns:
        pd.Series: average weight of product
    """
    return w_store_data["total_kg"] / w_store_data["total_prod"] * 1000


def npm_ed_table(store_data):
    """
    Returns a table of the mean value of the 'ed' column in the given DataFrame, grouped by the unique values in the 'npm_score' column.

    Args:
    - store_data (pandas.DataFrame): The DataFrame to group by and aggregate.

    Returns:
    - pandas.DataFrame: A new DataFrame containing the mean value of the 'ed' column for each unique value in the 'npm_score' column.
    """
    # Group the DataFrame by the 'npm_score' column and calculate the mean value of the 'ed' column for each group
    table = store_data.groupby(["npm_score"])["ed"].mean()

    # Reset the index of the resulting DataFrame to make 'npm_score' a column again
    table = table.reset_index()

    # Return the resulting table
    return table


# Define a function to perform calculations on the column
def apply_reduction_ed(store_weight, ed_reduction):
    # Modify the values of the column
    modified_column = np.where(
        store_weight["indicator_reform"] == 1,
        store_weight["ed"] - ed_reduction * store_weight["ed"] / 100,
        store_weight["ed"],
    )

    return modified_column


def plot_volume_weighted_ed(density_plot, file_name, path_png):
    density_plot["ed"].plot(kind="hist", weights=density_plot["kg_w"], alpha=0.2)
    density_plot["new_ed"].plot(
        kind="hist", weights=density_plot["kg_w_new"], alpha=0.2
    )
    plt.axvline(
        x=(density_plot["ed"] * density_plot["kg_w"]).sum()
        / density_plot["kg_w"].sum(),
        color="b",
        linestyle="--",
    )
    plt.axvline(
        x=(density_plot["new_ed"] * density_plot["kg_w_new"]).sum()
        / density_plot["kg_w_new"].sum(),
        color="r",
        linestyle="--",
    )
    plt.text(
        (density_plot["ed"] * density_plot["kg_w"]).sum() / density_plot["kg_w"].sum()
        + 10,
        0.35,
        "Mean before: "
        + str(
            round(
                (density_plot["ed"] * density_plot["kg_w"]).sum()
                / density_plot["kg_w"].sum(),
                1,
            )
        )
        + "kcal/day",
    )
    plt.text(
        (density_plot["new_ed"] * density_plot["kg_w_new"]).sum()
        / density_plot["kg_w_new"].sum()
        + 10,
        0.25,
        "Mean after: "
        + str(
            round(
                (density_plot["new_ed"] * density_plot["kg_w_new"]).sum()
                / density_plot["kg_w_new"].sum(),
                1,
            )
        )
        + "kcal/day",
    )
    plt.title("Volume weighted energy density")
    plt.xlabel("Volume weighted energy density")
    plt.legend(loc="upper right")
    path = path_png + "volume_weighted_ed_" + file_name + ".png"
    plt.savefig(PROJECT_DIR / path, bbox_inches="tight")
    plt.clf()


def percent_ed_target(avg_retailer):
    bar_plot = (
        alt.Chart(avg_retailer)
        .mark_bar()
        .encode(
            x=alt.X("diff_percentage", title="Percentage Difference"),
            y=alt.Y("store_cat", title="Store"),
            color=alt.condition(
                alt.datum.diff_percentage > 0,
                alt.value("#0000FF"),  # The positive color
                alt.value("#FDB633"),  # The negative color
            ),
        )
        .properties(height=300, width=600)
    )

    return configure_plots(
        bar_plot,
        "Percentage difference between store average energy density and the absolute target",
        "",
        16,
        14,
        14,
    )


def plot_avg_ed_target(baseline_abs_targ):
    # Reshape the data using melt()
    melted_data = baseline_abs_targ.melt(
        id_vars="Store",
        value_vars=["Baseline", "Absolute Target"],
        var_name="Variable",
        value_name="Value",
    )

    # Create the plot
    base = alt.Chart(baseline_abs_targ).encode(
        y=alt.Y("Store:N", title="Stores"),
    )

    # Points and lines
    points = base.mark_circle(size=50).encode(
        x=alt.X(
            "Baseline:Q",
            scale=alt.Scale(
                domain=[160, (baseline_abs_targ["Baseline"].max() + 5).round(0)]
            ),
            axis=alt.Axis(title="Energy Density", grid=False),
        ),
        color="Store",
    )

    # Add points to the plot
    line = (
        alt.Chart(baseline_abs_targ)
        .mark_rule(color="black", size=2, strokeDash=[2, 2])
        .encode(
            x=alt.X(
                "Absolute Target",
                scale=alt.Scale(
                    domain=[160, (baseline_abs_targ["Baseline"].max() + 5).round(0)]
                ),
            ),
        )
    )

    lines = (
        alt.Chart(melted_data)
        .mark_line(color="blue")
        .encode(x=alt.X("Value:Q"), y="Store", color="Store")
    )

    # Add points to the plot
    shapes = (
        lines.mark_point()
        .encode(
            opacity=alt.value(1),
            shape=alt.Shape(
                "Variable:N", scale=alt.Scale(range=["circle", "triangle"])
            ),
        )
        .transform_calculate(category="datum.Variable")
    )

    # Annotation
    text = (
        alt.Chart(
            pd.DataFrame({"value": [baseline_abs_targ["Absolute Target"].loc[0]]})
        )
        .mark_text(
            align="right", baseline="top", dx=-10, dy=-10, fontSize=16, color="black"
        )
        .encode(
            x="value:Q",
            text=alt.value(
                "Target = " + str(baseline_abs_targ["Absolute Target"].loc[0].round(1))
            ),
        )
    )

    # Annotation for 'ed' values
    ed_text_left = (
        alt.Chart(baseline_abs_targ)
        .transform_filter(
            alt.datum["Baseline"] < baseline_abs_targ["Absolute Target"].loc[0]
        )
        .mark_text(baseline="middle", fontSize=12, color="black", dx=-20)
        .encode(
            x=alt.X("Baseline:Q", title=""),
            y=alt.Y("Store:N", title=""),
            text=alt.Text("Baseline:Q", format=".1f"),
        )
    )
    ed_text_right = (
        alt.Chart(baseline_abs_targ)
        .transform_filter(
            alt.datum["Baseline"] > baseline_abs_targ["Absolute Target"].loc[0]
        )
        .mark_text(baseline="middle", fontSize=12, color="black", dx=20)
        .encode(
            x=alt.X("Baseline:Q", title=""),
            y=alt.Y("Store:N", title=""),
            text=alt.Text("Baseline:Q", format=".1f"),
        )
    )

    chart = (
        points + lines + shapes + line + text + ed_text_left + ed_text_right
    ).properties(width=500, height=400)
    return configure_plots(
        chart,
        "Store baseline compared to the absolute target",
        "",
        18,
        14,
        15,
    )


def avg_ed_comp_reduced(source):
    # Reshape the data using melt()
    melted_data = source.melt(
        id_vars="store",
        value_vars=["Baseline", "Relative Target"],
        var_name="Variable",
        value_name="Value",
    )

    # Create the plot
    plot = (
        alt.Chart(melted_data)
        .mark_line()
        .encode(
            x=alt.X(
                "Value",
                scale=alt.Scale(
                    domain=[
                        (melted_data.Value.min() - 10).round(0),
                        (melted_data.Value.max() + 10).round(0),
                    ]
                ),
                axis=alt.Axis(title="Energy Density", grid=False),
            ),
            y=alt.Y("store", title="Store"),
            color="store",
        )
        .properties(width=450, height=400)
    )

    # Add points to the plot
    points = (
        plot.mark_point()
        .encode(
            opacity=alt.value(1),
            shape=alt.Shape(
                "Variable:N", scale=alt.Scale(range=["triangle", "circle"])
            ),
        )
        .transform_calculate(category="datum.Variable")
    )

    # Add annotations to the points
    left_annotations = (
        points.mark_text(
            align="right",
            dx=-10,  # Shift the 'reduced values' annotation to the left
            fontSize=12,
        )
        .encode(
            text=alt.Text("Value:Q", format=".1f"),
            x=alt.X(
                "Value:Q",
                scale=alt.Scale(
                    domain=[
                        (melted_data.Value.min() - 10).round(0),
                        (melted_data.Value.max() + 10).round(0),
                    ]
                ),
            ),
            y="store",
        )
        .transform_filter(
            alt.FieldEqualPredicate(field="Variable", equal="Relative Target")
        )
    )

    right_annotations = (
        points.mark_text(
            align="left",
            dx=10,  # Shift the 'energy density' annotation to the right
            fontSize=12,
        )
        .encode(
            text=alt.Text("Value:Q", format=".1f"),
            x=alt.X(
                "Value:Q",
                scale=alt.Scale(
                    domain=[
                        (melted_data.Value.min() - 10).round(0),
                        (melted_data.Value.max() + 10).round(0),
                    ]
                ),
            ),
            y="store",
        )
        .transform_filter(alt.FieldEqualPredicate(field="Variable", equal="Baseline"))
    )

    # Combine the chart, points, and annotations
    layered_plot = alt.layer(plot, points, left_annotations, right_annotations)

    return configure_plots(
        layered_plot,
        "Relative target compared to the baseline across stores",
        "",
        18,
        14,
        15,
    )


# Workshop plots (stores removed)
def avg_ed_comp_reduced_workshop(source):
    # Reshape the data using melt()
    melted_data = source.melt(
        id_vars="store",
        value_vars=["Baseline", "Relative Target"],
        var_name="Variable",
        value_name="Value",
    )

    # Create the plot
    plot = (
        alt.Chart(melted_data)
        .mark_line()
        .encode(
            x=alt.X(
                "Value",
                scale=alt.Scale(
                    domain=[
                        (melted_data.Value.min() - 10).round(0),
                        (melted_data.Value.max() + 10).round(0),
                    ]
                ),
                axis=alt.Axis(
                    title="Sales weighted average calorie density (kcal/100g) across whole portfolio",
                    grid=False,
                ),
            ),
            y=alt.Y(
                "store",
                title="Store (>1.5% market share)",
                axis=alt.Axis(labels=False),
                sort=None,
            ),
            color=alt.Color("store", legend=None),
        )
        .properties(width=450, height=400)
    )

    # Add points to the plot
    points = (
        plot.mark_point()
        .encode(
            opacity=alt.value(1),
            shape=alt.Shape(
                "Variable:N", scale=alt.Scale(range=["triangle", "circle"])
            ),
        )
        .transform_calculate(category="datum.Variable")
    )

    # Add annotations to the points
    left_annotations = (
        points.mark_text(
            align="right",
            dx=-10,  # Shift the 'reduced values' annotation to the left
            fontSize=12,
        )
        .encode(
            text=alt.Text("Value:Q", format=".1f"),
            x=alt.X(
                "Value:Q",
                scale=alt.Scale(
                    domain=[
                        (melted_data.Value.min() - 10).round(0),
                        (melted_data.Value.max() + 10).round(0),
                    ]
                ),
            ),
            y=alt.Y(
                "store",
                title="Store (>1.5% market share)",
                axis=alt.Axis(labels=False),
                sort=None,
            ),
        )
        .transform_filter(
            alt.FieldEqualPredicate(field="Variable", equal="Relative Target")
        )
    )

    right_annotations = (
        points.mark_text(
            align="left",
            dx=10,  # Shift the 'energy density' annotation to the right
            fontSize=12,
        )
        .encode(
            text=alt.Text("Value:Q", format=".1f"),
            x=alt.X(
                "Value:Q",
                scale=alt.Scale(
                    domain=[
                        (melted_data.Value.min() - 10).round(0),
                        (melted_data.Value.max() + 10).round(0),
                    ]
                ),
            ),
            y=alt.Y(
                "store",
                title="Store (>1.5% market share)",
                axis=alt.Axis(labels=False),
                sort=None,
            ),
        )
        .transform_filter(alt.FieldEqualPredicate(field="Variable", equal="Baseline"))
    )

    # Combine the chart, points, and annotations
    layered_plot = alt.layer(plot, points, left_annotations, right_annotations)

    return configure_plots(
        layered_plot,
        "Relative target compared to the baseline across stores",
        "",
        18,
        14,
        15,
    )


def percent_ed_target_workshop(avg_retailer):
    bar_plot = (
        alt.Chart(avg_retailer)
        .mark_bar()
        .encode(
            x=alt.X("diff_percentage", title="Percentage Difference"),
            y=alt.Y(
                "store_cat",
                title="Store (>1.5% market share)",
                axis=alt.Axis(labels=False),
                sort=None,
            ),
            color=alt.condition(
                alt.datum.diff_percentage > 0,
                alt.value("#0000FF"),  # The positive color
                alt.value("#FDB633"),  # The negative color
            ),
        )
        .properties(height=300, width=600)
    )

    return configure_plots(
        bar_plot,
        "Percentage difference between store average energy density and the absolute target",
        "",
        16,
        14,
        14,
    )


# %%


def plot_avg_ed_target_workshop(baseline_abs_targ):
    # Reshape the data using melt()
    melted_data = baseline_abs_targ.melt(
        id_vars="store_letter",
        value_vars=["Baseline", "Target"],
        var_name="Variable",
        value_name="Value",
    )
    # Create the plot
    base = alt.Chart(baseline_abs_targ).encode(
        y=alt.Y(
            "store_letter:N",
            title="",
            sort=None,
        ),
    )
    # Points and lines
    points = base.mark_circle(size=1).encode(
        x=alt.X(
            "Baseline:Q",
            scale=alt.Scale(
                domain=[160, (baseline_abs_targ["Baseline"].max() + 5).round(0)]
            ),
            axis=alt.Axis(
                title="Sales weighted average calorie density (kcal/100g) across whole portfolio",
                grid=False,
            ),
        ),
        color=alt.Color("store_letter"),  # , sort="Baseline:Q"),
    )

    # Add points to the plot
    line = (
        alt.Chart(baseline_abs_targ)
        .mark_rule(color="black", size=2, strokeDash=[2, 2])
        .encode(
            x=alt.X(
                "Target",
                scale=alt.Scale(
                    domain=[160, (baseline_abs_targ["Baseline"].max() + 5).round(0)]
                ),
            ),
        )
    )

    lines = (
        alt.Chart(melted_data)
        .mark_line(color="blue")
        .encode(
            x=alt.X("Value:Q"),
            y=alt.Y("store_letter", sort=None),
            color=alt.Color("store_letter", legend=None),
        )
    )

    # Add points to the plot
    shapes = (
        lines.mark_point()
        .encode(
            opacity=alt.value(1),
            shape=alt.Shape(
                "Variable:N",
                scale=alt.Scale(range=["circle", "triangle"]),
                title=None,
                legend=alt.Legend(
                    orient="none",
                    legendX=400,
                    legendY=200,
                    direction="vertical",
                    titleAnchor="middle",
                ),
            ),
        )
        .transform_calculate(category="datum.Variable")
    )

    # Annotation
    text = (
        alt.Chart(pd.DataFrame({"value": [int(baseline_abs_targ["Target"].loc[0])]}))
        .mark_text(
            align="right", baseline="top", dx=-10, dy=-10, fontSize=16, color="black"
        )
        .encode(
            x="value:Q",
            text=alt.value(
                "Target " + "\u2264 " + str(int(baseline_abs_targ["Target"].loc[0]))
            ),
        )
    )

    # Annotation for 'ed' values
    ed_text_left = (
        alt.Chart(baseline_abs_targ)
        .transform_filter(alt.datum["Baseline"] < baseline_abs_targ["Target"].loc[0])
        .mark_text(baseline="middle", fontSize=12, color="black", dx=-20)
        .encode(
            x=alt.X("Baseline:Q", title=""),
            y=alt.Y("store_letter:N", title="", sort=None),
            text=alt.Text("Baseline:Q", format=".1f"),
        )
    )
    ed_text_right = (
        alt.Chart(baseline_abs_targ)
        .transform_filter(alt.datum["Baseline"] >= baseline_abs_targ["Target"].loc[0])
        .mark_text(baseline="middle", fontSize=12, color="black", dx=20)
        .encode(
            x=alt.X("Baseline:Q", title=""),
            y=alt.Y("store_letter:N", title="", sort=None),
            text=alt.Text("Baseline:Q", format=".0f"),
        )
    )

    chart = (
        points + lines + shapes + line + text + ed_text_left + ed_text_right
    ).properties(width=500, height=500)
    return configure_plots(
        chart,
        "",
        "",
        18,
        14,
        15,
    )


# %%


def plot_density(plt_df_sub, full_results):
    chart = (
        alt.Chart(plt_df_sub)
        .transform_density(
            "share", as_=["size", "density"], groupby=["when"], bandwidth=50
        )
        .mark_line()
        .encode(
            x=alt.X(
                "size:Q",
                axis=alt.Axis(
                    title="Sales Weighted Average Energy Density (kcal/100g)"
                ),
            ),
            y=alt.Y("density:Q", axis=alt.Axis(title="Weighted Sales (%)", format="%")),
            color=alt.Color("when:N", legend=alt.Legend(title="")),
        )
    )

    share_before = ((full_results["ed"] * full_results["kg_w"])).sum() / full_results[
        "kg_w"
    ].sum()
    share_after = (
        (full_results["new_ed"] * full_results["kg_w_new"])
    ).sum() / full_results["kg_w_new"].sum()

    vertical_lines = [share_before, share_after]
    mean_values = [0.01, 0.02]
    label_position = [share_before + 200, share_after]

    # Create a DataFrame for the vertical lines, mean values, and line styles
    labels_df = pd.DataFrame({"x": vertical_lines, "mean_value": mean_values})

    labels_pos = pd.DataFrame({"x": label_position, "mean_value": mean_values})

    labels_pos["label"] = [
        "Mean before:" + str(round(share_before, 0)),
        "Mean after:" + str(round(share_after, 0)),
    ]  # Example labels, replace with desired labels
    labels_df["line_style"] = [
        "solid",
        "dashed",
    ]  # Example line styles, replace with desired styles
    labels_pos["offset"] = [0.002, 0.0015]

    layered_chart = alt.layer(
        chart,
        alt.Chart(labels_df)
        .mark_rule()
        .encode(
            x="x:Q",
            color=alt.ColorValue("black"),
            strokeDash=alt.StrokeDash("line_style:N", legend=None),
        ),
        alt.Chart(labels_pos)
        .mark_text(align="right", baseline="bottom", fontSize=12)
        .encode(
            x="x:Q", y=alt.Y("offset:Q"), text="label", color=alt.ColorValue("black")
        ),
    ).properties(height=300, width=600)

    return configure_plots(
        layered_chart,
        "Distribution of Sales Weighted Average Energy Density",
        "",
        16,
        14,
        14,
    )


# %%


def annex_chart_kcal(df):
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            y=alt.X("kcal_pp_new:Q", scale=alt.Scale(domain=[1400, 1700])),
            x="ed_reduction",
            color="sales_change_low",
        )
    )
    rule = alt.Chart(df).mark_rule(color="red").encode(y="mean(kcal_pp_baseline):Q")

    return configure_plots(
        chart + rule,
        "",
        "",
        16,
        14,
        14,
    )


# %%


def annex_chart_spend(df):
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            y=alt.X("spend_new:Q", scale=alt.Scale(domain=[15, 20])),
            x="ed_reduction",
            color="sales_change_low",
        )
    )
    rule = alt.Chart(df).mark_rule(color="red").encode(y="mean(spend_baseline):Q")

    return configure_plots(
        chart + rule,
        "",
        "",
        16,
        14,
        14,
    )


# %%


def create_outputs_scenarios(option, file_name):
    # Create paths
    path_png = "outputs/figures/png/scenarios/ed/" + file_name + "/"
    path_svg = "outputs/figures/svg/scenarios/ed/" + file_name + "/"
    path_html = "outputs/figures/html/scenarios/ed/" + file_name + "/"
    Path(PROJECT_DIR / path_png).mkdir(parents=True, exist_ok=True)
    Path(PROJECT_DIR / path_svg).mkdir(parents=True, exist_ok=True)
    Path(PROJECT_DIR / path_html).mkdir(parents=True, exist_ok=True)
    # Input scenarios here
    full_results = results_data_df[
        (results_data_df["ed_reduction"] == option[0])
        & (results_data_df["sales_change_high"] == option[1])
        & (results_data_df["sales_change_low"] == option[2])
    ]

    # A line for each one pager
    print_row = suitable[
        [
            "sales_change_high",
            "sales_change_low",
            "ed_reduction",
            "mean_ed_kg_new",
            "mean_ed_kcal_new",
            "mean_ed_kg_baseline",
            "mean_ed_kcal_baseline",
            "kcal_pp_baseline",
            "kcal_pp_new",
            "spend_baseline",
            "spend_new",
            "mean_ed_kg_diff_percentage",
            "mean_ed_kcal_diff_percentage",
            "kcal_pp_diff_percentage",
            "total_prod_diff_percentage",
            "spend_diff_percentage",
            "kcal_diff",
        ]
    ]
    opt_df = (
        print_row[
            (print_row["ed_reduction"] == option[0])
            & (print_row["sales_change_high"] == option[1])
            & (print_row["sales_change_low"] == option[2])
        ].T
    ).reset_index()
    opt_df.columns = ["Metric", "Value"]

    scenario_outputs = opt_df[
        opt_df["Metric"].isin(
            [
                "mean_ed_kg_diff_percentage",
                "kcal_diff",
                "spend_diff_percentage",
                "mean_ed_kg_diff_percentage",
                "mean_ed_kg_new",
            ]
        )
    ].copy()
    rename_dict = {
        "kcal_diff": "Kcal per capita reduction",
        "spend_diff_percentage": "Average weekly spend per person change",
        "mean_ed_kg_diff_percentage": "Relative target",
        "mean_ed_kg_new": "Absolute target",
    }

    scenario_outputs["Metric"].replace(rename_dict, inplace=True)

    path = "outputs/data/scenarios/ed/metric_table_suitable_" + file_name + ".csv"
    opt_df.to_csv(PROJECT_DIR / path, index=False)
    path = "outputs/data/scenarios/ed/scenario_outputs_" + file_name + ".csv"
    scenario_outputs.to_csv(PROJECT_DIR / path, index=False)

    grouped = full_results.groupby(
        [
            "product_code",
            "store_cat",
            "product_share_reform",
            "product_share_sale",
            "ed_reduction",
            "sales_change_high",
            "sales_change_low",
        ]
    )
    density_plot = grouped.mean().reset_index()
    plot_volume_weighted_ed(density_plot, file_name, path_png)

    # from store_weight, generate the average ED weighted by kg for each retailer
    avg_retailer = (
        (store_weight["kg_w"] * store_weight["ed"])
        .groupby(store_weight["store_cat"])
        .sum()
        / store_weight["kg_w"].groupby(store_weight["store_cat"]).sum()
    ).reset_index(name="ed")
    avg_retailer["weight"] = (
        store_weight.groupby(["store_cat"])["kg_w"].sum().reset_index()["kg_w"]
    )
    avg_retailer["target"] = (
        density_plot["new_ed"] * density_plot["kg_w_new"]
    ).sum() / density_plot["kg_w_new"].sum()
    avg_retailer["diff"] = avg_retailer["ed"] - avg_retailer["target"]
    avg_retailer["diff_percentage"] = 100 * (
        avg_retailer["target"] / avg_retailer["ed"] - 1
    )
    baseline_abs_targ = avg_retailer[["store_cat", "target", "ed"]].copy()
    baseline_abs_targ.columns = ["Store", "Absolute Target", "Baseline"]

    perc_plot = percent_ed_target(avg_retailer)
    avg_ed_targ = plot_avg_ed_target(baseline_abs_targ)

    avg_ed_targ_work = plot_avg_ed_target_workshop(baseline_abs_targ)

    save_altair(
        perc_plot,
        "scenarios/ed/" + file_name + "/perc_plot_" + file_name,
        driver=webdr,
    )
    save_altair(
        avg_ed_targ,
        "scenarios/ed/" + file_name + "/avg_ed_targ_" + file_name,
        driver=webdr,
    )
    save_altair(
        avg_ed_targ_work,
        "scenarios/ed/" + file_name + "/avg_ed_targ_workshop" + file_name,
        driver=webdr,
    )

    values = avg_retailer["ed"]
    weights = avg_retailer["weight"]
    target_average = avg_retailer["target"].mean()

    current_average = np.average(values, weights=weights)
    percentage_reduction = (current_average - target_average) / current_average * 100

    reduced_values = values * (1 - percentage_reduction / 100)
    reductions = values - reduced_values
    reduction_percentages = reductions / values * 100
    weighted_average = np.average(reduced_values, weights=weights)

    print("Reduced values:", reduced_values)
    print("Weighted average:", weighted_average)
    print("Current average:", current_average)
    print("Target average:", target_average)

    print("Overall Percentage reduction needed: {:.2f}%".format(percentage_reduction))
    for i in range(len(values)):
        print("Value {}: {:.2f}% reduction".format(i + 1, reduction_percentages[i]))

    if np.isclose(weighted_average, target_average):
        print("The target average is the same as the calculated weighted average.")
    else:
        print("The target average is different from the calculated weighted average.")

    ed_reduced_df = pd.concat(
        [avg_retailer[["store_cat", "ed", "target"]], reduced_values], axis=1
    )
    ed_reduced_df.columns = ["store", "Baseline", "absolute target", "Relative Target"]

    average_ed_reduced = avg_ed_comp_reduced(
        ed_reduced_df[["store", "Baseline", "Relative Target"]]
    )
    save_altair(
        average_ed_reduced,
        "scenarios/ed/" + file_name + "/avg_ed_reduced_" + file_name,
        driver=webdr,
    )

    ed_reduced_df.columns = [
        "Store",
        "Average Energy Density",
        "Absolute Target",
        "Relative Target",
    ]
    # Save tables
    path = "outputs/data/scenarios/ed/avg_ed_reduced" + file_name + ".csv"
    ed_reduced_df.to_csv(PROJECT_DIR / path, index=False)
    path = "outputs/data/scenarios/ed/avg_retailer" + file_name + ".csv"
    avg_retailer.to_csv(PROJECT_DIR / path, index=False)


# %%
store_weight = weighted_ed(store_data)
store_weight["prod_weight_g"] = store_weight.pipe(prod_weight_g)


# %%
# assumptions:
# 1 - reformulation (lowering ED) only happens in products that have at least 400 kcal/100g (high ED)
# 2 - sales changes reflect a switch between high ED to low ED (with cut off 400kcal/100g). The percentage is calculated as a proportion of current sales so whilst a nominal x% is taken for different products in absolute numbers this can mean different number of products


# %%
# version 3

num_iterations = 20
product_share_reform_values = [0.5]  # share of products that are reformulated
product_share_sale_values = [1]  # share of products whose sales are shifted
ed_reduction_values = [
    0,
    2.5,
    5,
    7.5,
    10,
    12.5,
]  # percentage reduction in energy density e.g. 1 indicates a 1% reduction in energy density
high_sales_change_values = [12.5]  # percentage shift in sales
low_sales_change_values = [2.5, 5, 7.5, 10]  # percentage shift in sales
cutoff = 400  # cut off for high energy density


# %%
results = []
results_data = []


# Nested loop to iterate through different values of product_share and ed_reduction
for product_share_reform in product_share_reform_values:
    for product_share_sale in product_share_sale_values:
        for ed_reduction in ed_reduction_values:
            for sales_change_high in high_sales_change_values:
                for sales_change_low in low_sales_change_values:
                    # Repeat the code num_iterations times
                    for _ in range(num_iterations):
                        # split into high and low ed
                        ed_cut = store_weight["ed"] >= cutoff
                        high_ed = store_weight[ed_cut].copy()
                        low_ed = store_weight[~ed_cut].copy()

                        # generate list of products to reformulate
                        # unique_products = pd.DataFrame(
                        #    store_weight["product_code"].unique(),
                        #    columns=["product_code"],
                        # )
                        unique_products = pd.DataFrame(
                            store_weight[(store_weight["ed"] >= cutoff)][
                                "product_code"
                            ].unique(),
                            columns=["product_code"],
                        )
                        unique_products["indicator_reform"] = np.random.choice(
                            [0, 1],
                            size=len(unique_products),
                            p=[1 - product_share_reform, product_share_reform],
                        )

                        # calculation in high ed products
                        high_ed = high_ed.merge(
                            unique_products, on="product_code", how="left"
                        )
                        high_ed["indicator_sale"] = np.random.choice(
                            [0, 1],
                            size=len(high_ed),
                            p=[1 - product_share_sale, product_share_sale],
                        )
                        high_ed["new_total_prod"] = np.where(
                            high_ed["indicator_sale"] == 1,
                            high_ed["total_prod"] * (1 - sales_change_high / 100),
                            high_ed["total_prod"],
                        )

                        # calculations in low ed products
                        low_ed["indicator_reform"] = 0
                        low_ed["indicator_sale"] = np.random.choice(
                            [0, 1],
                            size=len(low_ed),
                            p=[1 - product_share_sale, product_share_sale],
                        )
                        low_ed["new_total_prod"] = np.where(
                            low_ed["indicator_sale"] == 1,
                            low_ed["total_prod"] * (1 + sales_change_low / 100),
                            low_ed["total_prod"],
                        )

                        randomised = pd.concat([high_ed, low_ed], ignore_index=True)
                        randomised["new_total_kg"] = (
                            randomised["new_total_prod"]
                            * randomised["prod_weight_g"]
                            / 1000
                        )
                        randomised["new_ed"] = randomised.pipe(
                            apply_reduction_ed, ed_reduction
                        )
                        randomised["new_kcal_tot"] = (
                            randomised["new_ed"]
                            / 100
                            * randomised["prod_weight_g"]
                            * randomised["new_total_prod"]
                        )
                        randomised["kcal_w_new"] = (
                            randomised["new_kcal_tot"]
                            / randomised["new_kcal_tot"].sum()
                        )
                        randomised["kg_w_new"] = (
                            randomised["new_total_kg"]
                            / randomised["new_total_kg"].sum()
                        )

                        mean_ed_kg_new = (
                            randomised["kg_w_new"] * randomised["new_ed"]
                        ).sum()
                        mean_ed_kcal_new = (
                            randomised["kcal_w_new"] * randomised["new_ed"]
                        ).sum()

                        mean_ed_kg_baseline = (
                            randomised["kg_w"] * randomised["ed"]
                        ).sum()
                        mean_ed_kcal_baseline = (
                            randomised["kcal_w"] * randomised["ed"]
                        ).sum()

                        kcal_pp_baseline = (
                            randomised["total_kcal"].sum() / 66000000 / 365
                        )
                        kcal_pp_new = randomised["new_kcal_tot"].sum() / 66000000 / 365

                        total_prod_baseline = randomised["total_prod"].sum()
                        total_prod_new = randomised["new_total_prod"].sum()

                        kcal_high_baseline = randomised[randomised["ed"] >= cutoff][
                            "total_kcal"
                        ].sum()
                        kcal_high_new = randomised[randomised["new_ed"] >= cutoff][
                            "new_kcal_tot"
                        ].sum()

                        kcal_low_baseline = randomised[randomised["ed"] < cutoff][
                            "total_kcal"
                        ].sum()
                        kcal_low_new = randomised[randomised["new_ed"] < cutoff][
                            "new_kcal_tot"
                        ].sum()

                        spend_baseline = (
                            ((randomised["total_prod"] * randomised["spend"]).sum())
                            / 66000000
                            / 52
                        )
                        spend_new = (
                            ((randomised["new_total_prod"] * randomised["spend"]).sum())
                            / 66000000
                            / 52
                        )

                        iteration = _

                        # Append the results to the list
                        results.append(
                            {
                                "product_share_reform": product_share_reform,
                                "product_share_sale": product_share_sale,
                                "sales_change_high": sales_change_high,
                                "sales_change_low": sales_change_low,
                                "ed_reduction": ed_reduction,
                                "mean_ed_kg_new": mean_ed_kg_new,
                                "mean_ed_kcal_new": mean_ed_kcal_new,
                                "mean_ed_kg_baseline": mean_ed_kg_baseline,
                                "mean_ed_kcal_baseline": mean_ed_kcal_baseline,
                                "kcal_pp_baseline": kcal_pp_baseline,
                                "kcal_pp_new": kcal_pp_new,
                                "total_prod_baseline": total_prod_baseline,
                                "total_prod_new": total_prod_new,
                                "spend_baseline": spend_baseline,
                                "spend_new": spend_new,
                            }
                        )

                        results_data.append(
                            randomised.assign(
                                product_share_reform=product_share_reform,
                                product_share_sale=product_share_sale,
                                ed_reduction=ed_reduction,
                                sales_change_high=sales_change_high,
                                sales_change_low=sales_change_low,
                                iteration=iteration,
                            )
                        )


# Create the DataFrame from the list of results
results_df = pd.DataFrame(results)

results_data_df = pd.concat(results_data, ignore_index=True)


# %%
# note that when all products are selected (e.g. shares are 1) - all iterations are the same
avg = (
    results_df.groupby(
        [
            "product_share_reform",
            "product_share_sale",
            "sales_change_high",
            "sales_change_low",
            "ed_reduction",
        ]
    )[
        [
            "mean_ed_kg_new",
            "mean_ed_kcal_new",
            "mean_ed_kg_baseline",
            "mean_ed_kcal_baseline",
            "kcal_pp_baseline",
            "kcal_pp_new",
            "total_prod_baseline",
            "total_prod_new",
            "spend_baseline",
            "spend_new",
        ]
    ]
    .mean()
    .reset_index()
)

# %%
# note that when all products are selected (e.g. shares are 1) - all iterations are the same
sd = (
    results_df.groupby(
        [
            "product_share_reform",
            "product_share_sale",
            "sales_change_high",
            "sales_change_low",
            "ed_reduction",
        ]
    )[
        [
            "mean_ed_kg_new",
            "kcal_pp_new",
            "spend_new",
        ]
    ]
    .std()
    .reset_index()
)

# %%
# Extract columns with the suffix '_baseline'
baseline_columns = avg.filter(like="_baseline")

# Extract columns with the suffix '_new'
new_columns = avg.filter(like="_new")

baseline_columns.columns = baseline_columns.columns.str.replace("_baseline", "")
new_columns.columns = new_columns.columns.str.replace("_new", "")

result = (new_columns - baseline_columns) / baseline_columns * 100
df = pd.concat([avg, result.add_suffix("_diff_percentage")], axis=1)
df["kcal_diff"] = df["kcal_pp_new"] - df["kcal_pp_baseline"]


# %%
df.to_csv(PROJECT_DIR / "outputs/reports/simulation_ED.csv", index=False)


# %%

webdr = google_chrome_driver_setup()

save_altair(
    annex_chart_kcal(df),
    "annex/energy_density_kcal",
    driver=webdr,
)

# %%

webdr = google_chrome_driver_setup()

save_altair(
    annex_chart_spend(df),
    "annex/energy_density_spend",
    driver=webdr,
)

# %%
# suitable scenarios
suitable = df[
    (df["mean_ed_kg_diff_percentage"] < 0)
    & (df["kcal_pp_new"] <= 1567)
    & (df["spend_diff_percentage"] >= -1)
]


# %%
# Create new sub-folders
Path(PROJECT_DIR / "outputs/data/scenarios/ed/").mkdir(parents=True, exist_ok=True)
Path(PROJECT_DIR / "outputs/figures/png/scenarios/").mkdir(parents=True, exist_ok=True)
Path(PROJECT_DIR / "outputs/figures/html/scenarios/").mkdir(parents=True, exist_ok=True)
Path(PROJECT_DIR / "outputs/figures/svg/scenarios/").mkdir(parents=True, exist_ok=True)

# %%
options = suitable[
    ["ed_reduction", "sales_change_high", "sales_change_low"]
].values.tolist()

# %%
options[3]

# %%
# Load web-driver
webdr = google_chrome_driver_setup()

# x3 examples
create_outputs_scenarios(options[3], "125_125_5")
create_outputs_scenarios(options[7], "10_15_5")
create_outputs_scenarios(options[0], "5_125_25")

# %%
# Save file
suitable.to_csv(
    PROJECT_DIR / "outputs/data/scenarios/ed/suitable_scenarios.csv", index=False
)

# %%

# Workshop plots (stores removed)

# Absolute target plots
avg_retailer = pd.read_csv(
    PROJECT_DIR / "outputs/data/scenarios/ed/avg_retailer10_15_5.csv"
)

# %%


store_letters = {
    "Total Iceland": "Store A",
    "The Co-Operative": "Store B",
    "Total Asda": "Store C",
    "Lidl": "Store D",
    "Aldi": "Store E",
    "Total Sainsbury's": "Store F",
    "Ocado Internet": "Store G",
    "Total Waitrose": "Store H",
    "Total Morrisons": "Store I",
    "Total Tesco": "Store J",
    "Aldi": "Store K",
    "Total Marks & Spencer": "Store L",
}
# %%

avg_retailer["store_letter"] = avg_retailer["store_cat"].map(store_letters)

# %%

# Sort the data based on category order
category_order = list(
    avg_retailer.sort_values(by="diff", ascending=False)["store_letter"]
)
avg_retailer["store_cat"] = pd.Categorical(avg_retailer["store_letter"], category_order)
avg_retailer = avg_retailer.sort_values("store_letter")
perc_plot_work = percent_ed_target_workshop(avg_retailer)

baseline_abs_targ = avg_retailer[["store_letter", "target", "ed"]].copy()
baseline_abs_targ.columns = ["store_letter", "Target", "Baseline"]
# Sort the data based on category order
category_order = list(
    baseline_abs_targ.sort_values(by="Baseline", ascending=False)["store_letter"]
)
baseline_abs_targ["store_letter"] = pd.Categorical(
    baseline_abs_targ["store_letter"], category_order
)
baseline_abs_targ = baseline_abs_targ.sort_values("store_letter")
baseline_abs_targ["Target"] = round_to_nearest_5(baseline_abs_targ["Target"])
baseline_abs_targ["Baseline"] = round_to_nearest_5(baseline_abs_targ["Baseline"])

# %%

avg_ed_targ_work = plot_avg_ed_target_workshop(baseline_abs_targ)
avg_ed_targ_work
# %%

webdr = google_chrome_driver_setup()

save_altair(
    avg_ed_targ_work,
    "scenarios/ed/workshop/avg_ed_target_10_15_5",
    driver=webdr,
)

# %%

# Relative target plot
df_reduced = pd.read_csv(
    PROJECT_DIR / "outputs/data/scenarios/ed/avg_ed_reduced10_15_5.csv"
)
df_reduced.columns = ["store", "Baseline", "absolute target", "Relative Target"]
# Sort the data based on category order
category_order = list(
    df_reduced.sort_values(by="Relative Target", ascending=False)["store"]
)
df_reduced["store"] = pd.Categorical(df_reduced["store"], category_order)
df_reduced = df_reduced.sort_values("store")
average_ed_reduced_work = avg_ed_comp_reduced_workshop(
    df_reduced[["store", "Baseline", "Relative Target"]]
)

# Save plots
# Create new sub-folders
Path(PROJECT_DIR / "outputs/figures/png/scenarios/ed/workshop/").mkdir(
    parents=True, exist_ok=True
)
Path(PROJECT_DIR / "outputs/figures/html/scenarios/ed/workshop/").mkdir(
    parents=True, exist_ok=True
)
Path(PROJECT_DIR / "outputs/figures/svg/scenarios/ed/workshop/").mkdir(
    parents=True, exist_ok=True
)
# %%

# Load web-driver
webdr = google_chrome_driver_setup()
save_altair(
    perc_plot_work,
    "scenarios/ed/workshop/perc_plot_10_15_5",
    driver=webdr,
)
save_altair(
    avg_ed_targ_work,
    "scenarios/ed/workshop/avg_ed_target_10_15_5",
    driver=webdr,
)
save_altair(
    average_ed_reduced_work,
    "scenarios/ed/workshop/avg_ed_reduced_10_15_5",
    driver=webdr,
)


# %%

# %%
# density_125_125_5
option = options[3]


full_results = results_data_df[
    (results_data_df["ed_reduction"] == option[0])
    & (results_data_df["sales_change_high"] == option[1])
    & (results_data_df["sales_change_low"] == option[2])
]

print_row = suitable[
    [
        "sales_change_high",
        "sales_change_low",
        "ed_reduction",
        "mean_ed_kg_new",
        "mean_ed_kcal_new",
        "mean_ed_kg_baseline",
        "mean_ed_kcal_baseline",
        "kcal_pp_baseline",
        "kcal_pp_new",
        "spend_baseline",
        "spend_new",
        "mean_ed_kg_diff_percentage",
        "mean_ed_kcal_diff_percentage",
        "kcal_pp_diff_percentage",
        "total_prod_diff_percentage",
        "spend_diff_percentage",
        "kcal_diff",
    ]
]

opt_df = (
    print_row[
        (print_row["ed_reduction"] == option[0])
        & (print_row["sales_change_high"] == option[1])
        & (print_row["sales_change_low"] == option[2])
    ].T
).reset_index()

opt_df.columns = ["Metric", "Value"]


scenario_outputs = opt_df[
    opt_df["Metric"].isin(
        [
            "mean_ed_kg_diff_percentage",
            "kcal_diff",
            "spend_diff_percentage",
            "mean_ed_kg_diff_percentage",
            "mean_ed_kg_new",
        ]
    )
].copy()


rename_dict = {
    "kcal_diff": "Kcal per capita reduction",
    "spend_diff_percentage": "Average weekly spend per person change",
    "mean_ed_kg_diff_percentage": "Relative target",
    "mean_ed_kg_new": "Absolute target",
}

baseline = (
    (full_results["ed"] * full_results["kg_w"])
    .groupby(full_results["product_code"])
    .sum()
    / full_results.groupby(["product_code"])["kg_w"].sum()
).reset_index(name="share")

target = (
    (full_results["new_ed"] * full_results["kg_w_new"])
    .groupby(full_results["product_code"])
    .sum()
    / full_results.groupby(["product_code"])["kg_w_new"].sum()
).reset_index(name="share")

plt_df = pd.concat([baseline.assign(when="baseline"), target.assign(when="target")])

plt_df_sub = plt_df.sample(5000)
density_125_125_5 = plot_density(plt_df_sub, full_results)
webdr = google_chrome_driver_setup()
save_altair(density_125_125_5, "scenarios/ed/workshop/density_125_125_5", driver=webdr)
# %%
# density_10_15_5
option = options[7]


full_results = results_data_df[
    (results_data_df["ed_reduction"] == option[0])
    & (results_data_df["sales_change_high"] == option[1])
    & (results_data_df["sales_change_low"] == option[2])
]

print_row = suitable[
    [
        "sales_change_high",
        "sales_change_low",
        "ed_reduction",
        "mean_ed_kg_new",
        "mean_ed_kcal_new",
        "mean_ed_kg_baseline",
        "mean_ed_kcal_baseline",
        "kcal_pp_baseline",
        "kcal_pp_new",
        "spend_baseline",
        "spend_new",
        "mean_ed_kg_diff_percentage",
        "mean_ed_kcal_diff_percentage",
        "kcal_pp_diff_percentage",
        "total_prod_diff_percentage",
        "spend_diff_percentage",
        "kcal_diff",
    ]
]

opt_df = (
    print_row[
        (print_row["ed_reduction"] == option[0])
        & (print_row["sales_change_high"] == option[1])
        & (print_row["sales_change_low"] == option[2])
    ].T
).reset_index()

opt_df.columns = ["Metric", "Value"]


scenario_outputs = opt_df[
    opt_df["Metric"].isin(
        [
            "mean_ed_kg_diff_percentage",
            "kcal_diff",
            "spend_diff_percentage",
            "mean_ed_kg_diff_percentage",
            "mean_ed_kg_new",
        ]
    )
].copy()


rename_dict = {
    "kcal_diff": "Kcal per capita reduction",
    "spend_diff_percentage": "Average weekly spend per person change",
    "mean_ed_kg_diff_percentage": "Relative target",
    "mean_ed_kg_new": "Absolute target",
}

baseline = (
    (full_results["ed"] * full_results["kg_w"])
    .groupby(full_results["product_code"])
    .sum()
    / full_results.groupby(["product_code"])["kg_w"].sum()
).reset_index(name="share")

target = (
    (full_results["new_ed"] * full_results["kg_w_new"])
    .groupby(full_results["product_code"])
    .sum()
    / full_results.groupby(["product_code"])["kg_w_new"].sum()
).reset_index(name="share")

plt_df = pd.concat([baseline.assign(when="baseline"), target.assign(when="target")])

plt_df_sub = plt_df.sample(5000)
density_10_15_5 = plot_density(plt_df_sub, full_results)
webdr = google_chrome_driver_setup()
save_altair(density_10_15_5, "scenarios/ed/workshop/density_10_15_5", driver=webdr)
# %%
# density_5_125_25
option = options[0]


full_results = results_data_df[
    (results_data_df["ed_reduction"] == option[0])
    & (results_data_df["sales_change_high"] == option[1])
    & (results_data_df["sales_change_low"] == option[2])
]

print_row = suitable[
    [
        "sales_change_high",
        "sales_change_low",
        "ed_reduction",
        "mean_ed_kg_new",
        "mean_ed_kcal_new",
        "mean_ed_kg_baseline",
        "mean_ed_kcal_baseline",
        "kcal_pp_baseline",
        "kcal_pp_new",
        "spend_baseline",
        "spend_new",
        "mean_ed_kg_diff_percentage",
        "mean_ed_kcal_diff_percentage",
        "kcal_pp_diff_percentage",
        "total_prod_diff_percentage",
        "spend_diff_percentage",
        "kcal_diff",
    ]
]

opt_df = (
    print_row[
        (print_row["ed_reduction"] == option[0])
        & (print_row["sales_change_high"] == option[1])
        & (print_row["sales_change_low"] == option[2])
    ].T
).reset_index()

opt_df.columns = ["Metric", "Value"]


scenario_outputs = opt_df[
    opt_df["Metric"].isin(
        [
            "mean_ed_kg_diff_percentage",
            "kcal_diff",
            "spend_diff_percentage",
            "mean_ed_kg_diff_percentage",
            "mean_ed_kg_new",
        ]
    )
].copy()


rename_dict = {
    "kcal_diff": "Kcal per capita reduction",
    "spend_diff_percentage": "Average weekly spend per person change",
    "mean_ed_kg_diff_percentage": "Relative target",
    "mean_ed_kg_new": "Absolute target",
}

baseline = (
    (full_results["ed"] * full_results["kg_w"])
    .groupby(full_results["product_code"])
    .sum()
    / full_results.groupby(["product_code"])["kg_w"].sum()
).reset_index(name="share")

target = (
    (full_results["new_ed"] * full_results["kg_w_new"])
    .groupby(full_results["product_code"])
    .sum()
    / full_results.groupby(["product_code"])["kg_w_new"].sum()
).reset_index(name="share")

plt_df = pd.concat([baseline.assign(when="baseline"), target.assign(when="target")])

plt_df_sub = plt_df.sample(5000)
density_5_125_25 = plot_density(plt_df_sub, full_results)
webdr = google_chrome_driver_setup()
save_altair(density_5_125_25, "scenarios/ed/workshop/density_5_125_25", driver=webdr)
# %%
