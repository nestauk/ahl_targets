"""
This file produces the following charts:
1) Distribution of converted NPM score by weighted sales before any interventions.
2) Distribution of converted NPM score by weighted sales after the npm simulation runs on the new data (to check it's similar to as expected).
3) Set the sales shifts to 0 so only reformulation contributes (maybe set them to something constant):
    a) Distribution of converted NPM score by weighted sales after the npm simulation has run.
    b) Distribution of converted NPM score by weighted sales after the HFSS simulation has run.

Decision to make: Do we arbitrarily choose an iteration to plot, or do we average it somehow? Currently an iteration is just chosen.
"""

# Imports
from nesta_ds_utils.loading_saving.S3 import download_obj
import pandas as pd
from ahl_targets import BUCKET_NAME
from boto3.s3.transfer import TransferConfig
import logging
from ahl_targets import PROJECT_DIR
import matplotlib.pyplot as plt
import seaborn as sns

# Load in data without sales shifts (reformulation only)
npm_model_output = download_obj(
    BUCKET_NAME,
    "in_home/processed/targets/share_action_custom_plots/model_results_41.35.csv",
    download_as="dataframe",
)

# Load detailed output
df = download_obj(
    BUCKET_NAME,
    f"in_home/processed/targets/share_action_custom_plots/model_results_detailed_data_41.35.parquet",
    download_as="dataframe",
    kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
)

logging.info("Data loaded.")

## Plot code to plot sales-weighted

# Apply converted npm scaling
df["converted_npm"] = (df["npm_score"] * -2) + 70
df["new_converted_npm"] = (df["new_npm"] * -2) + 70

# For the purpose of these illustrative graphs - I have just taken one iteration to plot. There is a chance this is an outlier and we can revisit this if needed.
df = df[df["iteration"] == 85]


# Define function to do the plots: weighted sales vs converted npm
def plot_weighted_sales_vs_converted_npm(
    df,
    x="converted_npm",
    weights="kg_w",
    title="Distribution of converted NPM score by weighted sales",
    bw_adjust=1,
    hfss_cutoff=62,
):
    # Initialise plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the kdensity plot
    sns.kdeplot(data=df, x=x, weights=weights, ax=ax, bw_adjust=bw_adjust)

    # Configure axes
    ax.set_xlim(left=0)
    ax.set_ylim(top=0.04)
    ax.set_title(title)
    ax.set_xlabel("Converted NPM")
    ax.set_ylabel("Percentage of total kg sold")
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f"{tick * 100:.2f}%" for tick in y_ticks])

    # Add dotted line at hfss_cutoff
    ax.axvline(
        hfss_cutoff, color="r", linestyle="--", label=f"HFSS Cut-off ({hfss_cutoff})"
    )
    ax.legend()

    # Save plots
    plt.savefig(PROJECT_DIR / f"outputs/share_action_plots/{title}.png")

    plt.show()


# Plot (This can take a while for large datasets e.g. plotting multiple iterations)

plot_weighted_sales_vs_converted_npm(
    df,
    title="Prior distribution of converted NPM score by weighted sales",
)
plot_weighted_sales_vs_converted_npm(
    df,
    title="Distribution of converted NPM score by weighted sales after npm model simulation",
    x="new_converted_npm",
    weights="kg_w_new",
)

###### HFSS SIMULATION

# Load in the HFSS simulation data
hfss_model_output = download_obj(
    BUCKET_NAME,
    "in_home/processed/targets/share_action_custom_plots/hfss_model_results_41.31.csv",
    download_as="dataframe",
)

# Load detailed output
hfss_df = download_obj(
    BUCKET_NAME,
    f"in_home/processed/targets/share_action_custom_plots/hfss_model_results_detailed_data_41.31.parquet",
    download_as="dataframe",
    kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
)

logging.info("HFSS data loaded.")

## Plot code to plot sales-weighted

# Apply converted npm scaling
hfss_df["converted_npm"] = (hfss_df["npm_score"] * -2) + 70
hfss_df["new_converted_npm"] = (hfss_df["new_npm"] * -2) + 70

# For the purpose of these illustrative graphs - I have just taken one iteration to plot. There is a chance this is an outlier and we can revisit this if needed.
hfss_df = hfss_df[hfss_df["iteration"] == 55]

plot_weighted_sales_vs_converted_npm(
    hfss_df,
    title="Distribution of converted NPM score by weighted sales after hfss model simulation",
    x="new_converted_npm",
    weights="kg_w_new",
)
