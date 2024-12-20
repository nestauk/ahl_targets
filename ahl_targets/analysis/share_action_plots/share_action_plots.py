"""
This file produces the following charts:
1) Distribution of converted NPM score by weighted sales before any interventions.
2) Distribution of converted NPM score by weighted sales after the npm simulation runs on the new data (to check it's similar to as expected).
3) Set the sales shifts to 0 so only reformulation contributes (maybe set them to something constant):
    a) Distribution of converted NPM score by weighted sales after the npm simulation has run.
    b) Distribution of converted NPM score by weighted sales after the HFSS simulation has run.

"""

# Imports
from nesta_ds_utils.loading_saving.S3 import download_obj
import pandas as pd
from ahl_targets import BUCKET_NAME
from boto3.s3.transfer import TransferConfig
import logging
from ahl_targets import PROJECT_DIR

# First run:
# What does the plot look like now?

# # Load in most up to date run of the npm model
# npm_model_output = download_obj(
#         BUCKET_NAME,
#         "in_home/processed/targets/oct_24_update/model_results_73.85.csv",
#         download_as="dataframe",
#     )

# # Load detailed output
# df = download_obj(
#         BUCKET_NAME,
#         f"in_home/processed/targets/oct_24_update/model_results_detailed_data_73.85.parquet",
#         download_as="dataframe",
#         kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
#     )

# Load in data without sales shifts
npm_model_output = download_obj(
    BUCKET_NAME,
    "in_home/processed/targets/share_action_custom_plots/model_results_41.32.csv",
    download_as="dataframe",
)

# Load detailed output
df = download_obj(
    BUCKET_NAME,
    f"in_home/processed/targets/share_action_custom_plots/model_results_detailed_data_41.32.parquet",
    download_as="dataframe",
    kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
)


logging.info("Data loaded.")

## Plot code to plot sales-weighted

# Apply converted npm scaling
df["converted_npm"] = (df["npm_score"] * -2) + 70
df["new_converted_npm"] = (df["new_npm"] * -2) + 70

# Define function to do the plots: weighted sales vs converted npm

# Plot should be a density plot

import matplotlib.pyplot as plt
import seaborn as sns


# This takes a while
def plot_weighted_sales_vs_converted_npm(
    df,
    x="converted_npm",
    weights="kg_w",
    title="Distribution of converted NPM score by weighted sales",
    bw_adjust=3,
    hfss_cutoff=62,
):
    # Initialise plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the kdensity plot
    sns.kdeplot(data=df, x=x, weights=weights, ax=ax, bw_adjust=bw_adjust)

    # Configure axes
    ax.set_xlim(left=0)
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


# This is faster as just uses a sample
# def plot_weighted_sales_vs_converted_npm_sample(df, title, sample_size=100000, bw_adjust = 5):
#     sample_df = df.sample(n=sample_size, random_state=42)
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.kdeplot(data = sample_df, x="converted_npm", weights="kg_w", ax=ax, bw_adjust=bw_adjust)
#     ax.set_title(title)
#     ax.set_xlabel("Converted NPM")
#     ax.set_ylabel("Weighted sales")
#     plt.show()

plot_weighted_sales_vs_converted_npm(
    df,
    title="Distribution of converted NPM score by weighted sales before npm model simulation",
)
plot_weighted_sales_vs_converted_npm(
    df,
    title="Distribution of converted NPM score by weighted sales after npm model simulation",
    x="new_converted_npm",
    weights="kg_w_new",
)
