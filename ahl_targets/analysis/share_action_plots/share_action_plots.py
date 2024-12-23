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
import numpy as np
from ahl_targets import BUCKET_NAME
from boto3.s3.transfer import TransferConfig
import logging
from ahl_targets import PROJECT_DIR
import matplotlib.pyplot as plt
import seaborn as sns

# Load in data without sales shifts (reformulation only)
npm_model_output = download_obj(
    BUCKET_NAME,
    "in_home/processed/targets/share_action_custom_plots/model_results_41.36.csv",
    download_as="dataframe",
)

# Load detailed output
df = download_obj(
    BUCKET_NAME,
    f"in_home/processed/targets/share_action_custom_plots/model_results_detailed_data_41.36.parquet",
    download_as="dataframe",
    kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
)

logging.info("Data loaded.")

## Plot code to plot sales-weighted

# Apply converted npm scaling
df["converted_npm"] = (df["npm_score"] * -2) + 70
df["new_converted_npm"] = (df["new_npm"] * -2) + 70

# For the purpose of these illustrative graphs - I have just taken one iteration to plot. There is a chance this is an outlier and we can revisit this if needed.
df = df[df["iteration"] == 80]


# Define function to do the plots: weighted sales vs converted npm
def plot_weighted_sales_vs_converted_npm(
    df,
    x="converted_npm",
    weights="kg_w",
    title="Distribution of converted NPM score by weighted sales",
    bw_adjust=1,
    hfss_cutoff=62,
    zero_range=False,
):
    # Initialise plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Optional 0 of the range of reformulated values (used for HFSS plot to show how extreme this is, while still smoothing the rest of the data)
    if zero_range:
        # Create a kde plot
        kde = sns.kdeplot(data=df, x=x, weights=weights, ax=ax, bw_adjust=bw_adjust)

        # Extract the KDE values and positions
        line = ax.get_lines()[0]
        x_data, y_data = line.get_data()

        # Define the range where KDE values should be zeroed out
        zero_range = (54, 62)  # Range where all values are selected for reformulation

        # Modify the KDE values to zero within the specified range
        y_modified = np.where(
            (x_data > zero_range[0]) & (x_data < zero_range[1]), 0, y_data
        )

        # Clear the existing plot
        ax.clear()

        # Plot the modified KDE
        ax.plot(x_data, y_modified, label="NPM Distribution", color="blue")

    else:
        # Plot the kdensity plot
        sns.kdeplot(data=df, x=x, weights=weights, ax=ax, bw_adjust=bw_adjust)

    # Configure axes
    ax.set_xlim(left=0)
    ax.set_ylim(top=0.05)
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
    hfss_cutoff=70,  # New cut-off for reformulating
)

###### HFSS SIMULATION

# Load in the HFSS simulation data
hfss_model_output = download_obj(
    BUCKET_NAME,
    "in_home/processed/targets/share_action_custom_plots/hfss_model_results_43.51.csv",
    download_as="dataframe",
)

# Load detailed output
hfss_df = download_obj(
    BUCKET_NAME,
    f"in_home/processed/targets/share_action_custom_plots/hfss_model_results_detailed_data_43.51.parquet",
    download_as="dataframe",
    kwargs_boto={"Config": TransferConfig(io_chunksize=20947892)},
)

logging.info("HFSS data loaded.")

## Plot code to plot sales-weighted

# Apply converted npm scaling
hfss_df["converted_npm"] = (hfss_df["npm_score"] * -2) + 70
hfss_df["new_converted_npm"] = (hfss_df["new_npm"] * -2) + 70

# For the purpose of these illustrative graphs - I have just taken one iteration to plot. There is a chance this is an outlier and we can revisit this if needed.
hfss_df = hfss_df[hfss_df["iteration"] == 0]

plot_weighted_sales_vs_converted_npm(
    hfss_df,
    title="Distribution of converted NPM score by weighted sales after hfss model simulation",
    x="new_converted_npm",
    weights="kg_w_new",
    bw_adjust=0.8,
    zero_range=True,
)
