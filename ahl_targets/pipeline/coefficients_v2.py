"""Re-calculating the coefficients for the new baseline file"""

from ahl_targets.getters import get_data, get_data_v2 as g2
import pandas as pd
import statsmodels.api as sm
from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME
from ahl_targets import PROJECT_DIR


if __name__ == "__main__":
    # store_data = get_data.model_data()
    # store_data = g2.new_model_data() # Need to merge npm onto this
    store_data = pd.read_csv(PROJECT_DIR / "outputs/outputs/df_npm.csv")
    reg_data = (
        store_data[["product_code", "npm_score", "ed", "rst_4_market_sector"]]
        .drop_duplicates()
        .dropna()
    )

    coefficients = []
    error = []
    stderror = []
    lowci = []
    highci = []

    # Iterate over unique categories and run the regression for each subset
    for category in reg_data["rst_4_market_sector"].unique():
        subset = reg_data[reg_data["rst_4_market_sector"] == category]

        # Fit the regression model
        X = subset["npm_score"]
        X = sm.add_constant(X)  # Add a constant term to include intercept
        y = subset["ed"]
        model = sm.OLS(y, X)
        results = model.fit()

        # Extract the intercept and coefficient values
        coefficient = results.params[1]

        # Add the coefficients to the DataFrame
        coefficients.append(
            {"rst_4_market_sector": category, "Coefficient": coefficient}
        )

        # Extract standard error
        std_error = results.bse[1]

        # Add the standard error to the DataFrame
        stderror.append({"rst_4_market_sector": category, "Standard Error": std_error})

        # Extract R squared
        rsquared = results.rsquared

        # Add the R squared to the DataFrame
        error.append({"rst_4_market_sector": category, "R Squared": rsquared})

        # Extract confidence intervals for coefficients
        low = results.conf_int(alpha=0.05).iloc[1, 0]
        high = results.conf_int(alpha=0.05).iloc[1, 1]

        lowci.append({"rst_4_market_sector": category, "Low CI": low})
        highci.append({"rst_4_market_sector": category, "High CI": high})

    # Print the coefficients table
    coefficients_df = pd.DataFrame(coefficients)

    # Print the error table
    error_df = pd.DataFrame(error)

    # Print the standard error table
    stderror_df = pd.DataFrame(stderror)

    # Print the confidence interval tables
    lowci_df = pd.DataFrame(lowci)
    highci_df = pd.DataFrame(highci)

    # combine tables
    combined_df = (
        coefficients_df.merge(stderror_df, on="rst_4_market_sector")
        .merge(error_df, on="rst_4_market_sector")
        .merge(lowci_df, on="rst_4_market_sector")
        .merge(highci_df, on="rst_4_market_sector")
        .drop_duplicates()
    )

    # Local save
    coefficients_df.to_csv(PROJECT_DIR / "outputs/coefficients_v2.csv", index=False)

    # upload to S3
    upload_obj(
        coefficients_df,
        BUCKET_NAME,
        "in_home/processed/targets/coefficients_v2.csv",
        kwargs_writing={"index": False},
    )

    # upload_obj(
    #     combined_df,
    #     BUCKET_NAME,
    #     "in_home/processed/targets/ed_npm_regression_output.csv",
    #     kwargs_writing={"index": False},
    # )
