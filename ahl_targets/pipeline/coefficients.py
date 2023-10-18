from ahl_targets.getters import get_data
import pandas as pd
import statsmodels.api as sm
from nesta_ds_utils.loading_saving.S3 import upload_obj
from ahl_targets import BUCKET_NAME


if __name__ == "__main__":
    store_data = get_data.model_data()

    coefficients = []

    # Iterate over unique categories and run the regression for each subset
    for category in store_data["rst_4_market_sector"].unique():
        subset = store_data[store_data["rst_4_market_sector"] == category]

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

    # Print the coefficients table
    coefficients_df = pd.DataFrame(coefficients)

    # upload to S3
    upload_obj(
        coefficients_df,
        BUCKET_NAME,
        "in_home/processed/targets/coefficients.csv",
        kwargs_writing={"index": False},
    )
