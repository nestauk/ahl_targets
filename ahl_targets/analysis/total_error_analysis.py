from ahl_targets.getters.get_data import full_npm, model_data
import pandas as pd
from ahl_targets.getters.simulated_outcomes import npm_agg, npm_robustness
from ahl_targets import PROJECT_DIR
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import seaborn as sns
from ahl_targets.utils import simulation_utils as su


df = full_npm()
model = model_data()

keep_products = model.product_code.unique().tolist()

df = df[df.product_code.isin(keep_products)]

# NPM

df.product_code.nunique()

df_grouped = df.groupby("product_code")["npm_score"].nunique()

df_grouped.value_counts(normalize=True)

# volume

model.reported_volume.value_counts(normalize=True)

model.reported_volume.value_counts()

model.reported_volume_up.value_counts()


# randomisation

model_result = npm_agg()

model_result["npm_base_r"] = -2 * model_result["mean_npm_kg_baseline"] + 70
model_result["npm_new_r"] = -2 * model_result["mean_npm_kg_new"] + 70

model_result["kcal_diff"] = (
    model_result["kcal_pp_new"] - model_result["kcal_pp_baseline"]
)
model_result["spend_diff"] = model_result["spend_new"] - model_result["spend_baseline"]
model_result["spend_diff_pp"] = (
    model_result["spend_diff"] / model_result["spend_baseline"] * 100
)

model_result["npm_diff"] = (
    model_result["mean_npm_kg_new"] - model_result["mean_npm_kg_baseline"]
)
model_result["npm_r_diff"] = model_result["npm_new_r"] - model_result["npm_base_r"]


result_df = model_result[
    (model_result["npm_reduction"] == 3)
    & (model_result["sales_change_high"] == 10.5)
    & (model_result["sales_change_low"] == 9)
]

model_result["npm_diff"].describe()
model_result["kcal_diff"].describe()
model_result["npm_r_diff"].describe()
model_result["spend_diff"].describe()


# Assuming `result_df` is already defined and contains the relevant data

# Plotting the histogram
plt.hist(result_df["kcal_diff"], bins=10, edgecolor="black")

# Adding labels and title
plt.xlabel("kcal_diff")
plt.ylabel("Frequency")
plt.title("Histogram of kcal_diff")

# Displaying the histogram
plt.show()


# Plotting the histogram
plt.hist(result_df["npm_diff"], bins=10, edgecolor="black")

# Adding labels and title
plt.xlabel("npm_diff")
plt.ylabel("Frequency")
plt.title("Histogram of npm_diff")

# Displaying the histogram
plt.show()

# Plotting the histogram
plt.hist(result_df["npm_r_diff"], bins=10, edgecolor="black")

# Adding labels and title
plt.xlabel("npm_r_diff")
plt.ylabel("Frequency")
plt.title("Histogram of npm_r_diff")

# Displaying the histogram
plt.show()

model_agg = (
    model_result.groupby(
        [
            "product_share_reform",
            "product_share_sale",
            "sales_change_high",
            "sales_change_low",
            "npm_reduction",
        ]
    )[
        "kcal_pp_new",
        "kcal_pp_baseline",
        "spend_new",
        "spend_baseline",
        "kcal_diff",
        "spend_diff",
        "spend_diff_pp",
        "npm_r_diff",
        "npm_new_r",
        "npm_base_r",
    ]
    .mean()
    .reset_index()
)


# 95% confidence intervals

to_calc = result_df["kcal_diff"].tolist()

st.t.interval(0.95, df=len(to_calc) - 1, loc=np.mean(to_calc), scale=st.sem(to_calc))

to_calc = result_df["npm_diff"].tolist()

st.t.interval(0.95, df=len(to_calc) - 1, loc=np.mean(to_calc), scale=st.sem(to_calc))

to_calc = result_df["spend_diff"].tolist()

st.t.interval(0.95, df=len(to_calc) - 1, loc=np.mean(to_calc), scale=st.sem(to_calc))

to_calc = result_df["npm_r_diff"].tolist()

st.t.interval(0.95, df=len(to_calc) - 1, loc=np.mean(to_calc), scale=st.sem(to_calc))


# average per retailer

store_data = model_data()

store_weight_npm = su.weighted_npm(store_data)

store_weight_npm["npm_score_r"] = -2 * store_weight_npm["npm_score"] + 70


avg_retailer = (
    (
        (store_weight_npm["kg_w"] * store_weight_npm["npm_score_r"])
        .groupby(store_weight_npm["store_cat"])
        .sum()
        / store_weight_npm["kg_w"].groupby(store_weight_npm["store_cat"]).sum()
    )
    .reset_index(name="npm")
    .sort_values(by="npm", ascending=False)
)


# NPM - ED regression
reg_data = store_data[
    ["product_code", "npm_score", "ed", "rst_4_market_sector"]
].drop_duplicates()

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
    coefficients.append({"rst_4_market_sector": category, "Coefficient": coefficient})

    # Extract standard error
    std_error = results.bse[1]

    # Create scatter plot with regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(x=X.iloc[:, 1], y=y, ci=None, line_kws={"color": "red"})
    plt.title(category)  # Set the category name as the title

    # Set the minimum and maximum values for the x and y axes
    plt.xlim(-10, 30)
    plt.ylim(0, 900)

    # Set the ticks for the x and y axes
    plt.xticks(np.arange(-10, 31, 5))
    plt.yticks(np.arange(0, 901, 100))

    plt.savefig(PROJECT_DIR / "outputs/figures/png/regression/" f"plot_{category}.png")

plt.close("all")


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
    coefficients.append({"rst_4_market_sector": category, "Coefficient": coefficient})

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


# error distribution
npm_error = npm_robustness()

npm_error["npm_base_r"] = -2 * npm_error["mean_npm_kg_baseline"] + 70
npm_error["npm_new_r"] = -2 * npm_error["mean_npm_kg_new"] + 70

npm_error["kcal_diff"] = npm_error["kcal_pp_new"] - npm_error["kcal_pp_baseline"]
npm_error["spend_diff"] = npm_error["spend_new"] - npm_error["spend_baseline"]
npm_error["spend_diff_pp"] = npm_error["spend_diff"] / npm_error["spend_baseline"] * 100

npm_error["npm_diff"] = npm_error["mean_npm_kg_new"] - npm_error["mean_npm_kg_baseline"]
npm_error["npm_diff_r"] = npm_error["npm_new_r"] - npm_error["npm_base_r"]


npm_error["kcal_diff"].describe()

to_calc = npm_error["kcal_diff"].tolist()

st.t.interval(0.95, df=len(to_calc) - 1, loc=np.mean(to_calc), scale=st.sem(to_calc))

# Plotting the histogram
plt.hist(npm_error["kcal_diff"], bins=10, edgecolor="black")

# Adding labels and title
plt.xlabel("kcal_diff")
plt.ylabel("Frequency")
plt.title("Histogram of kcal_diff")

# Displaying the histogram
plt.show()


npm_error["spend_diff"].describe()

to_calc = npm_error["spend_diff"].tolist()

st.t.interval(0.95, df=len(to_calc) - 1, loc=np.mean(to_calc), scale=st.sem(to_calc))

# Plotting the histogram
plt.hist(npm_error["spend_diff"], bins=10, edgecolor="black")

# Adding labels and title
plt.xlabel("spend_diff")
plt.ylabel("Frequency")
plt.title("Histogram of spend_diff")

# Displaying the histogram
plt.show()


npm_error["npm_diff"].describe()

to_calc = npm_error["npm_diff"].tolist()

st.t.interval(0.95, df=len(to_calc) - 1, loc=np.mean(to_calc), scale=st.sem(to_calc))

# Plotting the histogram
plt.hist(npm_error["npm_diff"], bins=10, edgecolor="black")

# Adding labels and title
plt.xlabel("npm_diff")
plt.ylabel("Frequency")
plt.title("Histogram of npm_diff")

# Displaying the histogram
plt.show()


npm_error["npm_diff_r"].describe()

to_calc = npm_error["npm_diff_r"].tolist()

st.t.interval(0.95, df=len(to_calc) - 1, loc=np.mean(to_calc), scale=st.sem(to_calc))

# Plotting the histogram
plt.hist(npm_error["npm_diff_r"], bins=10, edgecolor="black")

# Adding labels and title
plt.xlabel("npm_diff_r")
plt.ylabel("Frequency")
plt.title("Histogram of npm_diff_r")

# Displaying the histogram
plt.show()
