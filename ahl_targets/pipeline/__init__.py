from ahl_targets import Path, PROJECT_DIR

# Makes necessary directories for outputs
output_subdirs = [
    "data",
    "figures",
    "figures/png",
]

for folder in output_subdirs:
    Path(PROJECT_DIR / "outputs" / f"{folder}").mkdir(
        parents=True,
        exist_ok=True,
    )

input_subdirs = [
    "processed",
    "raw",
]

for folder in input_subdirs:
    Path(PROJECT_DIR / "inputs" / f"{folder}").mkdir(
        parents=True,
        exist_ok=True,
    )


# model parameters
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
ed_high_sales_change_values = [12.5]  # percentage shift in sales
ed_low_sales_change_values = [
    0,
    2.5,
    5,
    7.5,
    10,
    12.5,
    15,
    17.5,
    20,
]  # percentage shift in sales
ed_cutoff = 400  # cut off for high energy density

npm_reduction_values = [2, 3, 4, 5]  # reduction in NPM
npm_high_sales_change_values = [12.5]  # percentage shift in sales
npm_low_sales_change_values = [
    2.5,
    5,
    7.5,
    10,
    12.5,
    15,
    17.5,
    20,
]  # percentage shift in sales
npm_cutoff = 4  # cut off for high NPM


product_share_reform_values_low = [
    1
]  # percentage of HFSS products that are reformulated to become non-HFSS
product_share_reform_values_medium = [
    0
]  # percentage of HFSS products that are reformulated to become non-HFSS
product_share_reform_values_high = [
    0
]  # percentage of HFSS products that are reformulated to become non-HFSS
hfss_high_sales_change_values = [10, 15]  # percentage shift in sales
hfss_low_sales_change_values = [2.5, 5]  # percentage shift in sales
hfss_cutoff = [10, 15, 20]
