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
