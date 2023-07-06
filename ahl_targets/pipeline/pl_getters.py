import polars as pl
import asyncio

async def val_fields() -> pl.DataFrame:
    return pl.read_csv(
        "s3://ahl-private-data/in_home/latest_data/validation_field.csv",
        encoding="ISO-8859-1",
    )

async def product_master() -> pl.DataFrame:
    return pl.read_csv(
        "s3://ahl-private-data/in_home/latest_data/product_master.csv",
        encoding="ISO-8859-1",
    )

async def uom() -> pl.DataFrame:
    return pl.read_csv(
        "s3://ahl-private-data/in_home/latest_data/uom.csv",
        encoding="ISO-8859-1",
    )

async def product_codes() -> pl.DataFrame:
    return pl.read_csv(
        "s3://ahl-private-data/in_home/latest_data/product_codes.csv",
        encoding="ISO-8859-1",
    )

async def product_values() -> pl.DataFrame:
    return pl.read_csv(
        "s3://ahl-private-data/in_home/latest_data/product_values.csv",
        encoding="ISO-8859-1",
    )

async def product_attribute() -> pl.DataFrame:
    return pl.read_csv(
        "s3://ahl-private-data/in_home/latest_data/product_attribute.csv",
        encoding="ISO-8859-1",
    )

async def product_metadata() -> pl.DataFrame:
    tasks = []
    (
        val_fields_df,
        product_master_df,
        uom_df,
        product_codes_df,
        product_values_df,
        product_attribute_df,
    ) = (
        asyncio.create_task(val_fields()),
        asyncio.create_task(product_master()),
        asyncio.create_task(uom()),
        asyncio.create_task(product_codes()),
        asyncio.create_task(product_values()),
        asyncio.create_task(product_attribute()),
    )
    return await asyncio.gather(*tasks)