import polars as pl

# Path to your files (adjust as needed)
PATH_AGG_TRADES = "./cmf/data/train/agg_trades.parquet"
PATH_ORDERBOOK_SOL = "./cmf/data/train/orderbook_solusdt.parquet"
PATH_TARGET_DATA_SOL = "./cmf/data/train/target_data_solusdt.parquet"
PATH_TARGET_SOL = "./cmf/data/train/target_solusdt.parquet"

print("=== Building SOLUSDT pipeline ===")

# 1) agg_trades (filtered to SOLUSDT)
agg_trades_sol_lf = (
    pl.scan_parquet(PATH_AGG_TRADES)
    .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
    .filter(pl.col("instrument") == "SOLUSDT")  # keep only SOL
    .group_by(["timestamp_ns", "instrument"])
    .agg([
        pl.col("bid_max_price").first().alias("bid_max_price"),
        pl.col("ask_max_price").first().alias("ask_max_price"),
        pl.col("bid_min_price").first().alias("bid_min_price"),
        pl.col("ask_min_price").first().alias("ask_min_price"),
        pl.col("bid_mean_price").first().alias("bid_mean_price"),
        pl.col("ask_mean_price").first().alias("ask_mean_price"),
        pl.col("bid_count").first().alias("bid_count"),
        pl.col("ask_count").first().alias("ask_count"),
        pl.col("bid_amount").first().alias("bid_amount"),
        pl.col("ask_amount").first().alias("ask_amount"),
    ])
)

# 2) orderbook_solusdt (ensure it has instrument="SOLUSDT")
orderbook_solusdt_lf = (
    pl.scan_parquet(PATH_ORDERBOOK_SOL)
    .with_columns([
        pl.col("timestamp_ns").cast(pl.Datetime("ns")),
        pl.lit("SOLUSDT").alias("instrument")  # if not already present
    ])
)

# 3) target_data_solusdt
target_data_solusdt_lf = (
    pl.scan_parquet(PATH_TARGET_DATA_SOL)
    .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
)

# 4) target_solusdt
target_solusdt_lf = (
    pl.scan_parquet(PATH_TARGET_SOL)
    .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
)

# Combine them (inner join on timestamp_ns & instrument where needed)
combined_sol_lf = (
    agg_trades_sol_lf
    .join(orderbook_solusdt_lf, on=["timestamp_ns","instrument"], how="inner")
    .join(target_data_solusdt_lf, on=["timestamp_ns"], how="inner")
    .join(target_solusdt_lf, on=["timestamp_ns"], how="inner")
)

# Collect and save to file
combined_sol_df = combined_sol_lf.collect()
print("combined_sol_df shape:", combined_sol_df.shape)
combined_sol_df.write_parquet("combined_SOLUSDT.parquet")
print("Wrote combined_SOLUSDT.parquet\n")



print("=== Building BTC/ETH pipeline ===")

# 1) agg_trades (filtered to BTCUSDT or ETHUSDT)
agg_trades_btceth_lf = (
    pl.scan_parquet(PATH_AGG_TRADES)
    .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
    .filter(pl.col("instrument").is_in(["BTCUSDT", "ETHUSDT"]))
    .group_by(["timestamp_ns", "instrument"])
    .agg([
        pl.col("bid_max_price").first().alias("bid_max_price"),
        pl.col("ask_max_price").first().alias("ask_max_price"),
        pl.col("bid_min_price").first().alias("bid_min_price"),
        pl.col("ask_min_price").first().alias("ask_min_price"),
        pl.col("bid_mean_price").first().alias("bid_mean_price"),
        pl.col("ask_mean_price").first().alias("ask_mean_price"),
        pl.col("bid_count").first().alias("bid_count"),
        pl.col("ask_count").first().alias("ask_count"),
        pl.col("bid_amount").first().alias("bid_amount"),
        pl.col("ask_amount").first().alias("ask_amount"),
    ])
)

# 2) orderbook_embedding (already only has BTC/ETH)
orderbook_embedding_lf = (
    pl.scan_parquet("./cmf/data/train/orderbook_embedding.parquet")
    .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
    .group_by(["timestamp_ns", "instrument"])
    .agg([
        pl.col("mid_price").first().alias("mid_price"),
        pl.col("vwap_1_lvl").first().alias("vwap_1_lvl"),
        pl.col("vwap_3_lvl").first().alias("vwap_3_lvl"),
        pl.col("vwap_5_lvl").first().alias("vwap_5_lvl"),
        pl.col("vwap_10_lvl").first().alias("vwap_10_lvl"),
    ])
)

# If you have any BTC/ETH target data, define it similarly:
# target_data_btceth_lf = ...
# etc.

combined_btceth_lf = (
    agg_trades_btceth_lf
    .join(orderbook_embedding_lf, on=["timestamp_ns","instrument"], how="inner")
    # .join(target_data_btceth_lf, on=["timestamp_ns"], how="inner")   # If you have it
    # .join(target_btceth_lf, on=["timestamp_ns"], how="inner")       # If you have it
)

# Collect and save
combined_btceth_df = combined_btceth_lf.collect()
print("combined_btceth_df shape:", combined_btceth_df.shape)
combined_btceth_df.write_parquet("combined_BTCETH.parquet")
print("Wrote combined_BTCETH.parquet\n")
