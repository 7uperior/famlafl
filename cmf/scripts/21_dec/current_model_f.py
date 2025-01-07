import polars as pl

# 1) agg_trades (BTC/ETH + maybe SOL), group by instrument
#    Because `agg_trades` has an "instrument" column, we do group_by(["timestamp_ns","instrument"]).
agg_trades_lf = (
    pl.scan_parquet("./cmf/data/train/agg_trades.parquet")
    .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
    .group_by(["timestamp_ns", "instrument"])  # has instrument
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

# 2) orderbook_embedding (BTC/ETH only?), group by instrument as well
orderbook_embedding_lf = (
    pl.scan_parquet("./cmf/data/train/orderbook_embedding.parquet")
    .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
    .group_by(["timestamp_ns", "instrument"])  # has instrument
    .agg([
        pl.col("mid_price").first().alias("mid_price"),
        pl.col("vwap_1_lvl").first().alias("vwap_1_lvl"),
        pl.col("vwap_3_lvl").first().alias("vwap_3_lvl"),
        pl.col("vwap_5_lvl").first().alias("vwap_5_lvl"),
        pl.col("vwap_10_lvl").first().alias("vwap_10_lvl"),
    ])
)

# 3) orderbook_solusdt (NO instrument column, or ignoring it)
#    So we do NOT group by instrument. We'll only keep timestamp_ns (and all the asks/bids)
orderbook_solusdt_lf = (
    pl.scan_parquet("./cmf/data/train/orderbook_solusdt.parquet")
    .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
    # no group_by(["timestamp_ns","instrument"]), because there's no "instrument" column
)

# 4) target_data_solusdt (NO instrument column, or ignoring it)
target_data_solusdt_lf = (
    pl.scan_parquet("./cmf/data/train/target_data_solusdt.parquet")
    .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
    # no group_by instrument
)

# 5) target_solusdt (NO instrument column, or ignoring it)
target_solusdt_lf = (
    pl.scan_parquet("./cmf/data/train/target_solusdt.parquet")
    .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
)

# ============== FULL OUTER JOINS ==============
# We'll do it step by step, or all at once. Step by step is easier to debug.

# First, combine the data that has instrument: agg_trades + orderbook_embedding
# Outer join to keep all rows from both frames even if instrument or timestamp mismatch
trades_plus_embed_lf = (
    agg_trades_lf.join(
        orderbook_embedding_lf,
        on=["timestamp_ns", "instrument"],
        how="outer",  # full outer
        suffix="_embedding"
    )
)

# Next, outer join with orderbook_solusdt_lf
# NOTE: orderbook_solusdt has no instrument col,
# so we can only join on "timestamp_ns".
with_sol_lf = (
    trades_plus_embed_lf.join(
        orderbook_solusdt_lf,
        on="timestamp_ns",  # only timestamp
        how="outer",
        suffix="_solusdt"
    )
)

# Then, outer join with target_data_solusdt
with_target_data_lf = (
    with_sol_lf.join(
        target_data_solusdt_lf,
        on="timestamp_ns",  # only timestamp
        how="outer",
        suffix="_target_data"
    )
)

# Finally, outer join with target_solusdt
combined_lf = (
    with_target_data_lf.join(
        target_solusdt_lf,
        on="timestamp_ns",  # only timestamp
        how="outer",
        suffix="_target"
    )
)

# ============== Collect & Save ==============
combined_df = combined_lf.collect()
print("Final shape:", combined_df.shape)
print("Head:\n", combined_df.head(10))

# Write to parquet
combined_df.write_parquet("combined_full_outer_no_instrument_sol.parquet")
print("Wrote combined_full_outer_no_instrument_sol.parquet")
