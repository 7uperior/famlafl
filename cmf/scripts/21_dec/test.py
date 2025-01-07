    # Step 1: Define LazyFrames
    agg_trades_lf = (
        pl.scan_parquet("./cmf/data/train/agg_trades.parquet")
        .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
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
            pl.col("ask_amount").first().alias("ask_amount")
        ])
    )

    orderbook_embedding_lf = (
        pl.scan_parquet("./cmf/data/train/orderbook_embedding.parquet")
        .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
        .group_by(["timestamp_ns", "instrument"])
        .agg([
            pl.col("mid_price").first().alias("mid_price"),
            pl.col("vwap_1_lvl").first().alias("vwap_1_lvl"),
            pl.col("vwap_3_lvl").first().alias("vwap_3_lvl"),
            pl.col("vwap_5_lvl").first().alias("vwap_5_lvl"),
            pl.col("vwap_10_lvl").first().alias("vwap_10_lvl")
        ])
    )

    orderbook_solusdt_lf = (
        pl.scan_parquet("./cmf/data/train/orderbook_solusdt.parquet")
        .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
    )

    target_data_solusdt_lf = (
        pl.scan_parquet("./cmf/data/train/target_data_solusdt.parquet")
        .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
    )

    target_solusdt_lf = (
        pl.scan_parquet("./cmf/data/train/target_solusdt.parquet")
        .with_columns(pl.col("timestamp_ns").cast(pl.Datetime("ns")))
    )

    # Step 2: Combine LazyFrames with Suffixes
    combined_lf = (
        agg_trades_lf
        .join(orderbook_embedding_lf, on=["timestamp_ns", "instrument"], how="inner", suffix="_orderbook_embedding")
        .join(orderbook_solusdt_lf, on="timestamp_ns", how="inner", suffix="_orderbook_solusdt")
        .join(target_data_solusdt_lf, on="timestamp_ns", how="inner", suffix="_target_data")
        .join(target_solusdt_lf, on="timestamp_ns", how="inner", suffix="_target")
    )

    # Step 3: Process data in chunks using .slice() and .collect()
    chunk_size = 100_000  # Adjust as needed
    offset = 0
    df_chunks = []

    while True:
        # Slice the LazyFrame for a specific chunk
        chunk_lf = combined_lf.slice(offset, chunk_size)
        # Collect the chunk into a DataFrame
        chunk_df = chunk_lf.collect()

        # If the chunk is empty, break the loop
        if chunk_df.is_empty():
            break

        print(chunk_df.head(3))  # Example: Display the first few rows of the chunk
        df_chunks.append(chunk_df)
        offset += chunk_size

    # Concatenate chunks if any data was loaded
    if df_chunks:
        combined_df = pl.concat(df_chunks, how="vertical")
        print("Combined DataFrame:")
        print(combined_df.head(5))
    else:
        combined_df = None
        print("No data was loaded.")