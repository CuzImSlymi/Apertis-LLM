import logging
from transformers import AutoTokenizer
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField
from .config import TokenizeConfig

def get_tokenizer(path: str):
    """Loads a Hugging Face tokenizer from a given path or name."""
    return AutoTokenizer.from_pretrained(path)

def tokenize_partition(iterator, tokenizer_path, max_seq_length):
    """
    Tokenizes a partition of text data. This function is executed on each Spark worker.
    """
    # Tokenizer is instantiated once per worker, which is efficient
    tokenizer = get_tokenizer(tokenizer_path)
    # Ensure a pad token exists for consistency, though padding is not used in encoding here
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    for row in iterator:
        text = row['value']
        # Encode text to token IDs, truncating to the max sequence length
        tokens = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_length,
            padding=False  # Padding is handled by training data loaders, not here
        )
        # Yield a dictionary matching the desired output schema
        yield {"input_ids": tokens}

def run_tokenize_stage(spark: SparkSession, config: TokenizeConfig):
    """Orchestrates the distributed tokenization of the text data."""
    logging.info("Starting tokenize stage...")
    
    # Read the deduplicated text data
    df = spark.read.text(config.input_dir).repartition(config.num_partitions)
    
    # Broadcast tokenizer path and max length to all workers
    tokenizer_path_bc = spark.sparkContext.broadcast(config.tokenizer_path)
    max_seq_length_bc = spark.sparkContext.broadcast(config.max_seq_length)

    # Apply the tokenization function to each partition of the data
    tokenized_rdd = df.rdd.mapPartitions(
        lambda iterator: tokenize_partition(
            iterator,
            tokenizer_path_bc.value,
            max_seq_length_bc.value
        )
    )
    
    # Define the schema for the output DataFrame
    schema = StructType([
        StructField("input_ids", ArrayType(IntegerType()), nullable=False)
    ])
    
    tokenized_df = spark.createDataFrame(tokenized_rdd, schema=schema)
    
    output_format = config.output_format.lower()
    # Handle the 'arrow' format case by defaulting to 'parquet', which is more standard for Spark
    if output_format == "arrow":
        logging.warning("Output format 'arrow' is not a native Spark save format. Defaulting to 'parquet'.")
        output_format = "parquet"
    
    logging.info(f"Writing tokenized data to {config.output_dir} in {output_format} format...")
    
    if output_format == "parquet":
        tokenized_df.write.mode("overwrite").format("parquet").save(config.output_dir)
    else:
        # Raise an error for any other unsupported formats
        raise ValueError(f"Unsupported output format: {output_format}. Please use 'parquet'.")
        
    logging.info("Tokenize stage completed successfully.")