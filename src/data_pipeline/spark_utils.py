import os
import sys
import logging
from pyspark.sql import SparkSession
from .config import SparkConfig

def get_spark_session(config: SparkConfig) -> SparkSession:
    """
    Initializes and returns a SparkSession with the given configuration.
    Handles session creation for both local and cluster modes.
    """
    logging.info(f"Initializing SparkSession with master: {config.master}")
    builder = SparkSession.builder.appName("ApertisDataPipeline").master(config.master)
    
    # Apply memory and core configurations
    if config.driver_memory:
        builder.config("spark.driver.memory", config.driver_memory)
    if config.executor_memory:
        builder.config("spark.executor.memory", config.executor_memory)
    if config.num_executors:
        builder.config("spark.executor.instances", str(config.num_executors))
    if config.executor_cores:
        builder.config("spark.executor.cores", str(config.executor_cores))

    # Apply any extra configurations from the config file
    for key, value in config.extra_configs.items():
        builder.config(key, value)
    
    # Ensure Arrow is optimized and that Spark uses the same Python environment
    # This is crucial for avoiding version mismatches in cluster environments
    # In virtual environments, os.sys.executable points to the venv's Python
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    try:
        spark = builder.getOrCreate()
        # Set a default log level to avoid overly verbose output
        log_level = os.environ.get("APERTIS_SPARK_LOG_LEVEL", "WARN")
        spark.sparkContext.setLogLevel(log_level)
        logging.info("SparkSession initialized successfully.")
        logging.info(f"Spark UI available at: {spark.sparkContext.uiWebUrl}")
        return spark
    except Exception as e:
        logging.error(f"Failed to initialize SparkSession: {e}", exc_info=True)
        raise

def teardown_spark_session(spark: SparkSession):
    """Stops the given SparkSession."""
    logging.info("Tearing down SparkSession.")
    try:
        spark.stop()
        logging.info("SparkSession stopped successfully.")
    except Exception as e:
        logging.error(f"Error stopping SparkSession: {e}", exc_info=True)