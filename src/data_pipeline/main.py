import os
import sys
import argparse
import logging
from .config import DataPipelineConfig, create_sample_pipeline_config
from .spark_utils import get_spark_session, teardown_spark_session
from .download import run_download_stage
from .clean import run_clean_stage
from .deduplicate import run_deduplicate_stage
from .tokenize import run_tokenize_stage

def setup_logging():
    """Configures the root logger for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def run_pipeline(config: DataPipelineConfig):
    """
    Orchestrates the execution of the data pipeline stages based on the config.
    """
    setup_logging()
    
    spark = None
    try:
        spark = get_spark_session(config.spark)
        
        # Execute each stage if it's present in the config's 'stages' list
        if "download" in config.stages:
            run_download_stage(spark, config.download)
        
        if "clean" in config.stages:
            run_clean_stage(spark, config.clean)
        
        if "deduplicate" in config.stages:
            run_deduplicate_stage(spark, config.deduplicate)
        
        if "tokenize" in config.stages:
            run_tokenize_stage(spark, config.tokenize)
            
        logging.info("Data pipeline finished all stages successfully.")
        
    except Exception as e:
        # Catch any exception during the pipeline run for graceful shutdown
        logging.error(f"Data pipeline failed with an error: {e}", exc_info=True)
    finally:
        # Ensure the Spark session is always stopped, even if an error occurs
        if spark:
            teardown_spark_session(spark)

def main():
    """
    Main entry point for running the data pipeline from the command line.
    This function is not used by the Apertis CLI but allows standalone execution.
    """
    parser = argparse.ArgumentParser(description="Apertis Data Processing Pipeline (Standalone)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the data pipeline YAML configuration file."
    )
    parser.add_argument(
        "--create-config",
        type=str,
        metavar="PATH",
        help="Create a sample configuration file at the specified path and exit."
    )
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_pipeline_config(args.create_config)
        print(f"Sample configuration file created at {args.create_config}")
        sys.exit(0)
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
        
    config = DataPipelineConfig.from_yaml(args.config)
    run_pipeline(config)

if __name__ == "__main__":
    main()