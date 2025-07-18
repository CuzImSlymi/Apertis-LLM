import os
import gzip
import requests
import logging
from pyspark.sql import SparkSession
from .config import DownloadConfig

def download_warc_paths(url: str) -> list[str]:
    """Downloads and decompresses the list of WARC file paths from Common Crawl."""
    logging.info(f"Downloading WARC paths from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with gzip.open(response.raw, 'rt') as f:
            paths = [line.strip() for line in f]
        
        logging.info(f"Successfully downloaded {len(paths)} WARC paths.")
        return paths
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download WARC paths from {url}: {e}")
        raise

def download_and_save_warc(path: str, output_dir: str):
    """Downloads a single WARC file and saves it to the specified output directory."""
    base_url = "https://data.commoncrawl.org/"
    full_url = f"{base_url}{path}"
    # Sanitize the filename to be safe for local filesystems
    file_name = path.replace('/', '_')
    output_path = os.path.join(output_dir, file_name)
    
    try:
        with requests.get(full_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                # Download in chunks to handle large files efficiently
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        return f"SUCCESS: {path}"
    except requests.exceptions.RequestException as e:
        logging.warning(f"Failed to download {full_url}: {e}. Skipping.")
        return f"FAILURE: {path}"

def run_download_stage(spark: SparkSession, config: DownloadConfig):
    """Orchestrates the distributed download of WARC files."""
    logging.info("Starting download stage...")
    
    if config.source.lower() != "common_crawl":
        raise NotImplementedError(f"Download source '{config.source}' is not supported.")

    # The output directory for this stage is where files will be saved directly.
    # The next stage will read from this directory of raw files.
    os.makedirs(config.output_dir, exist_ok=True)
    
    warc_paths = download_warc_paths(config.warc_paths_url)
    if not warc_paths:
        logging.error("No WARC paths found. Aborting download stage.")
        return

    if config.num_warc_files > 0:
        warc_paths = warc_paths[:config.num_warc_files]
        logging.info(f"Processing the first {config.num_warc_files} WARC files.")
    
    # Distribute the list of paths to the Spark workers
    paths_rdd = spark.sparkContext.parallelize(warc_paths, config.num_partitions)
    
    # Each worker now downloads and saves its assigned file directly to the output directory.
    # This is a highly scalable pattern that avoids collecting large amounts of data on the driver.
    # Note: This assumes the output_dir is accessible by all workers. In local mode, this is true.
    # In a cluster, this would need to be a shared filesystem like HDFS, S3, etc :pray:
    results_rdd = paths_rdd.map(
        lambda path: download_and_save_warc(path, config.output_dir)
    )

    # Collect only the small status strings (SUCCESS/FAILURE) to the driver for logging.
    results = results_rdd.collect()
    
    success_count = sum(1 for r in results if r.startswith("SUCCESS"))
    failure_count = len(results) - success_count
    
    logging.info(f"Download stage complete. Successfully downloaded {success_count} files. Failed to download {failure_count} files.")
    if failure_count > 0:
        logging.warning("Some files failed to download. Check logs for details.")
    
    logging.info(f"Downloaded files are located in: {config.output_dir}")