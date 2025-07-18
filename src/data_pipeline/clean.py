import os
import io
import logging
import fasttext
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
from warcio.archiveiterator import ArchiveIterator
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from .config import CleanConfig

# Suppress the known, harmless warning from BeautifulSoup for messy web data
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

class TextCleaner:
    def __init__(self, fasttext_model_path: str, language_whitelist: list[str]):
        self.model = fasttext.load_model(fasttext_model_path)
        self.language_whitelist = set(f"__label__{lang}" for lang in language_whitelist)

    def is_valid_language(self, text: str) -> bool:
        if not self.language_whitelist:
            return True
        # FastText expects a single line of text for prediction
        predictions = self.model.predict(text.replace("\n", " "), k=1)
        return predictions[0][0] in self.language_whitelist

    def clean_html(self, html_content: bytes) -> str:
        try:
            # Specify encoding to handle potential errors gracefully
            soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
            # Remove common non-content tags
            for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside"]):
                script_or_style.decompose()
            text = soup.get_text(separator='\n', strip=True)
            return text
        except Exception:
            # Return empty string if parsing fails for any reason
            return ""

    def process_warc_record(self, record) -> list[str]:
        if record.rec_type != 'response' or not record.http_headers or record.http_headers.get_statuscode() != '200':
            return []
        
        content = record.content_stream().read()
        if not content:
            return []
        
        cleaned_text = self.clean_html(content)
        if self.is_valid_language(cleaned_text):
            return [cleaned_text]
        return []

def process_warc_stream(stream_content: bytes, fasttext_model_path: str, language_whitelist: list[str]) -> list[str]:
    """
    Processes a WARC file stream content, yielding cleaned text records.
    This is designed to be used with Spark's binaryFiles RDD for scalability.
    """
    cleaner = TextCleaner(fasttext_model_path, language_whitelist)
    results = []
    with io.BytesIO(stream_content) as stream:
        for record in ArchiveIterator(stream):
            try:
                results.extend(cleaner.process_warc_record(record))
            except Exception as e:
                # Log error for a specific bad record but continue processing the rest of the file
                logging.warning(f"Error processing a record in WARC stream: {e}", exc_info=False)
    return results

def run_clean_stage(spark: SparkSession, config: CleanConfig):
    logging.info("Starting clean stage...")
    
    if not os.path.exists(config.fasttext_model_path):
        logging.error(f"FastText model not found at {config.fasttext_model_path}. Please download it.")
        raise FileNotFoundError(f"FastText model not found: {config.fasttext_model_path}")

    # Read entire directory of WARC files as a stream of (path, content) pairs
    binary_rdd = spark.sparkContext.binaryFiles(config.input_dir, config.num_partitions)

    # Broadcast necessary objects to all worker nodes
    fasttext_model_path_bc = spark.sparkContext.broadcast(config.fasttext_model_path)
    language_whitelist_bc = spark.sparkContext.broadcast(config.language_whitelist)

    # Use flatMap to process each WARC file stream and yield multiple text records.
    # This is highly scalable and avoids memory issues with large files.
    text_rdd = binary_rdd.flatMap(
        lambda x: process_warc_stream(x[1], fasttext_model_path_bc.value, language_whitelist_bc.value)
    )

    # Convert the RDD of strings into a DataFrame for easier filtering
    text_df = text_rdd.toDF(schema=StringType())

    # Filter based on text length using Spark's native SQL functions for performance
    filter_condition = f"length(value) >= {config.min_text_length} AND length(value) <= {config.max_text_length}"
    filtered_df = text_df.filter(filter_condition)

    logging.info(f"Writing cleaned and filtered text to {config.output_dir}...")
    
    filtered_df.write.mode("overwrite").format("text").save(config.output_dir)
    
    logging.info("Clean stage completed successfully.")