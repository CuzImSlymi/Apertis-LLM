import yaml
from dataclasses import dataclass, field, is_dataclass
from typing import List, Dict, Optional, Any, Union

@dataclass
class SparkConfig:
    master: str = "local[*]"
    driver_memory: str = "16g"
    executor_memory: str = "8g"
    num_executors: Optional[int] = None
    executor_cores: int = 4
    extra_configs: Dict[str, Any] = field(default_factory=lambda: {
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max": "2047m",
        "spark.sql.shuffle.partitions": "200"
    })

@dataclass
class DownloadConfig:
    source: str = "common_crawl"
    warc_paths_url: Optional[str] = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-50/warc.paths.gz"
    num_warc_files: int = 1000
    output_dir: str = "data/pipeline/raw_warc"
    num_partitions: int = 200

@dataclass
class CleanConfig:
    input_dir: str = "data/pipeline/raw_warc"
    output_dir: str = "data/pipeline/cleaned_text"
    min_text_length: int = 256
    max_text_length: int = 100000
    fasttext_model_path: str = "models/lid.176.bin"
    language_whitelist: List[str] = field(default_factory=lambda: ["en"])
    num_partitions: int = 200

@dataclass
class DeduplicateConfig:
    input_dir: str = "data/pipeline/cleaned_text"
    output_dir: str = "data/pipeline/deduplicated_text"
    minhash_threshold: float = 0.8
    num_minhash_permutations: int = 128
    lsh_num_bands: int = 16
    num_partitions: int = 200
    connected_components_iterations: int = 10

@dataclass
class TokenizeConfig:
    input_dir: str = "data/pipeline/deduplicated_text"
    output_dir: str = "data/pipeline/tokenized"
    tokenizer_path: str = "gpt2"
    max_seq_length: int = 2048
    output_format: str = "parquet"
    num_partitions: int = 200

@dataclass
class DataPipelineConfig:
    spark: SparkConfig = field(default_factory=SparkConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    clean: CleanConfig = field(default_factory=CleanConfig)
    deduplicate: DeduplicateConfig = field(default_factory=DeduplicateConfig)
    tokenize: TokenizeConfig = field(default_factory=TokenizeConfig)
    stages: List[str] = field(default_factory=lambda: ["download", "clean", "deduplicate", "tokenize"])

    @classmethod
    def from_yaml(cls, path: str) -> 'DataPipelineConfig':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        def _from_dict(data_class, data):
            if not is_dataclass(data_class):
                return data

            init_kwargs = {}
            for f in data_class.__dataclass_fields__.values():
                if f.name in data:
                    field_type = f.type
                    field_value = data[f.name]
                    
                    origin = getattr(field_type, '__origin__', None)
                    if origin in (Union, Optional):
                        actual_type = field_type.__args__[0]
                        if is_dataclass(actual_type):
                            init_kwargs[f.name] = _from_dict(actual_type, field_value)
                        else:
                            init_kwargs[f.name] = field_value
                    elif is_dataclass(field_type):
                        init_kwargs[f.name] = _from_dict(field_type, field_value)
                    else:
                        init_kwargs[f.name] = field_value
            return data_class(**init_kwargs)

        return _from_dict(cls, config_dict)

def create_sample_pipeline_config(output_path: str):
    sample_config = {
        "spark": {
            "master": "local[*]",
            "driver_memory": "16g",
            "executor_memory": "8g",
            "num_executors": None,
            "executor_cores": 4,
            "extra_configs": {
                "spark.sql.execution.arrow.pyspark.enabled": "true",
                "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                "spark.kryoserializer.buffer.max": "2047m",
                "spark.sql.shuffle.partitions": "200"
            }
        },
        "download": {
            "source": "common_crawl",
            "warc_paths_url": "https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-50/warc.paths.gz",
            "num_warc_files": 1000,
            "output_dir": "data/pipeline/raw_warc",
            "num_partitions": 200
        },
        "clean": {
            "input_dir": "data/pipeline/raw_warc",
            "output_dir": "data/pipeline/cleaned_text",
            "min_text_length": 256,
            "max_text_length": 100000,
            "fasttext_model_path": "models/lid.176.bin",
            "language_whitelist": ["en"],
            "num_partitions": 200
        },
        "deduplicate": {
            "input_dir": "data/pipeline/cleaned_text",
            "output_dir": "data/pipeline/deduplicated_text",
            "minhash_threshold": 0.8,
            "num_minhash_permutations": 128,
            "lsh_num_bands": 16,
            "num_partitions": 200,
            "connected_components_iterations": 10
        },
        "tokenize": {
            "input_dir": "data/pipeline/deduplicated_text",
            "output_dir": "data/pipeline/tokenized",
            "tokenizer_path": "gpt2",
            "max_seq_length": 2048,
            "output_format": "parquet",
            "num_partitions": 200
        },
        "stages": ["download", "clean", "deduplicate", "tokenize"]
    }
    with open(output_path, 'w') as f:
        yaml.dump(sample_config, f, indent=2, sort_keys=False, default_flow_style=False)