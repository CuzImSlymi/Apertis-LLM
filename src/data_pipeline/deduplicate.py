import os
import re
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id, expr, size, lower, regexp_replace, split, array_distinct
from pyspark.sql.types import ArrayType, StringType, Row as SparkRow
from datasketch import MinHash
from .config import DeduplicateConfig

def create_minhash_signature(shingles: list[str], num_perm: int):
    """Creates a MinHash signature from a list of shingles."""
    if not shingles:
        return None
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode('utf8'))
    return m.hashvalues.tolist()

def find_clusters_in_partition(rows, threshold, num_perm):
    """
    Finds duplicate clusters within a single partition of data using MinHashLSH.
    This function is designed to be used with mapPartitions.
    """
    from datasketch import MinHashLSH
    
    # mapPartitions provides an iterator, convert to list to iterate multiple times
    row_list = list(rows)
    if not row_list:
        return
        
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    # first pass: Insert all documents from the partition into the LSH index
    for row in row_list:
        if row['minhash']:
            m = MinHash(num_perm=num_perm, hashvalues=row['minhash'])
            lsh.insert(row['id'], m)

    processed_ids = set()
    # Second pass;: Query for each document to find its cluster
    for row in row_list:
        doc_id = row['id']
        if doc_id not in processed_ids and row['minhash']:
            m = MinHash(num_perm=num_perm, hashvalues=row['minhash'])
            cluster = lsh.query(m)
            if len(cluster) > 1:
                # Yield a tuple of sorted IDs to represent a unique cluster
                yield tuple(sorted(list(cluster)))
            # Mark all members of the found cluster as processed to avoid redundant work
            for cid in cluster:
                processed_ids.add(cid)

def find_connected_components(edges):
    """
    A simple, non-distributed connected components algorithm to merge overlapping clusters.
    Runs on the driver with a relatively small amount of edge data.
    """
    adj = {}
    for edge_list in edges:
        for i in range(len(edge_list)):
            for j in range(i + 1, len(edge_list)):
                u, v = edge_list[i], edge_list[j]
                if u not in adj: adj[u] = set()
                if v not in adj: adj[v] = set()
                adj[u].add(v)
                adj[v].add(u)

    visited = set()
    components = []
    for node in adj:
        if node not in visited:
            component = []
            stack = [node]
            visited.add(node)
            while stack:
                curr = stack.pop()
                component.append(curr)
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            components.append(component)
    return components

def run_deduplicate_stage(spark: SparkSession, config: DeduplicateConfig):
    logging.info("Starting deduplicate stage (Dependency-Free Fallback)...")

    df = spark.read.text(config.input_dir).repartition(config.num_partitions)
    df_with_id = df.withColumn("id", monotonically_increasing_id())

    # Preprocess text and generate shingles using native Spark functions fort performance
    df_with_shingles = df_with_id \
        .withColumn("cleaned_text", lower(col("value"))) \
        .withColumn("cleaned_text", regexp_replace(col("cleaned_text"), r'[^\w\s]', '')) \
        .withColumn("words", split(col("cleaned_text"), r'\s+')) \
        .withColumn("shingles", array_distinct(col("words")))

    # Filter out documents that became empty after preprocessing
    filtered_shingles_df = df_with_shingles.filter(size(col("shingles")) > 0)
    
    # Generate MinHash signatures using a UDF
    minhash_udf = udf(lambda s: create_minhash_signature(s, config.num_minhash_permutations), ArrayType(StringType()))
    minhashed_df = filtered_shingles_df.withColumn("minhash", minhash_udf(col("shingles")))
    
    minhashed_df.cache()
    doc_count = minhashed_df.count()
    logging.info(f"Generated MinHashes for {doc_count} non-empty documents.")

    # Distribute the LSH clustering process using mapPartitions
    cluster_edges_rdd = minhashed_df.select("id", "minhash").rdd.mapPartitions(
        lambda rows: find_clusters_in_partition(rows, config.minhash_threshold, config.num_minhash_permutations)
    ).distinct()

    # Collect the cluster information. This is now a much smaller dataset of cluster tuples.
    collected_edges = cluster_edges_rdd.collect()

    if not collected_edges:
        logging.info("No duplicate clusters found. Writing all data.")
        df_with_id.select("value").write.mode("overwrite").format("text").save(config.output_dir)
        minhashed_df.unpersist()
        return

    logging.info(f"Found {len(collected_edges)} raw duplicate clusters. Merging and finding representatives...")
    # Merge overlapping clusters on the driver
    components = find_connected_components(collected_edges)

    # From each final component, choose one representative (the one with the minimum ID)
    representatives = [min(component) for component in components]
    
    # Get a set of all document IDs that are part of any duplicate cluster
    all_duplicate_ids = set()
    for component in components:
        all_duplicate_ids.update(component)

    # Create a DataFrame of the representative IDs to keep
    reps_rows = [SparkRow(id=int(i)) for i in representatives]
    reps_df = spark.createDataFrame(reps_rows)

    # Get the original documents that were NOT in any duplicate cluster
    non_duplicates_df = df_with_id.filter(~col("id").isin(all_duplicate_ids))

    # Get the text for the representative documents from the duplicate clusters
    duplicate_representatives_df = df_with_id.join(reps_df, "id", "inner")

    # Combine the two sets of documents (unique originals + one from each duplicate cluster)
    final_df = non_duplicates_df.select("value").union(duplicate_representatives_df.select("value"))

    deduplicated_count = final_df.count()
    logging.info(f"Writing {deduplicated_count} deduplicated documents to {config.output_dir}...")

    final_df.write.mode("overwrite").format("text").save(config.output_dir)

    minhashed_df.unpersist()
    logging.info("Deduplicate stage completed successfully.")