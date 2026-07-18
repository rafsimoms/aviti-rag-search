import pandas as pd
import yaml
from pathlib import Path
import ranx
import time
from ranx import Qrels, Run, evaluate as ranx_evaluate
import random
import json
from retrievers import BM25Retriever, DenseRetriever

def load_config(path):
    return yaml.safe_load(Path(path).read_text())

def get_data(path):
    data = []
    with open(path, 'r', encoding="UTF-8") as f:
        for line in f:
            line = line.rstrip()
            data.append(json.loads(line))
    return data

def evaluate_retriever(retriever, path):
    data = get_data(path)
    qrels = {}
    run = {}
    latencies = []
    for line in data:
        qrels[line['query_id']] = {str(art_id): 1 for art_id in line['ground_truth']}
        t0 = time.perf_counter()
        results = retriever.search(line["query_text"])
        latencies.append(time.perf_counter() - t0)
        run[line['query_id']] = {str(art_id): float(score) for art_id, score in results}
    qrels_ranx = Qrels(qrels)
    run_ranx = Run(run)
    map_at_10 = ranx_evaluate(qrels_ranx, run_ranx, "map@10")
    avg_latency_ms = 1000 * sum(latencies) / len(latencies)
    print(f"map10: {map_at_10}, latency: {avg_latency_ms}")
    return map_at_10
    

if __name__ == "__main__":
    config = load_config("../config.yaml")
    index_path = Path("../" + config["paths"]["dense_index"])

    retriever = DenseRetriever(config)
    if (index_path / "dense.faiss").exists():
        retriever.load(index_path)
    else:
        df = pd.read_parquet("../" + config["paths"]["processed_articles"])
        retriever.fit(df)
        retriever.save(index_path)

    evaluate_retriever(retriever, "../" + config["paths"]["calibration"])

