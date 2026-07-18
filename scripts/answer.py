import pandas as pd
import argparse
from evaluate import build_retriver, load_config
from tqdm.auto import tqdm


def main():
    config = load_config('../config.yaml')
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever', required=True, choices=['random', 'bm25', 'dense', 'hybrid', 'reranker'])
    args = parser.parse_args()
    retriever = build_retriver(args.retriever, config)
    df = pd.read_feather('../' + config['paths']['test'])
    rows = []
    for idx, line in tqdm(df.iterrows(), desc='predict'):
        results = retriever.search(line["query_text"])
        results = " ".join([str(article_id) for article_id, score in results])
        rows.append({'query_id': line['query_id'], 'answer': results})
    pd.DataFrame(rows).to_csv('../' + config['paths']['answer'], index=False)
    print("done")

if __name__ == '__main__':
    main()




