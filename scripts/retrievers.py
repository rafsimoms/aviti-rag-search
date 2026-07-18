import pickle
import re
import pandas as pd
import pymorphy3
from rank_bm25 import BM25Okapi
import random
from aggregate import get_chunk_articles, aggregate_scores
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
from pathlib import Path

 
morph = pymorphy3.MorphAnalyzer()
lemma_cache = {}
token_re = re.compile(r"[а-яёa-z0-9]+")

def tokenize(text):
    tokens = token_re.findall(text.lower())
    lemmas = []
    for token in tokens:
        if token not in lemma_cache:
            lemma_cache[token] = morph.parse(token)[0].normal_form
        lemmas.append(lemma_cache[token])
    return lemmas

class RandomRetriever:
    def __init__(self, article_ids, top_k=10):
        self.article_ids = article_ids
        self.top_k = top_k
 
    def search(self, query: str) -> list[tuple[str, float]]:
        sample = random.sample(self.article_ids, min(self.top_k, len(self.article_ids)))
        return [(art_id, random.random()) for art_id in sample]


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.chunk_articles = get_chunk_articles(config['paths']['processed_articles'])
    
    def fit(self, df):
        raise NotImplementedError
 
    def _search_chunks(self, query):
        raise NotImplementedError
    
    def search(self, query):
        chunk_scores = self._search_chunks(query)
        aggregated = aggregate_scores(chunk_scores=chunk_scores, chunk_articles=self.chunk_articles, 
                                      mode=self.config['search']['aggregation'],
                                      top_k=self.config['search']['agg_top_k'])
        return aggregated[:self.config['search']['top_k']]


class BM25Retriever(BaseRetriever):
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'chunk_ids': self.chunk_ids}, f)
 
    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.bm25 = state['bm25']
        self.chunk_ids = state['chunk_ids']
    
    def fit(self, df):
        self.chunk_ids = df['chunk_id'].astype(str).tolist()
        corpus = [tokenize(text) for text in df['chunk']]
        self.bm25 = BM25Okapi(corpus)
    
    def _search_chunks(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        pairs = list(zip(self.chunk_ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:self.config['search']['candidate_chunks']]
    

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = config['models']['embedding']
        self.batch_size = config['models']['batch_size']
        self.candidate_chunks = config['search']['candidate_chunks']
        self.passage_prefix = 'passage: '
        self.query_prefix = 'query: '
        self._model = None
        self.index = None
        self.chunk_ids = None
        
    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
        
    def _encode(self, texts, prefix, show_progress=False):
        texts = [prefix + t for t in texts]
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)
        
    def fit(self, df):
        df = df.reset_index(drop=True)
        self.chunk_ids = df["chunk_id"].astype(str).tolist()
        embeddings = self._encode(df["chunk"].tolist(), self.passage_prefix, show_progress=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        return self
        
    def _search_chunks(self, query):
        query_emb = self._encode([query], self.query_prefix)
        top_n = min(self.candidate_chunks, self.index.ntotal)
        scores, positions = self.index.search(query_emb, top_n)
        return [
            (self.chunk_ids[pos], float(score))
            for pos, score in zip(positions[0], scores[0])
            if pos != -1
        ]
    
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "dense.faiss"))
        pd.DataFrame({"chunk_id": self.chunk_ids}).to_parquet(path / "dense_mapping.parquet")
 
    def load(self, path):
        path = Path(path)
        self.index = faiss.read_index(str(path / "dense.faiss"))
        self.chunk_ids = pd.read_parquet(path / "dense_mapping.parquet")["chunk_id"].tolist()
        return self
        

        
class HybridRetriever(BaseRetriever):

    def __init__(self, config, retrievers):
        super().__init__(config)
        self.retrievers = retrievers
        self.rrf_k = config["search"]["rrf_k"]
        self.candidate_chunks = config["search"]["candidate_chunks"]

    def fit(self, df):
        for retriever in self.retrievers:
            retriever.fit(df)
        return self
    
    def _search_chunks(self, query):
        fused = {}
        for retriever in self.retrievers:
            ranked = retriever._search_chunks(query)
            for rank, (chunk_id, _score) in enumerate(ranked, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        pairs = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return pairs[:self.candidate_chunks]

class RerankRetriever(BaseRetriever):
    def __init__(self, config, base_retriever):
        super().__init__(config)
        self.base_retirver = base_retriever
        self.batch_size = config['models']['batch_size']
        self.model_name = config['models']['reranker']
        self.rerank_candidates = config['search']['rerank_candidates']
        df = pd.read_parquet('../' + config['paths']['processed_articles'])
        self.chunk_texts = dict(zip(df['chunk_id'].astype(str), df['chunk']))
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model
    
    def _search_chunks(self, query):
        candidates = self.base_retirver._search_chunks(query)[:self.rerank_candidates]
        pairs = [(query, self.chunk_texts[chunk_id]) for chunk_id, score in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        reranked = [(chunk_id, float(score)) for (chunk_id, rernker_score), score in zip(candidates, scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked








    

