
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import pickle
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArticleEncoder:

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        embedding_dim: int = 384,
        max_length: int = 128,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Cache for embeddings: news_id -> embedding
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Load cache if exists
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
        
        logger.info(f"ArticleEncoder initialized on {device}")
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def encode_text(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:

        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Get embeddings
                outputs = self.model(**encoded)
                embeddings = self._mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_article(
        self,
        news_id: str,
        title: str,
        abstract: Optional[str] = None,
        category: Optional[str] = None,
        use_cache: bool = True
    ) -> np.ndarray:

        if use_cache and news_id in self.embedding_cache:
            return self.embedding_cache[news_id]
        
        # Combine title and abstract
        if abstract and isinstance(abstract, str) and len(abstract.strip()) > 0:
            text = f"{title} [SEP] {abstract}"
        else:
            text = title
        
        # Encode
        embedding = self.encode_text(text)[0]
        
        # Cache
        self.embedding_cache[news_id] = embedding
        
        return embedding
    
    def encode_articles_batch(
        self,
        articles: List[Dict],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:

        embeddings = {}
        texts_to_encode = []
        news_ids = []
        
        # Collect texts that need encoding
        for article in articles:
            news_id = article['news_id']
            
            # Skip if cached
            if news_id in self.embedding_cache:
                embeddings[news_id] = self.embedding_cache[news_id]
                continue
            
            # Prepare text
            title = article['title']
            abstract = article.get('abstract', '')
            
            # Handle NaN/None abstracts
            if abstract and isinstance(abstract, str) and len(abstract.strip()) > 0:
                text = f"{title} [SEP] {abstract}"
            else:
                text = title
            
            texts_to_encode.append(text)
            news_ids.append(news_id)
        
        if len(texts_to_encode) == 0:
            logger.info("All articles already cached")
            return embeddings
        
        logger.info(f"Encoding {len(texts_to_encode)} new articles...")
        
        # Encode in batches
        iterator = range(0, len(texts_to_encode), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding articles")
        
        for i in iterator:
            batch_texts = texts_to_encode[i:i+batch_size]
            batch_ids = news_ids[i:i+batch_size]
            
            batch_embeddings = self.encode_text(batch_texts, batch_size=len(batch_texts))
            
            for news_id, embedding in zip(batch_ids, batch_embeddings):
                embeddings[news_id] = embedding
                self.embedding_cache[news_id] = embedding
        
        logger.info(f"Encoded {len(texts_to_encode)} articles. Cache size: {len(self.embedding_cache)}")
        
        return embeddings
    
    def encode_from_dataframe(self, df, batch_size: int = 32) -> Dict[str, np.ndarray]:
        articles = df.to_dict('records')
        return self.encode_articles_batch(articles, batch_size=batch_size)
    
    def get_embedding(self, news_id: str) -> Optional[np.ndarray]:

        return self.embedding_cache.get(news_id)
    
    def get_embeddings(self, news_ids: List[str]) -> np.ndarray:

        embeddings = []
        for news_id in news_ids:
            emb = self.get_embedding(news_id)
            if emb is not None:
                embeddings.append(emb)
            else:
                # Return zero vector if not found
                logger.warning(f"Article {news_id} not in cache, using zero vector")
                embeddings.append(np.zeros(self.embedding_dim))
        
        return np.array(embeddings)
    
    def save_cache(self, path: Optional[str] = None):

        if path is None:
            if self.cache_dir is None:
                raise ValueError("No cache directory specified")
            path = self.cache_dir / 'article_embeddings.pkl'
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.embedding_cache, f)
        
        logger.info(f"Saved {len(self.embedding_cache)} embeddings to {path}")
    
    def _load_cache(self):
        cache_path = self.cache_dir / 'article_embeddings.pkl'
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        else:
            logger.info("No cache file found, starting fresh")
    
    def clear_cache(self):
        self.embedding_cache.clear()
        logger.info("Cache cleared")


class CategoryEncoder:

    def __init__(self, categories: List[str], embedding_dim: int = 16):

        self.categories = sorted(categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.embedding_dim = embedding_dim
        
        # Create embedding layer
        self.embedding = nn.Embedding(len(self.categories), embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        logger.info(f"CategoryEncoder initialized with {len(self.categories)} categories")
    
    def encode(self, category: str) -> np.ndarray:

        idx = self.category_to_idx.get(category, 0)  # Use 0 as default for unknown
        with torch.no_grad():
            embedding = self.embedding(torch.tensor([idx]))
        return embedding.numpy()[0]
    
    def encode_batch(self, categories: List[str]) -> np.ndarray:

        indices = [self.category_to_idx.get(cat, 0) for cat in categories]
        with torch.no_grad():
            embeddings = self.embedding(torch.tensor(indices))
        return embeddings.numpy()


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from src.data.mind_loader import MINDDataLoader
    
    # Load data
    print("Loading MIND data...")
    loader = MINDDataLoader(data_dir='data/raw/train', dataset_type='train')
    loader.load_news()
    
    # Initialize encoder
    print("\nInitializing article encoder...")
    encoder = ArticleEncoder(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        embedding_dim=384,
        cache_dir='data/processed/embeddings'
    )
    
    # Encode all articles
    print("\nEncoding articles...")
    embeddings = encoder.encode_from_dataframe(loader.news_df.head(100), batch_size=32)
    
    print(f"\nEncoded {len(embeddings)} articles")
    print(f"Embedding shape: {list(embeddings.values())[0].shape}")
    
    # Save cache
    encoder.save_cache()
    print("\nCache saved successfully!")