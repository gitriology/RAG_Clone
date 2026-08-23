from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy


class ModelRegistry:

    _embedding_model = None
    _validation_model = None
    _reranker_model = None
    _nlp = None

    @classmethod
    def get_embedding_model(cls):
        if cls._embedding_model is None:
            cls._embedding_model = SentenceTransformer(
                "BAAI/bge-base-en-v1.5"
            )

        return cls._embedding_model

    @classmethod
    def get_validation_model(cls):
        if cls._validation_model is None:
            cls._validation_model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

        return cls._validation_model

    @classmethod
    def get_reranker_model(cls):
        if cls._reranker_model is None:
            cls._reranker_model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

        return cls._reranker_model

    @classmethod
    def get_nlp(cls):
        if cls._nlp is None:
            cls._nlp = spacy.load(
                "en_core_web_sm"
            )

        return cls._nlp