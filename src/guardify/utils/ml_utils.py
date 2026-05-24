class MlUtils:

    @staticmethod
    def get_vectorizer_config():
        from sklearn.feature_extraction.text import TfidfVectorizer

        return TfidfVectorizer(
            ngram_range=(1, 4),
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            analyzer="word",
            strip_accents="unicode",
            token_pattern=r"\b\w+\b",
        )
