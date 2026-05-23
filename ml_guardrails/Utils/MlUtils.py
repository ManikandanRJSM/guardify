class MlUtils:

    @staticmethod
    def get_vectorizer_config():
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 4),    # single words + 2/3 word phrases
            max_features=50000,    # top 10000 important terms
            sublinear_tf=True,     # reduces impact of very frequent words
            min_df=2,
            max_df=0.95,           # ignore terms appearing less than 2 times
            analyzer="word",
            strip_accents="unicode",
            token_pattern=r"\b\w+\b"
        )
        return vectorizer