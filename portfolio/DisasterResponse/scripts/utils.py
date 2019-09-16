import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def tokenize(text, remove_stopwords=True):
    """
    Tokenizes a given text.
    Args:
        text: text string
        remove_stopwords: option to remove stopwords or not
    Returns:
        (array) clean_tokens: array of clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stopw = set(stopwords.words('english')) if remove_stopwords else []

    cleaned_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens if token not in stopw]
    return cleaned_tokens
