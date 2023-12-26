import re
import nltk.corpus
nltk.download('stopwords')
import numpy as np
import pandas as pd
import statistics as st
from typing import Union, Text
from collections.abc import Iterable
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfRater:
    """
    A class for evaluating text using TF-IDF (Term Frequency-Inverse Document Frequency) analysis.

    This class provides methods for cleaning and preprocessing text data, as well as computing TF-IDF ratings.

    Attributes:
    - vectorizer (TfidfVectorizer): A TfidfVectorizer instance for determining TF-IDF representations of text data.
    - stemmer (PorterStemmer): An instance for stemming words to their root form.
    - stop (list): A collection of English language stopwords to be excluded from analysis.
    - stemmed_stop (list): Stemmed version of the stopwords list.
    - cleaned_texts (list): A list storing the processed and cleaned text data.
    - result (list): A list to store the calculated TF-IDF ratings.

    Methods:
    - cleanup(text: Text) -> Text: Cleans and preprocesses input text for analysis.
    - rate(texts: Iterable[Text]) -> np.array: Computes TF-IDF ratings for a collection of input texts.

    Usage Example:
    rater = TfIdfRater()
    ratings = rater.rate(["This is an example text.", "Another example for rating."])
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(use_idf=True, min_df=3)
        self.stemmer = PorterStemmer()
        
        self.stop = stopwords.words('english') + ['hey', 'hi']  # stopwords
        self.stemmed_stop = [self.stemmer.stem(x) for x in self.stop]  # stemmed stopwords
        
        self.cleaned_texts = []
        self.result = []
        
        # Regular expression for cleanup
        self.re_form = r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"

    def cleanup(self, text: Text) -> Text:
        """
        Perform text cleaning and preprocessing operations on the input text.

        Parameters:
        text (Text): The input text to be cleaned and preprocessed.

        Returns:
        Text: The preprocessed and cleaned text output.

        Method Overview:
        - Lowercasing the text to ensure uniformity in handling text data.
        - Stripping leading and trailing spaces from the text to eliminate unnecessary whitespace.
        - Removing punctuation and links using a predefined regular expression pattern.
        - Normalizing redundant whitespace by consolidating multiple spaces into single spaces.
        - Stemming the words using the provided stemmer instance.
        - Removing common stop words to refine the text for further analysis.

        Example:
        cleaned_text = cleanup("The quick brown fox jumps over the lazy dog.")
        # Output: 'quick brown fox jump over lazy dog'

        Notes:
        - This function is designed to prepare text data for natural language processing tasks.
        - The stemming and stop word removal steps may result in a loss of information based on specific application requirements.
        """
        text = text.lower()  # Convert to lowercase
        text = text.strip()  # Remove leading and trailing spaces
        text = re.sub(self.re_form, "", text)  # Remove punctuation and links
        text = re.sub(' +', ' ', text)  # Remove redundant spaces
        text = ' '.join([self.stemmer.stem(word) for word in text.split(' ')])  # Stemming
        text = " ".join([word for word in text.split() if word not in self.stemmed_stop])  # Remove stopwords
        return text

    def rate(self, texts: Union[np.array, list]) -> np.array:
        """
        Cleans the input texts, fits a TF-IDF vectorizer, and calculates the mean TF-IDF score for each text, resulting in a rating.

        Parameters:
        texts (Iterable[Text]): A collection of input texts to be rated.

        Returns:
        np.array: An array containing the TF-IDF rating for each input text.

        Method Overview:
        - Cleans each input text using the cleanup method and stores the cleaned texts.
        - Fits a TF-IDF vectorizer to the cleaned texts to prepare for vectorization and TF-IDF score calculation.
        - Calculates the mean TF-IDF score for each cleaned text, resulting in a rating.
        - Scales the calculated ratings based on the maximum IDF value observed across all texts.

        Example:
        rated_texts = rate(["This is an example text.", "Another example for rating."])
        # Output: array([0.63, 1.00])

        Notes:
        - This function processes the input texts, calculates their TF-IDF ratings, and ensures that the ratings are scaled appropriately based on the maximum IDF value observed across all texts.
        """
        for text in texts:
            cleaned_text = self.cleanup(text)
            self.cleaned_texts.append(cleaned_text)

        self.vectorizer.fit(self.cleaned_texts)

        for cleaned_text in self.cleaned_texts:
            out = self.vectorizer.transform([cleaned_text]).mean()
            self.result.append(out)

        mx_idf = max(self.result)
        self.result = np.array(self.result) / mx_idf  # Scale the ratings
        return self.result

if __name__ == '__main__':
    pass