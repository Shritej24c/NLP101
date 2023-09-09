from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup
import re
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Token
import numpy as np
from nltk.stem.porter import PorterStemmer
import os

class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    """
    A transformer class for preprocessing text data using spaCy.
    
    This class is designed to be used in a sklearn pipeline. It uses the spaCy library to 
    preprocess text data by performing a series of operations such as tokenization, lowercasing, 
    stop word removal, punctuation removal, email and URL removal, stemming, lemmatization.
    """

    
    def __init__(self, model, *, batch_size = 64, lemmatize=True, lower=True, remove_stop=True, 
                remove_punct=True, remove_email=True, remove_url=True, remove_num=False, stemming = False,
                add_user_mention_prefix=True, remove_hashtag_prefix=False):

        """
        Initialize the preprocessor.
        
        Parameters
        ----------
        model : spaCy model
            The spaCy model to use for preprocessing the text data.
        batch_size : int, optional (default=64)
            The batch size to use when processing the text data in batches.
        lemmatize : bool, optional (default=True)
            Whether to perform lemmatization on the text data.
        lower : bool, optional (default=True)
            Whether to lowercase the text data.
        remove_stop : bool, optional (default=True)
            Whether to remove stop words from the text data.
        remove_punct : bool, optional (default=True)
            Whether to remove punctuation from the text data.
        remove_email : bool, optional (default=True)
            Whether to remove email addresses from the text data.
        remove_url : bool, optional (default=True)
            Whether to remove URL addresses from the text data.
        remove_num : bool, optional (default=False)
            Whether to remove numbers from the text data.
        stemming : bool, optional (default=False)
            Whether to perform stemming on the text data.
        add_user_mention_prefix : bool, optional (default=True)
            Whether to change the behavior of the spaCy 
            tokenizer to treat the @ symbol as a prefix.
        remove_hashtag_prefix : bool, optional (default=False)
            Whether to change the behavior of the spaCy 
            tokenizer to no longer treat the # symbol as a prefix.
        """
                
        self.model = model
        self.batch_size = batch_size
        self.remove_stop = remove_stop
        self.remove_punct = remove_punct
        self.remove_num = remove_num
        self.remove_url = remove_url
        self.remove_email = remove_email
        self.lower = lower
        self.add_user_mention_prefix = add_user_mention_prefix
        self.remove_hashtag_prefix = remove_hashtag_prefix

        if lemmatize and stemming:
            raise ValueError("Only one of 'lammetize' and 'stemming' can be True.")

        self.lemmatize = lemmatize
        self.stemming = stemming


    def basic_clean(self, text):
        """Clean the input text by removing HTML tags and line breaks.
        
        Parameters
        ----------
        text : str
            The input text to clean.
            
        Returns
        -------
        str
            The cleaned text.
        """
        # Use BeautifulSoup to remove HTML tags
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()

        # Replace line breaks with spaces
        return re.sub(r'[\n\r]', ' ', text)

    def spacy_preprocessor(self, texts):
        """        
        Preprocesses text data using spaCy's NLP library.
        Removes specified items such as stop words, punctuation, numbers, URLs, and emails.
        Also lemmatizes, stems, lowercase or keeps the text as is based on specified parameters.

        Parameters
        ----------
        texts : list or numpy array
            A list or numpy array of text data to preprocess

        Returns
        -------
        final_result : list
            A list of preprocessed text data
        """

        final_result = []
        nlp = spacy.load(self.model)

        # Disable unnecessary pipelines in spaCy model
        if self.lemmatize:
            # Disable parser and named entity recognition
            disabled = nlp.select_pipes(disable= [ 'parser', 'ner'])
        else:
            # Disable tagger, parser, attribute ruler, lemmatizer and named entity recognition
            disabled = nlp.select_pipes(disable= ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])

        # Add @ as a prefix so that we can separate the word from @
        prefixes = list(nlp.Defaults.prefixes)
        if self.add_user_mention_prefix:
            prefixes += ['@']

        # Remove # as a prefix so that we can keep hashtags and words together
        if self.remove_hashtag_prefix:
            prefixes.remove(r'#')

        # Compile prefix regex based on selected prefixes
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        nlp.tokenizer.prefix_search = prefix_regex.search

        # Create a matcher to remove specified items from text data
        matcher = Matcher(nlp.vocab)
        if self.remove_stop:
            matcher.add("stop_words", [[{"is_stop" : True}]])
        if self.remove_punct:
            matcher.add("punctuation",[ [{"is_punct": True}]])
        if self.remove_num:
            matcher.add("numbers", [[{"like_num": True}]])
        if self.remove_url:
            matcher.add("urls", [[{"like_url": True}]])
        if self.remove_email:
            matcher.add("emails", [[{"like_email": True}]])

        # Determine number of cores to use in parallel processing
        num_cores = os.cpu_count()
        if num_cores < 3:
            use_cores = 1
        else:
            use_cores = num_cores // 2 + 1

        # Set custom attribute to track if a token should be removed
        Token.set_extension('is_remove', default=False, force=True, )

        cleaned_text = []

        # Process text data in parallel using spaCy's nlp.pipe()
        np.random.seed(0)
        for doc in nlp.pipe(texts, batch_size=self.batch_size,n_process=use_cores ):
            matches = matcher(doc)
            
            # Mark tokens for removal based on match results
            for _, start, end in matches:
                for token in doc[start:end]:
                    token._.is_remove = True

            ## Join the preprocessed text string based on the selected method (lemma, stem, text)
            if self.lemmatize:                     
                text = ' '.join(token.lemma_ for token in doc if (token._.is_remove==False))
            elif self.stemming:
                text = ' '.join(PorterStemmer().stem(token.text) for token in doc if (token._.is_remove==False))
            else:
                text = ' '.join(token.text for token in doc if (token._.is_remove==False))
                                    
            if self.lower:
                text=text.lower()
            cleaned_text.append(text)
        return cleaned_text

    def fit(self, X, y=None):
        """
        Dummy method for compatibility with sklearn's API. 
        Fits the data and returns self.
        
        Parameters
        ----------
        X : numpy array or list
            A list or array of text data.
        y : None
            This argument is ignored and only present for compatibility with sklearn API.
        
        Returns
        -------
        self : object
            Returns self
        """
        return self

    def transform(self, X, y=None):
        """
        Transform the input data by preprocessing the text using spaCy's NLP library.
        
        Parameters
        ----------
        X : list or numpy array
            A list of text data to preprocess
        y : object, optional
            Ignored. Present for API compatibility by convention.

        Returns
        -------
        final_result : list
            A list of preprocessed text data
        
        """
        try:
            # Check if input is a list or numpy array
            if not isinstance(X, (list, np.ndarray)):
                raise TypeError(f'Expected list or numpy array, got {type(X)}')

            # Clean the text data
            x_clean = [self.basic_clean(text).encode('utf-8', 'ignore').decode() for text in X]

            # Preprocess the text data using spaCy
            x_clean_final = self.spacy_preprocessor(x_clean)

            return x_clean_final
        except Exception as error:
            print(f'An exception occurred: {repr(error)}')

