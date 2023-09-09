from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import spacy
import re
import sys
import os
import custom_preprocessor_mod as cp
from pathlib import Path


class ManualFeatures(TransformerMixin, BaseEstimator):

    """A transformer class for extracting manual features from text data.
    
    This class is designed to be used in a scikit-learn pipeline. It uses the spaCy
    library to extract a variety of manual features from text data, such as 
    part-of-speech (POS) features, named entity recognition (NER) features, 
    and count-based features.
    """



    def __init__(self, spacy_model, pos_features = True, ner_features = True, count_features = True):

        """
        Initialize the feature extractor.

        Parameters
        ----------
        spacy_model : str
            The name of the spaCy model to use for feature extraction.
        pos_features : bool, optional (default=True)
            Whether to extract part-of-speech (POS) features from the text data.
        ner_features : bool, optional (default=True)
            Whether to extract named entity recognition (NER) features from the text data.
        count_features : bool, optional (default=True)
            Whether to extract count-based features from the text data.
        """

        self.spacy_model = spacy_model
        self.pos_features = pos_features
        self.ner_features = ner_features
        self.count_features = count_features

    def get_cores(self):
        """
        Get the number of CPU cores to use in parallel processing.
        """
        # Get the number of CPU cores available on the system.
        num_cores = os.cpu_count()
        if num_cores < 3:
            use_cores = 1
        else:
            use_cores = num_cores // 2 + 1
        return num_cores

    def get_pos_features(self, cleaned_text):
        """
        Extract part-of-speech (POS) features from the cleaned text.
        
        Parameters:
        cleaned_text (list): A list of cleaned text strings.
        
        Returns:
        numpy.ndarray: A 2D numpy array with shape (len(cleaned_text), 4) containing the count of nouns, 
        auxiliaries, verbs, and adjectives for each text in the input list.
        """
        nlp = spacy.load(self.spacy_model)
        noun_count = []
        aux_count = []
        verb_count = []
        adj_count =[]
        
        # Disable the lemmatizer and NER pipelines for improved performance
        with nlp.disable_pipes(*['lemmatizer', 'ner']):
            n_process = self.get_cores()
            for doc in nlp.pipe(cleaned_text, batch_size=1000, n_process=n_process):\
                # Extract nouns, auxiliaries, verbs, and adjectives from the document
                nouns = [token.text for token in doc if token.pos_ in ["NOUN","PROPN"]]
                auxs =  [token.text for token in doc if token.pos_ in ["AUX"]]
                verbs =  [token.text for token in doc if token.pos_ in ["VERB"]]
                adjectives =  [token.text for token in doc if token.pos_ in ["ADJ"]]
                
                # Store the count of each type of word in separate lists
                noun_count.append(len(nouns))
                aux_count.append(len(auxs))
                verb_count.append(len(verbs))
                adj_count.append(len(adjectives))
        
        # Stack the count lists vertically to form a 2D numpy array
        return np.transpose(np.vstack((noun_count, aux_count, verb_count, adj_count)))



    def get_ner_features(self, cleaned_text):
        """
        Extract named entity recognition (NER) features from the cleaned text.
        
        Parameters:
        cleaned_text (list): A list of cleaned text strings.
        
        Returns:
        numpy.ndarray: A 2D numpy array with shape (len(cleaned_text), 1) containing 
        the count of named entities for each text in the input list.
        """
        nlp = spacy.load(self.spacy_model)
        count_ner = []
        
        # Disable the tok2vec, tagger, parser, attribute ruler, and lemmatizer pipelines for improved performance
        with nlp.disable_pipes(*['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer']):
            for doc in nlp.pipe(cleaned_text, batch_size=1000, n_process=-1):
                ners = [ent.label_ for ent in doc.ents]
                count_ner.append(len(ners))
        
        # Convert the list of NER counts to a 2D numpy array
        return np.array(count_ner).reshape(-1, 1)


    def get_count_features(self, cleaned_text):

        """
        Extract count-based features from the cleaned text.
        
        Parameters:
        cleaned_text (list): A list of cleaned text strings.
        
        Returns:
        numpy.ndarray: A 2D numpy array with shape (len(cleaned_text), 6) containing the count of words, characters, characters without spaces, average word length, count of digits, and count of numbers for each text in the input list.
        """

        list_count_words =[]
        list_count_characters =[]
        list_count_characters_no_space =[]
        list_avg_word_length=[]
        list_count_digits=[]
        list_count_numbers=[]
        for sent in cleaned_text:
          # Remove all digits followed by spaces from the text
            words = re.sub(r'\d+\s','',sent)

            # Extract all numbers from the text
            numbers = re.findall(r'\d+', sent)

            count_word = len(words.split())
            count_char = len(words)
            count_char_no_space = len(''.join(words.split()))
            avg_word_length = count_char_no_space/(count_word + 1)
            count_numbers = len(numbers)
            count_digits = len(''.join(numbers))

            list_count_words.append(count_word)
            list_count_characters.append(count_char)
            list_count_characters_no_space.append(count_char_no_space)
            list_avg_word_length.append(avg_word_length)
            list_count_digits.append(count_digits)
            list_count_numbers.append(count_numbers)

        # Stack the count lists vertically to form a 2D numpy array
        count_features = np.vstack((list_count_words, list_count_characters,
                                  list_count_characters_no_space, list_avg_word_length,
                                  list_count_digits,list_count_numbers ))
        return np.transpose(count_features)


    def fit(self, X, y=None):
        """
        Fit the feature extractor to the input data.
        
        This method does not actually do any fitting, as the feature extractor is stateless.
        It simply returns the instance of the class.
        
        Parameters:
        X (list or numpy.ndarray): The input data.
        y (list or numpy.ndarray, optional): The target labels. Not used in this implementation.
        
        Returns:
        FeatureExtractor: The instance of the class.
        """
        return self


    def transform(self, X, y=None):
        """
        Transform the input data into a set of features.
        
        Parameters:
        X (list or numpy.ndarray): The input data.
        y (list or numpy.ndarray, optional): The target labels. Not used in this implementation.
        
        Returns:
        tuple: A tuple containing a 2D numpy array with shape (len(X), num_features) where num_features is the number of features extracted and a list of feature names.
        
        Raises:
        TypeError: If the input data is not a list or numpy array.
        Exception: If an error occurs while transforming the data into features.
        """
        try:
            # Check if the input data is a list or numpy array
            if not isinstance(X, (list, np.ndarray)):
                raise TypeError(f"Expected list or numpy array, got {type(X)}")

            # Initialize the preprocessor
            preprocessor1 = cp.SpacyPreprocessor(model='en_core_web_sm', lemmatize=False, lower=False,
                                                remove_stop=False)
            preprocessor2 = cp.SpacyPreprocessor(model='en_core_web_sm', lemmatize=False, lower=False,
                                                remove_stop=False, remove_punct=False)

            feature_names = []
            if self.pos_features or self.ner_features:
                cleaned_x_count_ner_pos = preprocessor2.fit_transform(X)
            
            if self.count_features:
                cleaned_x_count_features = preprocessor1.fit_transform(X)
                count_features = self.get_count_features(cleaned_x_count_features)
                feature_names.extend(['count_words', 'count_characters',
                                      'count_characters_no_space', 'avg_word_length',
                                      'count_digits', 'count_numbers'])
            else:
                count_features = np.empty(shape=(0, 0))
            
            if self.pos_features:
                pos_features = self.get_pos_features(cleaned_x_count_ner_pos)
                feature_names.extend(['noun_count', 'aux_count', 'verb_count', 'adj_count'])
            else:
                pos_features = np.empty(shape=(0, 0))
            
            if self.ner_features:
                ner_features = self.get_ner_features(cleaned_x_count_ner_pos)
                feature_names.extend(['ner'])
            else:
                ner_features = np.empty(shape=(0, 0))
            
            # Stack the feature arrays horizontally to form a single 2D numpy array
            return np.hstack((count_features, ner_features, pos_features)), feature_names

        except Exception as error:
            print(f'An exception occured: {repr(error)}')
