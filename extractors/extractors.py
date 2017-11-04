import pandas as pd
from sklearn.base import TransformerMixin
import re
from nltk import pos_tag
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet as wn
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict





class Word(TransformerMixin):
    
    def __init__(self, offset=0):
        self.offset = offset
    
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            index = rowdata["index"]
            words = word_tokenize(rowdata["sentence"])
            word = words[index+self.offset] if (index+self.offset) in range(len(words)) else "NONE"
            result.append(word)
        return result


class WordLen(TransformerMixin):
        
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            result.append(len(rowdata["word"]))
        return result


class ContainsChar(TransformerMixin):
        
    def __init__(self, char):
        self.char = char
        
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            result.append(self.char in rowdata["word"])
        return result



class RegexMatches(TransformerMixin):

    def __init__(self, pattern):
        self.pattern = pattern
        
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            result.append(True if re.findall(self.pattern, rowdata["word"]) else False)
        return result


class POSTag(TransformerMixin):
    
    def __init__(self, offset=0):
        self.offset = offset
    
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            index = rowdata["index"]
            pos_tags = pos_tag(word_tokenize(rowdata["sentence"]))
            if (index+self.offset) in range(len(pos_tags)):
                tag = pos_tags[index+self.offset][1]
            else:
                if self.offset > 0:
                    tag = "</START>"
                else:
                    tag = "<START>"
            result.append(tag)
        return result


class Label(TransformerMixin):
        
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            result.append(rowdata["label"])
        return result


class Stem(TransformerMixin):
    
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
        
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            result.append(self.stemmer.stem(rowdata["word"]))
        return result


class SentenceLength(TransformerMixin):
        
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            result.append(len(word_tokenize(rowdata["sentence"])))
        return result


class DistinctWordsInSentence(TransformerMixin):
        
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            result.append(len(set(word_tokenize(rowdata["sentence"]))))
        return result


import editdistance

class EditDistance(TransformerMixin):
    
    def __init__(self, target_word):
        self.target_word = target_word
        
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            result.append(editdistance.eval(rowdata["word"], self.target_word))
        return result


class NumberSynonyms(TransformerMixin):
        
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            result.append(len(wn.synsets(rowdata["word"])))
        return result


class NumberHyponyms(TransformerMixin):
        
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            try:
                hypo = np.array([s.hyponyms() for s in wn.synsets(rowdata["word"])])
                result.append(len(wn.synsets(rowdata["word"])))
            except:
                result.append(0)
        return result


class NumberHypernyms(TransformerMixin):
        
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            try:
                hypo = np.array([s.hypernyms() for s in wn.synsets(rowdata["word"])])
                result.append(len(wn.synsets(rowdata["word"])))
            except:
                result.append(0)
        return result


class NumberPronounciations(TransformerMixin):
    
    def __init__(self):
        self.cmudict = cmudict.dict()
        
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            try:
                result.append(len(self.cmudict[rowdata["word"]]))
            except KeyError:
                result.append(0)
        return result


class NumberVowels(TransformerMixin):
       
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            result.append(len(re.findall(r"([aeiou])", rowdata["word"])))
        return result


class NumberConsonants(TransformerMixin):
       
    def transform(self, X):
        result = []
        for index, rowdata in X.iterrows():
            result.append(len(re.findall(r"([bcdfghjklmnpqrstvwxyz])", rowdata["word"])))
        return result