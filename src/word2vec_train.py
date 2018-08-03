# -*- coding: utf-8 -*-
import os.path
import environment
import sqlite3
import utils
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import tokenize

class SQLiteCorpus(object):
    def __init__(self, path_to_db, limit=None, random=False, bigrams=False):
        self.dbpath = path_to_db
        self.limit = limit
        self.random = random
        self.bigrams = self.computeBigrams() if bigrams else None
 
    def __iter__(self):
        for review in self.getReviews():
            review = self.preprocess(review[0])
            for sentence in utils.splitIntoSentences(review):
                unigrams = list(tokenize(sentence))
                yield self.bigrams[unigrams] if self.bigrams else unigrams
    
    def getReviews(self):
        '''
        Get reviews from the db (optionally limiting and randomizing)
        '''
        with sqlite3.connect(self.dbpath) as conn:
            conn.text_factory = bytes
            c = conn.cursor()
            query = "SELECT content FROM content"
            query += " ORDER BY RANDOM()" if self.random else ""
            query += " LIMIT {}".format(self.limit) if self.limit else ""
            reviews_list = c.execute(query)
        return reviews_list

    def computeBigrams(self):
        reviews = self.getReviews()
        sentences = []
        for review in reviews:
            review = self.preprocess(review[0])
            sentences += utils.splitIntoSentences(review)
        phrases = Phrases([tokenize(sentence) for sentence in sentences])
        return Phraser(phrases)

    def preprocess(self, text):
        # replace fancy characters
        text = utils.replaceChars(text)
        text = utils.expandContractions(text=text.lower())
        return text

if __name__ == "__main__":
    path = environment.PATH
    databasedir = environment.DB_DIR
    databasepath = environment.DB_PATH
    modelsdir = environment.MODELS_DIR
    print(path)
    
    # Train simple W2V model
    modelname = "word2vec_sg.model"
    if not os.path.isfile(modelsdir + modelname):
        corpus = SQLiteCorpus(databasepath)  # a memory-friendly iterator
        model = Word2Vec(corpus, sg=1, workers=4)
        model.save(modelsdir + modelname)
        print("W2V model trained.")
    else:
        print("W2V model already exists!")

    # Train W2V model using bigrams
    modelname = "word2vec_sg_bigrams.model"
    if not os.path.isfile(modelsdir + modelname):
        corpus = SQLiteCorpus(databasepath, bigrams=True)
        model = Word2Vec(corpus, sg=1, workers=4)
        model.save(modelsdir + modelname)
        print("W2V bigrams model trained.")
    else:
        print("W2V bigrams model already exists!")