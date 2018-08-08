import environment
import sys
import sqlite3
from nltk import sent_tokenize, RegexpTokenizer
from gensim.models.phrases import Phrases, Phraser

def getReviews():
    dbpath = environment.DB_PATH
    conn = sqlite3.connect(dbpath)
    c = conn.cursor()
    reviews = c.execute("SELECT content FROM content;")
    return sum(reviews, ())

def getSentences(review):
    return sent_tokenize(review)

def getUnigrams(sentence, tolower=False):
    tokenizer = RegexpTokenizer(r'\w+')
    sentence = sentence.lower() if tolower else sentence
    return tokenizer.tokenize(sentence)

def getBigrams(sentences, min_count=20):
    bigrams = Phrases(sentences, min_count=min_count)
    return Phraser(bigrams)

def main():
    reviews = getReviews()
    stream = []
    for review in reviews:
        for sentence in getSentences(review):
            stream.append(getUnigrams(sentence, tolower=True))
    bigrams = getBigrams(stream)
    trigrams = getBigrams(bigrams[stream])

    outpath = environment.MODELS_DIR + "phrases/"
    bigrams.save(outpath+"bigrams.phr")
    print("Bigrams saved.")
    trigrams.save(outpath+"trigrams.phr")
    print("Trigrams saved.")

if __name__ == "__main__":
    main()