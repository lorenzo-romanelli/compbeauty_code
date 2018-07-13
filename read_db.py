import numpy as np
#import networkx as nx
#import matplotlib.pyplot as plt
import sqlite3
import utils
from gensim.corpora import Dictionary
from gensim.utils import tokenize
from gensim.matutils import corpus2csc
from gensim.models import Phrases
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

def stem_document(document):
    '''
    Stem a document.
        <document> should be a list of tokens.
    '''
    return [stem_word(word) for word in document]

def stem_word(word):
    '''
    Stem a word using Porter stemmer algorithm.
    '''
    porter_stemmer = PorterStemmer()
    return porter_stemmer.stem(word)

def get_close_words(word_matrix, word_id, truncate, min_score=-1):
    n_elements = word_matrix.shape[0]
    word_vectors = np.empty((0, n_elements), dtype='f')
    word_vec = word_matrix[word_id]
    idxs = np.argsort(word_vec)[::-1]
    truncate = truncate if truncate <= n_elements else n_elements
    idxs = idxs[:truncate]
    words = [dictionary[word_id]]
    words.extend([dictionary[idx] for idx in idxs])
    word_vectors = np.append(word_vectors, word_matrix[idxs], axis=0)

    return words, word_vectors

def plot_words_from_cooccurrences(co_occurrences, labels):
    tsne = TSNE(n_components=2, random_state=0, metric="cosine")
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(co_occurrences)
    x_coords = Y[:,0]
    y_coords = Y[:,1]
    plt.scatter(x_coords[0], y_coords[0], c='r')
    plt.scatter(x_coords[1:], y_coords[1:])
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    for label, x, y in zip(labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()

#--------------------#
#       TABLES       #
#--------------------#
#   - reviews
#   - artists
#   - genres
#   - labels
#   - years
#   - content

# get reviews from db
db = "database.sqlite"
conn = sqlite3.connect(db)
c = conn.cursor()
limit = "LIMIT 1000"
reviews_list = c.execute("SELECT * FROM content ORDER BY RANDOM() {}".format(limit))

# replace fancy characters
reviews = [utils.replaceChars(r[1]) for r in reviews_list]

# expand contractions, remove stopwords, tokenize, stem
corpus = list()
count = 200
for i, review in enumerate(reviews):
    text = utils.expandContractions(text=review.lower())
    text_no_stopwords = remove_stopwords(text)
    tokenized_text = list(tokenize(text_no_stopwords))
    # if we don't want to stem, uncomment next line 
    # and comment the line below
    #corpus.append(tokenized_text)
    stemmed_text = stem_document(tokenized_text)
    if (stem_word("beauty") in stemmed_text):
        corpus.append(stemmed_text)
        count -= 1
    if count < 0:
        break
    #print("Analyzed {} out of {} reviews...".format(i+1, len(reviews)))

dictionary = Dictionary(corpus)

# word = "beauty"
word = stem_word("beauty")
beauty_id = dictionary.token2id[word]
bow_corpus = [dictionary.doc2bow(line) for line in corpus]
term_doc_mat = corpus2csc(bow_corpus)
term_term_mat = np.dot(term_doc_mat, term_doc_mat.T)
similarities = cosine_similarity(term_term_mat)

min_freq = 10

labels, vectors = get_close_words(word_matrix=similarities,
                                    word_id=beauty_id,
                                    truncate=500)
#print(vectors[0])
plot_words_from_cooccurrences(co_occurrences=vectors, labels=labels)
# def get_close_words(term_doc_matrix, word_id, min_score):
#     term_term_mat = np.dot(term_doc_mat, term_doc_mat.T)
#     word_vector = term_term_mat.getrow(word_id).toarray()[0]
#     close_words = [(idx, score) if score >= min_score for idx, score in enumerate(word_vector)]
#     return close_words



# TODO:
#   - removing stopwords?
#   - entity recognition?
#   - stemming?
#   - bigrams/trigrams? before/after stemming?
