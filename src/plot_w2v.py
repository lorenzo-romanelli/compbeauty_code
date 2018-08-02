import environment
from datetime import datetime
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def tsnePlot(vectors, model):
    '''
    Plots the word embedding using TSNE dimensionality reduction.
    '''
    labels, tokens = zip(*vectors)

    tsne_model = TSNE(n_components=2, random_state=0, metric="cosine", init="random")
    tokens_reduced = tsne_model.fit_transform(tokens)
    
    x_coords = tokens_reduced[:,0]
    y_coords = tokens_reduced[:,1]
    
    plt.figure(figsize=(25, 30))
    plt.scatter(x_coords[0], y_coords[0], c='r')
    plt.scatter(x_coords[1:], y_coords[1:])
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    for label, x, y in zip(labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    
    imgdir = environment.IMG_DIR
    lab = labels[0]
    mth = model["method"]
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    out = imgdir + lab + "_" + mth + "_" + now
    plt.savefig(out + ".pdf")


def getWordVectors(model, words):
    vectors = [(word[0], model.wv[word[0]]) for word in words]
    return vectors


def getVectorsOfSimilarWords(model, word, N=10):
    '''
    Returns vectors of the N words most similar to the input word.
    '''
    similar = model.wv.similar_by_word(word, topn=N)
    vectors = getWordVectors(model, similar)
    vectors = [(word, model.wv[word])] + vectors
    return vectors

if __name__ == "__main__":
    modelsdir = environment.MODELS_DIR
    #modelname = "word2vec_bigrams.model.gz"
    #model = Word2Vec.load(modelsdir + modelname)
    for model in environment.MODELS:
        if not model["bigrams"]: continue
        w2v_model = Word2Vec.load(modelsdir + model["name"])
        for word in environment.WORDS_LIST:
            vecs = getVectorsOfSimilarWords(w2v_model, word, N=500)
            tsnePlot(vecs, model)
            print("Plot for word \"{}\" done.".format(word))
