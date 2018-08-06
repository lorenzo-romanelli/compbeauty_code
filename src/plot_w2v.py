import environment
from datetime import datetime
from gensim.models import KeyedVectors
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
    plotsdir = imgdir + "embeddings/"
    lab = labels[0]
    src = model["source"]
    mth = model["method"]
    now = datetime.now().strftime("%y%m%d-%H%M")
    out = plotsdir + lab + "_" + src + "_" + mth + "_" + now
    plt.savefig(out + ".pdf")


def getWordVectors(model, words):
    vectors = [(word, model.wv[word]) for word in words]
    return vectors


def getVectorsOfSimilarWords(model, word, N=10):
    '''
    Returns vectors of the N words most similar to the input word.
    '''
    similar = model.wv.similar_by_word(word, topn=N)
    words = [word[0] for word in similar]
    vectors = getWordVectors(model, similar)
    vectors = [(word, model.wv[word])] + vectors
    return vectors


if __name__ == "__main__":
    modelsdir = environment.MODELS_DIR
    for model in environment.MODELS:
        if not model["bigrams"]: continue
        
        modelname = modelsdir + model["name"]
        try:
            if model["binary"]:
                w2v_model = KeyedVectors.load_word2vec_format(modelname, binary=True, limit=500000)
            else:
                w2v_model = KeyedVectors.load(modelname)
        except:
            print("Couldn't load the model {}.".format(model["name"]))
            continue
        
        for word in environment.WORDS_LIST:
            vecs = getVectorsOfSimilarWords(w2v_model, word, N=500)
            tsnePlot(vecs, model)
            print("Plot for word \"{}\" done.".format(word))
        del(w2v_model)
