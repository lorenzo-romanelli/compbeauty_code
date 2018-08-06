import environment
import csv
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE


def loadModel_kv(path, model):
    if model["binary"]:
        return KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        return KeyedVectors.load(path)


def loadSimilarWords(inpath):
    with open(inpath, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        return [w for w in reader]


def getWordVectors(model, words):
    vectors = [(word, model.wv[word]) for word in words]
    return vectors


def getSimilarWords(model, word, nitems=10):
    return model.most_similar(positive=[word], topn=nitems)


def saveSimilarWords(outpath, target, words):
    with open(outpath, "wb") as f:
        writer = csv.writer(f, delimiter=",")
        for word in words:
            w = word[0].encode("utf-8")
            s = word[1]
            writer.writerow([w, s])


def tsneReduce(word_vectors):
    labels, tokens = zip(*word_vectors)
    tsne_model = TSNE(n_components=2, random_state=0, metric="cosine", init="random")
    tokens_reduced = tsne_model.fit_transform(tokens)
    return zip(labels, tokens_reduced)


if __name__ == "__main__":
    models = environment.MODELS
    modelsdir = environment.MODELS_DIR

    for model in models:
        if not model["bigrams"]: continue
        name = model["name"]
        path = modelsdir + name
        try:
            kv_model = loadModel_kv(path, model)
        except:
            print("Couldn't load the model {}.".format(name))
            continue
        for word in environment.WORDS_LIST:
            similar = getSimilarWords(kv_model, word, 500)
            path = environment.TXT_DIR
            out = path + word + "_" + model["source"] + "_" + model["method"] + "_sim.csv"
            saveSimilarWords(out, word, similar)
            print("Written file for {}.".format(word))
        del(model)

