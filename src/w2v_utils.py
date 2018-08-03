import environment
import csv
from gensim.models import KeyedVectors

def loadModel_kv(path, model):
    if model["binary"]:
        return KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        return KeyedVectors.load(path)


def getSimilarWords(model, word, nitems=10):
    return model.most_similar(positive=[word], topn=nitems)


def saveSimilarWords(out, target, words):
    with open(out, "wb") as f:
        writer = csv.writer(f, delimiter=",")
        for word in words:
            writer.writerow(word)


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
            similar = getSimilarWords(kv_model, word)
            path = environment.TXT_DIR
            out = path + word + "_" + model["source"] + "_" + model["method"] + "_sim.csv"
            saveSimilarWords(out, word, similar)
            print("Written file for {}.".format(word))
        del(model)

