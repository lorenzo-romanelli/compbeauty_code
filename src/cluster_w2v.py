from __future__ import print_function
import environment
import w2v_utils as w2v
import plot_w2v as w2vp
import sys
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors


def kmeansCluterWords(word_vectors, N):
    labels, X = zip(*word_vectors)
    kmeans = KMeans(n_clusters=N, random_state=0).fit(X)
    return zip(labels, kmeans.labels_)


def main():
    csvdir = environment.TXT_DIR
    modelsdir = environment.MODELS_DIR
    models = environment.MODELS
    word = sys.argv[1]
    source = sys.argv[2]
    N = int(sys.argv[3])
    if len(sys.argv) == 5:
        method = sys.argv[4]
    else:
        method = "skip-gram"
    for m in models:
        if m["source"] == source and m["method"] == method and m["bigrams"]:
            name = m["name"]
            meta = m
            break
    model = w2v.loadModel_kv(modelsdir + name, meta)
    print("Loaded {}".format(name))

    csvpath = csvdir + word + "_" + source + "_" + method + "_sim.csv"
    words = w2v.loadSimilarWords(csvpath)
    words = [unicode(w[0], "utf8") for w in words]
    vectors = w2v.getWordVectors(model, words)
    print("Loaded vectors for {}.".format(word))
    clustered = kmeansCluterWords(vectors, N)
    for i in range(N):
        print(i, end=": ")
        for w in clustered:
            if w[1] == i: print(w[0], end=", ")
        print("\n") 


if __name__ == "__main__":
    main()
