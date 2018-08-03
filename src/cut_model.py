import environment
from gensim.models import KeyedVectors

def cut(modelpath, outpath, nitems):
    model = KeyedVectors.load_word2vec_format(modelpath, binary=True, limit=nitems)
    model.save_word2vec_format(fname=outpath, binary=True)

if __name__ == "__main__":
    modelsdir = environment.MODELS_DIR
    modelname = "GoogleNews_full.bin.gz"
    out = "bin.GoogleNews"
    limit = 500000
    cut(modelsdir + modelname, modelsdir + out, limit)
    print("Model cut.")
    
