import os.path

# Path variables
PATH = os.path.dirname(os.path.realpath(__file__))
DB_DIR = PATH + "/db/"
DB_PATH = DB_DIR + "database.sqlite"
MODELS_DIR = PATH + "/models/"
IMG_DIR = PATH + "/img/"

# List of words to analyze
WORDS_LIST = [
    "beauty", "beautiful", "beautifully", "ugliness", "ugly"
]

# Models
MODELS = [
    {
        name: "word2vec_bigrams.model.gz",
        bigrams: True,
        word2vec: True,
        source: "pitchfork",
        method: "CBOW"
    },
    {
        name: "word2vec.model.gz",
        bigrams: False,
        word2vec: True,
        source: "pitchfork",
        method: "CBOW"
    },
    {
        name: "word2vec_sg_bigrams.model.gz",
        bigrams: True,
        word2vec: True,
        source: "pitchfork",
        method: "skip-gram"
    },
    {
        name: "word2vec_sg.model.gz",
        bigrams: False,
        word2vec: True,
        source: "pitchfork",
        method: "skip-gram"
    }
]
