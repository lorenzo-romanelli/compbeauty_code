import os.path

# Path variables
PATH = os.path.dirname(os.path.realpath(__file__))
DB_DIR = PATH + "/db/"
DB_PATH = DB_DIR + "database.sqlite"
MODELS_DIR = PATH + "/models/"
IMG_DIR = PATH + "/img/"
TXT_DIR = PATH + "/text/"

# List of words to analyze
WORDS_LIST = [
    "beauty", "beautiful", "beautifully", 
    "ugliness", "ugly",
    "aesthetic", "aesthetics"
]

# Models
MODELS = [
    {
        "name": "word2vec_bigrams.model.gz",
        "bigrams": True,
        "word2vec": True,
        "source": "pitchfork",
        "method": "CBOW",
        "binary": False
    },
    {
        "name": "word2vec.model.gz",
        "bigrams": False,
        "word2vec": True,
        "source": "pitchfork",
        "method": "CBOW",
        "binary": False
    },
    {
        "name": "word2vec_sg_bigrams.model.gz",
        "bigrams": True,
        "word2vec": True,
        "source": "pitchfork",
        "method": "skip-gram",
        "binary": False
    },
    {
        "name": "word2vec_sg.model.gz",
        "bigrams": False,
        "word2vec": True,
        "source": "pitchfork",
        "method": "skip-gram",
        "binary": False
    },
    {
        "name": "GoogleNews.bin.gz",
        "bigrams": True,
        "word2vec": True,
        "source": "google",
        "method": "skip-gram",
        "binary": True
    }
]
