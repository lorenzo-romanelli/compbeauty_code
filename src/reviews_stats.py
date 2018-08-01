import sqlite3
import os.path

path = os.path.dirname(os.path.realpath(__file__))
databasedir = path + "/db/"
databasepath = databasedir + "database.sqlite"
modelsdir = path + "/models/"

def getGenres():
    with sqlite3.connect(databasepath) as conn:
        #conn.text_factory = bytes
        c = conn.cursor()
        query = '''
        SELECT      genre,  COUNT(*) as ct 
        FROM        genres 
        GROUP BY    genre 
        ORDER BY    ct      DESC
        '''
        genres = c.execute(query)
        return genres

if __name__ == "__main__":
    path = os.path.dirname(os.path.realpath(__file__))
    databasedir = path + "/db/"
    modelsdir = path + "/models/"
    genres = getGenres()
    for genre in genres:
        print(genre)