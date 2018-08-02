import sqlite3
import environment

path = environment.PATH
databasedir = environment.DB_DIR
databasepath = environment.DB_PATH
modelsdir = environment.MODELS_DIR

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
    genres = getGenres()
    for genre in genres:
        print(genre)