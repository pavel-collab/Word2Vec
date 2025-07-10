import sqlite3
from pathlib import Path
from tqdm import tqdm

'''
На вход подается список текстов
'''
def load_data_to_db(text_corpus: list, db_path='texts.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if not Path(db_path).exists():
        raise FileNotFoundError(f'there are no file {Path(db_path).absolute()}')

    # Создаем таблицу для документов, если не существует
    cursor.execute('''CREATE TABLE IF NOT EXISTS documents
                            (id INTEGER PRIMARY KEY, text TEXT)''')
    
    for i, doc in tqdm(enumerate(text_corpus)):
        cursor.execute("INSERT OR IGNORE INTO documents (id, text) VALUES (?, ?)", 
                            (i, doc))
    conn.commit()
    
def import_texts(db_path='texts.db', limit=100):
    if not Path(db_path).exists():
        raise FileNotFoundError(f'there are no file {Path(db_path).absolute()}')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''SELECT text FROM documents LIMIT ?''', (limit,))

    texts = cursor.fetchall()
    '''
    При извлечении из базы данных sqlite нам возвращаются не строки,
    а кортежи строк. В каждом кортеже один единственный элемент --
    сохраненная строка. А мы хотим получить просто список строк.
    '''
    for i in range(len(texts)):
        texts[i] = texts[i][0]
    return texts