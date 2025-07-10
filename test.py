import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess_text, build_vocabulary, generate_skip_grams, generate_cbow_pairs, data_import
from models.skip_gram import SkipGram
from models.cbow import CBOW
import random
from pathlib import Path
from db import load_data_to_db, import_texts
import pickle
from utils import reduce_to_k_dim, plot_embeddings
from utils import debug_print

DB_NAME = 'db.sqlite3'
EMBEDDING_DIM = 100
EPOCH_NUM = 50

debug_print('Start to import data')
if Path(DB_NAME).exists():
    text_corpus = import_texts(DB_NAME, limit=5000)
else:
    text_corpus = data_import(page_limit=5000)
    load_data_to_db(text_corpus, DB_NAME)
    
debug_print('Start to preprocess data')
processed_corpus = preprocess_text(text_corpus)
vocab, index_to_word = build_vocabulary(processed_corpus)
skip_grams = generate_skip_grams(processed_corpus)
cbow_pairs = generate_cbow_pairs(processed_corpus)

# возьмем несколько текстов, чтобы проверить логическую связь слов после обучения
random_text_subset = random.sample(text_corpus, 3)

processed_test_texts = preprocess_text(random_text_subset)
test_vocab, _ = build_vocabulary(processed_test_texts)

debug_print('Start to train skip gram')
try:
    skip_gram_model = SkipGram(len(vocab), EMBEDDING_DIM, vocab)    
    skip_gram_losses = skip_gram_model.train(skip_grams, epochs=EPOCH_NUM)
except Exception as ex:
    print(f"[ERR] error during train skip-gram: {ex}")
finally:
    with open('skip_gram_model.pkl', 'wb') as f:
        pickle.dump(skip_gram_model, f)
        
fig1, ax1 = plt.subplots(figsize=(8, 6))  # Размер графика: 8x6 дюймов
ax1.plot(np.arange(len(skip_gram_losses)), skip_gram_losses)
ax1.xlabel('Training step')
ax1.ylabel('Loss')
ax1.title('Loss for skip-gram')
plt.savefig('./images/skip_gram_model.png')
plt.close(fig1)

debug_print('Start to train cbow')
try:
    cbow_model = CBOW(len(vocab), EMBEDDING_DIM, vocab)
    cbow_losses = cbow_model.train(cbow_pairs, epochs=EPOCH_NUM)
except Exception as ex:
    print(f"[ERR] error during train skip-cbow: {ex}")
finally:
    with open('cbow_model.pkl', 'wb') as f:
        pickle.dump(cbow_model, f)
        
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(np.arange(len(cbow_losses)), cbow_losses)
ax2.xlabel('Training step')
ax2.ylabel('Loss')
ax2.title('Loss for cbow')
plt.savefig('./images/cbow_model.png')
plt.close(fig2)

# получаем список индексов
word_idxs = []
for word in test_vocab:
    word_idxs.append(vocab[word])

# получаем эмбеддинги для skip-gram и cbow
skip_gram_embeddings = []
for word in test_vocab:
    skip_gram_embeddings.append(skip_gram_model.embed(word))

cbow_embeddings = []
for word in test_vocab:
    cbow_embeddings.append(cbow_model.embed(word))
    
skip_gram_embeddings = np.array(skip_gram_embeddings)
cbow_embeddings = np.array(cbow_embeddings)

reduced_skip_gram_embeddings = reduce_to_k_dim(skip_gram_embeddings)
reduced_cbow_embeddings = reduce_to_k_dim(cbow_embeddings)

plot_embeddings(reduced_skip_gram_embeddings, test_vocab, save=True, model_name='skip_gram')

plot_embeddings(reduced_cbow_embeddings, test_vocab, save=True, model_name='cbow')