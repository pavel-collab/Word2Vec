from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

def data_import(page_limit=1000000):
    dataset = load_dataset("wiki_snippets", "wiki40b_en_100_0")["train"]
    text_corpus = []
    count = 0
    for wiki in dataset:
        text_corpus.append(wiki['passage_text'])
        count += 1
        if count == page_limit:
            break
    return text_corpus # возвращает список текстов

'''
снижение размерности многомерных векторов для отображения в двумерном пространстве
На вход подается матрица эмбедингов (n_samples, dim)
На выходе молучаем матрицу (n_samples, k)
'''
def reduce_to_k_dim(M, k=2):
    n_iters = 10 
    trunc_svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
    M_reduced = trunc_svd.fit_transform(M)
    return M_reduced

'''
Отрисовка векторов эмбеддингов и соответствующих слов в двумерном пространстве
На вход подается матрица эмбеддингов (n_samples, dim) и список слов,
причем порядок слов соответсвует порядку эмбеддингов в матрице
'''
def plot_embeddings(embedding_matrix, words):
    assert embedding_matrix.shape[1] == 2

    plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')

    x_values = embedding_matrix[:,0]
    y_values = embedding_matrix[:,1]
    for i, word in enumerate(words):
        x = x_values[i]
        y = y_values[i]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, word, fontsize=9)
    plt.show()
    
# Генерируем тренировочный корпус слов
def preprocess_text(text_corpus: list):
    preprocess_text_corpus = []
    for text in text_corpus:
        text = text.lower().split()
        preprocess_text_corpus.append(text)
    return preprocess_text_corpus


# Создаем словарь индексов для каждого уникального слова
def build_vocabulary(corpus: list):
    vocab = {}
    index_to_word = []
    
    for text in corpus:
        for token in text:
            if token not in vocab:
                vocab[token] = len(vocab)
                index_to_word.append(token)
                
    return vocab, index_to_word


# Генерация контекста (skip-gram pairs)
def generate_skip_grams(corpus, window_size=2):
    skip_grams = []
    for text in corpus:
        for i in range(len(text)):
            target_word = text[i]
            
            # Окружающий контекст вокруг целевого слова
            context_words = []
            start_idx = max(i-window_size, 0)
            end_idx = min(i+window_size+1, len(text))
            
            for j in range(start_idx, end_idx):
                if j != i:
                    context_words.append((target_word, text[j]))
                    
            skip_grams.extend(context_words)
            
    return skip_grams

# Генерация CBOW-пар
def generate_cbow_pairs(corpus, window_size=2):
    cbow_pairs = []
    
    for text in corpus:
        for i in range(window_size, len(text)-window_size):
            context_words = []
            start_idx = max(i-window_size, 0)
            end_idx = min(i+window_size+1, len(text))
            
            for j in range(start_idx, end_idx):
                if j != i:
                    context_words.append(text[j])
            
            cbow_pairs.append((text[i], context_words))
        
    return cbow_pairs
