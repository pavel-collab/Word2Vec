import numpy as np
import torch

# Softmax функция для вывода вероятностей классов
def softmax(x):
    e_x = torch.exp(x - np.max(x))  # Устойчивость к переполнению экспоненты
    return e_x / e_x.sum(axis=-1, keepdims=True)

class Word2Vec:
    def __init__(self, vocabulary_size, embedding_dim, vocab):
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Весовые матрицы
        # матрица эмбеддингов, именно ее мы обучаем
        self.W_in = np.random.randn(self.vocabulary_size, self.embedding_dim) * 0.01
        self.W_out = np.random.randn(self.embedding_dim, self.vocabulary_size) * 0.01
        
    def forward(self, input_vector):
        pass
    
    def backward(self, x, h, y_true, learning_rate=0.01):
        pass
    
    def train(self, training_data, epochs=100, learning_rate=0.01):
        pass
    
    def embed(self, word):
        idx = self.vocab[word]
        return self.W_in[idx]