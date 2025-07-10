from .word2vec import Word2Vec, softmax
from tqdm import tqdm
import torch

class CBOW(Word2Vec):
    
    # Прямой проход через сеть 
    def forward(self, context_vectors): #? на вход подается контекстная матрица one-hot векторов
        hidden_layer = torch.mean(torch.matmul(context_vectors, self.W_in), axis=0) # матрица умножается на матрицу и итог усредняется по строкам
        output_layer = torch.matmul(hidden_layer, self.W_out) # таким образом на выходе мы получаем вектор
        
        prediction = softmax(output_layer)
        return prediction, hidden_layer
    
    # Обратное распространение ошибок
    def backward(self, x, h, y_true, learning_rate=0.01):
        # Вычисляем ошибку на последнем слое
        delta = y_true - h
        
        # Обновляем выходной слой
        dW_out = torch.outer(x, delta)
        self.W_out += learning_rate * dW_out
        
        # Рассчитываем градиент для скрытого слоя
        dh = torch.matmul(delta, self.W_out.T)
        
        # Обновляем входной слой
        dW_in = torch.outer(h, dh)
        self.W_in += learning_rate * dW_in
    
    # Обучение модели
    def train(self, training_data, epochs=100, learning_rate=0.01):
        self.W_in.to(self.device)
        self.W_out.to(self.device)
        losses = []
        for epoch in tqdm(range(epochs)):
            total_loss = 0
            for center_word, context_words in training_data:
                # One-hot encoding контекстных слов
                one_hot_contexts = torch.zeros((len(context_words), self.vocabulary_size)) #? матрица из нескольких one-hot векторов?
                for i, w in enumerate(context_words):
                    one_hot_contexts[i][self.vocab[w]] = 1
                    
                # One-hot vector центрального слова
                one_hot_target = torch.zeros(self.vocabulary_size)
                one_hot_target[self.vocab[center_word]] = 1
                
                one_hot_target.to(self.device)
                one_hot_contexts.to(self.device)
                
                # Прямой проход
                pred, hidden_state = self.forward(one_hot_contexts)
                
                # Кроссэнтропийная ошибка
                loss = -torch.log(pred[torch.argmax(one_hot_target)])
                
                loss.to('cpu')
                total_loss += loss
                
                # Обратный проход и обновление весов
                self.backward(hidden_state, pred, one_hot_target, learning_rate)
            
            avg_loss = total_loss / len(training_data)
            
            losses.append(avg_loss)
        
        self.W_in.to('cpu')
        self.W_out.to('cpu')
        return losses