from .word2vec import Word2Vec, softmax
import numpy as np 

class CBOW(Word2Vec):
    
    # Прямой проход через сеть
    def forward(self, context_vectors): #? на вход подается контекстная матрица one-hot векторов
        hidden_layer = np.mean(np.dot(context_vectors, self.W_in), axis=0) # матрица умножается на матрицу и итог усредняется по строкам
        output_layer = np.dot(hidden_layer, self.W_out) # таким образом на выходе мы получаем вектор
        prediction = softmax(output_layer)
        return prediction, hidden_layer
    
    # Обратное распространение ошибок
    def backward(self, x, h, y_true, learning_rate=0.01):
        # Вычисляем ошибку на последнем слое
        delta = y_true - h
        
        # Обновляем выходной слой
        dW_out = np.outer(x, delta)
        self.W_out += learning_rate * dW_out
        
        # Рассчитываем градиент для скрытого слоя
        dh = np.dot(delta, self.W_out.T)
        
        # Обновляем входной слой
        dW_in = np.outer(h, dh)
        self.W_in += learning_rate * dW_in
    
    # Обучение модели
    def train(self, training_data, epochs=100, learning_rate=0.01):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for center_word, context_words in training_data:
                # One-hot encoding контекстных слов
                one_hot_contexts = np.zeros((len(context_words), self.vocabulary_size)) #? матрица из нескольких one-hot векторов?
                for i, w in enumerate(context_words):
                    one_hot_contexts[i][self.vocab[w]] = 1
                    
                # One-hot vector центрального слова
                one_hot_target = np.zeros(self.vocabulary_size)
                one_hot_target[self.vocab[center_word]] = 1
                
                # Прямой проход
                pred, hidden_state = self.forward(one_hot_contexts)
                
                # Кроссэнтропийная ошибка
                loss = -np.log(pred[np.argmax(one_hot_target)])
                total_loss += loss
                
                # Обратный проход и обновление весов
                self.backward(hidden_state, pred, one_hot_target, learning_rate)
            
            avg_loss = total_loss / len(training_data)
            
            losses.append(avg_loss)
        
        return losses