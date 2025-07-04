from word2vec import Word2Vec, softmax
import numpy as np 

class SkipGram(Word2Vec):
    # Прямой проход
    # на вход подается one-hot вектор центрального слова
    def forward(self, input_vector):
        # получаем промежуточное представление (по-сути получаем эмбеддинг слова)
        hidden_layer = np.dot(input_vector, self.W_in)
        # получаем распределение вероятностей на слова контекста
        output_layer = np.dot(hidden_layer, self.W_out)
        prediction = softmax(output_layer)
        return prediction, hidden_layer
    
    # Обратное распространение
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
    
    # Тренировка модели
    def train(self, training_data, epochs=100, learning_rate=0.01):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for center_word, context_word in training_data:
                # One-hot encoding центровых слов
                one_hot_input = np.zeros(self.vocabulary_size)
                one_hot_input[self.vocab[center_word]] = 1
                
                # One-hot вектор контекста
                # Здесь мы получаем вектор с несколькими единицами, так как мы складываем несколкьо векторов контекста
                one_hot_output = np.zeros(self.vocabulary_size)
                one_hot_output[self.vocab[context_word]] = 1
                
                # Прямой проход
                pred, hidden_state = self.forward(one_hot_input)
                
                # Вычисляем потерю cross-entropy
                loss = -np.log(pred[np.argmax(one_hot_output)])
                total_loss += loss
                
                # Обратный проход
                self.backward(hidden_state, pred, one_hot_output, learning_rate)
            
            avg_loss = total_loss / len(training_data)
            
            losses.append(avg_loss)
        
        return losses