from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def build_model(max_features=10000, maxlen=500):
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=128, input_length=maxlen))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model