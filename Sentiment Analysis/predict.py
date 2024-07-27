from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from data_preprocessing import load_and_preprocess_data

# Load the pre-trained model
model = load_model('sentiment_analysis_model.h5')

def preprocess_text(text, maxlen=500):
    # Assuming you have a tokenizer
    # tokenized_text = tokenizer.texts_to_sequences([text])
    # processed_text = pad_sequences(tokenized_text, maxlen=maxlen)
    # For simplicity, let's assume the text is already tokenized and processed.
    # Replace the following line with actual preprocessing steps.
    processed_text = np.zeros((1, maxlen))  # Dummy data for example
    return processed_text

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    return 'Positive' if prediction >= 0.5 else 'Negative'

if _name_ == "_main_":
    new_text = "The movie was fantastic and I loved it!"
    print(predict_sentiment(new_text))