from data_preprocessing import load_and_preprocess_data
from model import build_model

def train_model():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = build_model()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    batch_size = 32
    epochs = 3
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    
    model.save('sentiment_analysis_model.h5')
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

if _name_ == "_main_":
    train_model()