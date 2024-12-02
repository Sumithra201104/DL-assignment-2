# Install TensorFlow if needed
!pip install tensorflow matplotlib --quiet

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, GRU
import numpy as np
import matplotlib.pyplot as plt

# Simulated data
def generate_data(seq_len, n_samples=1000):
    X = np.random.randn(n_samples, seq_len, 1)
    y = (np.mean(X, axis=1) > 0).astype(int)  # Binary classification
    return X, y

seq_len = 50
X, y = generate_data(seq_len)

# Model definition
def build_rnn_model(rnn_type='SimpleRNN'):
    model = Sequential()
    if rnn_type == 'SimpleRNN':
        model.add(SimpleRNN(10, activation='tanh', input_shape=(seq_len, 1)))
    elif rnn_type == 'LSTM':
        model.add(LSTM(10, activation='tanh', input_shape=(seq_len, 1)))
    elif rnn_type == 'GRU':
        model.add(GRU(10, activation='tanh', input_shape=(seq_len, 1)))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and visualize gradient flow
def train_and_plot_gradients(model, rnn_type):
    history = model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    # Visualize gradient flow
    @tf.function
    def get_gradients():
        with tf.GradientTape() as tape:
            y_pred = model(X[:1], training=True)
            loss = tf.keras.losses.binary_crossentropy(y[:1], y_pred)
        return tape.gradient(loss, model.trainable_weights)

    gradients = get_gradients()
    gradients_norms = [tf.norm(g).numpy() for g in gradients if g is not None]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(gradients_norms)), gradients_norms, color='blue', alpha=0.7)
    plt.xlabel('Layer Index')
    plt.ylabel('Gradient Norm')
    plt.title(f'Gradient Norms for {rnn_type}')
    plt.show()

# Test with SimpleRNN
model_rnn = build_rnn_model('SimpleRNN')
train_and_plot_gradients(model_rnn, 'SimpleRNN')

# Test with LSTM
model_lstm = build_rnn_model('LSTM')
train_and_plot_gradients(model_lstm, 'LSTM')


# Test with GRU
model_gru = build_rnn_model('GRU')
train_and_plot_gradients(model_gru, 'GRU')
 
