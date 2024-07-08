import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import matplotlib.pyplot as plt

# Festlegen der Parameter
input_dim = 100
sampling_rate = 30000
duration = 0.003  # 3 milliseconds

# Erzeugen eines Testsignals
time_steps = int(sampling_rate * duration)
test_signal = np.random.rand(time_steps, input_dim)

# Funktion zum Erstellen und Trainieren des Autoencoders
def train_autoencoder(neurons):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(input_dim,)),
        Dense(input_dim, activation='sigmoid')  # Ausgangsneuronen haben dieselbe Dimension wie Eingangsneuronen
    ])
    model.compile(optimizer='adam', loss='mse')
    
    start_time = time.time()
    model.fit(test_signal, test_signal, epochs=10, batch_size=32, verbose=0)
    end_time = time.time()
    
    return end_time - start_time

# Schleife über verschiedene Anzahlen von Neuronen
neuron_counts = list(range(10, 801, 10))
times = []

for neurons in neuron_counts:
    elapsed_time = train_autoencoder(neurons)
    times.append(elapsed_time)
    print(f"Neurons: {neurons}, Time: {elapsed_time:.4f} seconds")

# Plotten der Ergebnisse
plt.figure(figsize=(10, 6))
plt.plot(neuron_counts, times, marker='o')
plt.xlabel('Number of Neurons')
plt.ylabel('Time (seconds)')
plt.title('Training Time for Different Numbers of Neurons')
plt.grid(True)
plt.show()
