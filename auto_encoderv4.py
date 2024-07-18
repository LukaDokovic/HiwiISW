import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal
import time
import matplotlib.pyplot as plt

# Festlegen der Parameter
input_dim = 90
encoding_dim = 6  # Dimension der Engstelle
sampling_rate = 30000
duration = 0.003  # 3 milliseconds

# Erzeugen eines Testsignals
time_steps = int(sampling_rate * duration)
test_signal = np.random.rand(time_steps)

# Funktion zum Erstellen und Trainieren des Autoencoders
def train_autoencoder(neurons, layers):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.3, stddev=0.05)))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu', kernel_initializer=RandomNormal(mean=0.3, stddev=0.05)))
    model.add(Dense(encoding_dim, activation='tanh', kernel_initializer=RandomNormal(mean=0.3, stddev=0.05)))  # Engstelle

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    start_time = time.time()
    for _ in range(100):
        model.predict(test_signal.reshape(1, input_dim))
    end_time = time.time()
    
    return (end_time - start_time) / 100

# Schleife über verschiedene Anzahlen von Neuronen und mehrere Läufe
neuron_counts = list(range(10, 5001, 1000))
layer_counts = [1, 3, 5]  # Verschiedene Anzahl von Schichten
results = {}

for layers in layer_counts:
    times = []
    for neurons in neuron_counts:
        run_times = []
        for _ in range(5):
            elapsed_time = train_autoencoder(neurons, layers)
            run_times.append(elapsed_time)
        avg_time = np.mean(run_times)
        times.append(avg_time)
        print(f"Layers: {layers}, Neurons: {neurons}, Average Time: {avg_time:.6f} seconds")
    results[layers] = times

# Plotten der Ergebnisse
plt.figure(figsize=(10, 6))
for layers, times in results.items():
    plt.plot(neuron_counts, times, marker='o', label=f'{layers} Layer(s)')
plt.xlabel('Number of Neurons')
plt.ylabel('Average Inference Time (seconds)')
plt.title('Average Inference Time for Different Numbers of Neurons and Layers')
plt.legend()
plt.grid(True)
plt.show()
