import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
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
np.random.rand()
# Funktion zum Erstellen und Trainieren des Autoencoders
def train_autoencoder(neurons):
    model = Sequential([
        #Input(shape=(test_signal)),
        Dense(neurons, activation='relu',input_shape=(90,), kernel_initializer=RandomNormal(mean=0.3, stddev=0.05)),
        Dense(encoding_dim, activation='tanh', kernel_initializer=RandomNormal(mean=0.3, stddev=0.05)),  # Engstelle
        #Dense(neurons, activation='relu'),
        #Dense(input_dim, activation='sigmoid')  # Ausgangsneuronen haben dieselbe Dimension wie Eingangsneuronen
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    start_time = time.time()
    model.predict(test_signal.reshape(1,90))
    #model.fit(test_signal, test_signal, epochs=10, batch_size=32, verbose=0)
    end_time = time.time()
    
    return end_time - start_time

# Schleife über verschiedene Anzahlen von Neuronen und mehrere Läufe
neuron_counts = list(range(10, 5000011, 1000000))
times = []
runs = 5

for neurons in neuron_counts:
    run_times = []
    for _ in range(runs):
        elapsed_time = train_autoencoder(neurons)
        run_times.append(elapsed_time)
    avg_time = np.mean(run_times)
    times.append(avg_time)
    print(f"Neurons: {neurons}, Average Time: {avg_time:.4f} seconds")

# Plotten der Ergebnisse
plt.figure(figsize=(10, 6))
plt.plot(neuron_counts, times, marker='o')
plt.xlabel('Number of Neurons')
plt.ylabel('Time (seconds)')
plt.title('Average Training Time for Different Numbers of Neurons')
plt.grid(True)
plt.show()