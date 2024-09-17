import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 1. Bilder laden und vorverarbeiten
def load_images(image_folder):
    images = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.jpg'):
            image = load_img(os.path.join(image_folder, file_name), target_size=(256, 256), color_mode='grayscale')
            image = img_to_array(image)
            images.append(image)
    images = np.array(images, dtype='float32') / 255.0  # Normalisierung
    return images

# Pfad zum Testdatensatz
test_image_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\bilderResized'  # Testdaten

# Testbilder laden
x_test = load_images(test_image_folder)

# 2. Modell laden
autoencoder = load_model('autoencoder_model.h5')

# 3. Testdatensatz durch das Modell laufen lassen und Verlustwerte berechnen
decoded_imgs = autoencoder.predict(x_test)
losses = np.mean(np.square(x_test - decoded_imgs), axis=(1, 2, 3))  # Mean Squared Error für jedes Bild

# 4. Verlustwerte runden und die Anzahl der Bilder pro Verlustwert zählen
rounded_losses = np.round(losses, 2)  # Verluste runden
unique_losses, counts = np.unique(rounded_losses, return_counts=True)  # Eindeutige Verluste und deren Häufigkeiten

# 5. Plotten der Anzahl der Bilder je Verlustwert
plt.figure(figsize=(10, 6))
plt.bar(unique_losses, counts, width=0.01, edgecolor='black')  # Balkendiagramm
plt.xlabel('Loss Value (MSE)')
plt.ylabel('Number of Images')
plt.title('Number of Images per Loss Value')
plt.grid(True)
plt.show()
