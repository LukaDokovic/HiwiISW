import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random

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

# 4. Stichproben von Bildern plotten: Erste 10 und 10 zufällig ausgewählte Bilder
sample_indices = list(range(10)) + random.sample(range(10, len(x_test)), 10)
n_images = len(sample_indices)
plt.figure(figsize=(15, 2 * n_images))
for i, idx in enumerate(sample_indices):
    ax = plt.subplot(n_images, 2, 2 * i + 1)
    plt.imshow(x_test[idx].reshape(256, 256), cmap='gray')
    plt.title(f"Original Image {idx+1}")
    plt.axis("off")
    
    ax = plt.subplot(n_images, 2, 2 * i + 2)
    plt.imshow(decoded_imgs[idx].reshape(256, 256), cmap='gray')
    plt.title(f"Reconstructed Image {idx+1}\nLoss: {losses[idx]:.4f}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# 5. Scatter-Plot der Loss-Werte für alle Bilder
plt.figure(figsize=(10, 6))
plt.scatter(range(len(losses)), losses, c='blue', alpha=0.5)
plt.xlabel('Image Index')
plt.ylabel('Loss Value (MSE)')
plt.title('Loss Value for Each Image')
plt.grid(True)
plt.show()
