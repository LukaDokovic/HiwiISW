import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

# 1. Bilder laden und vorverarbeiten
def load_images(image_folder):
    images = []
    image_names = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.jpg'):
            image = load_img(os.path.join(image_folder, file_name), target_size=(256, 256), color_mode='grayscale')
            image = img_to_array(image)
            image = np.repeat(image, 3, axis=-1)  # Graustufenbild zu einem 3-Kanal-Bild machen
            image = preprocess_input(image)  # VGG16 Preprocessing
            images.append(image)
            image_names.append(file_name)
    images = np.array(images, dtype='float32')
    return images, image_names

# Pfad zum Testdatensatz
test_image_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\test'  # Testdaten

# Testbilder laden
x_test, image_names = load_images(test_image_folder)

# Modell laden (falls es gespeichert ist)
# model = load_model('path_to_your_model.h5')

# Vorhersagen auf den Testbildern
predictions = model.predict(x_test)

# 2. Jedes Bild anzeigen und die Modellbeurteilung ausgeben
for i in range(len(x_test)):
    plt.imshow(x_test[i].reshape(256, 256, 3)[:,:,0], cmap='gray')  # Zeigt das Bild in Graustufen
    plt.axis('off')
    prediction = np.argmax(predictions[i])
    label = 'Keytool' if prediction == 1 else 'Kein Keytool'
    plt.title(f'Vorhersage: {label}')
    plt.show()
    print(f'Bild: {image_names[i]} - Vorhersage: {label}')
