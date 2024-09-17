import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, UpSampling2D, Reshape

# 1. Bilder laden und vorverarbeiten
def load_images(image_folder):
    images = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.jpg'):  # Auf '.jpg' prüfen
            image = load_img(os.path.join(image_folder, file_name), target_size=(256, 256), color_mode='grayscale')
            image = img_to_array(image)
            images.append(image)
    images = np.array(images, dtype='float32') / 255.0  # Normalisierung
    return images

# Pfade zu den Ordnern
train_image_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\bilder256'  # Trainingsdaten
val_image_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\val'  # Validierungsdaten

# Trainings- und Validierungsbilder laden
x_train = load_images(train_image_folder)
x_val = load_images(val_image_folder)

# 2. Datenaugmentation einrichten
datagen = ImageDataGenerator(
    rotation_range=20,  # Dreht das Bild zufällig um bis zu 20 Grad
    width_shift_range=0.2,  # Verschiebt das Bild horizontal um bis zu 20% der Breite
    height_shift_range=0.2,  # Verschiebt das Bild vertikal um bis zu 20% der Höhe
    shear_range=0.2,  # Verzieht das Bild um bis zu 20%
    zoom_range=0.2,  # Zoomt zufällig bis zu 20% rein oder raus
    horizontal_flip=True,  # Spiegelt das Bild horizontal
    fill_mode='nearest'  # Auffüllen der durch die Transformationen entstandenen leeren Bereiche
)

# 3. Encoder-Decoder-Modell definieren
def create_autoencoder(input_shape=(256, 256, 1)):
    model = Sequential()
    
    # Encoder
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    
    # Decoder
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Autoencoder-Modell erstellen
autoencoder = create_autoencoder()

# 4. Modell trainieren mit augmentierten Daten und Loss-Kurve darstellen
history = autoencoder.fit(
    datagen.flow(x_train, x_train, batch_size=32),  # Verwendung von augmentierten Trainingsdaten
    steps_per_epoch=len(x_train) // 32,
    epochs=20,
    validation_data=(x_val, x_val),  # Validierungsdaten werden nicht augmentiert
    verbose=1
)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve with Data Augmentation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 5. Modellbewertung
test_loss = autoencoder.evaluate(x_val, x_val)  # Testen mit Validierungsdaten
print(f'Test Loss: {test_loss}')