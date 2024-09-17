import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

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

# 2. Datenaugmentation einrichten (leicht angepasst)
datagen = ImageDataGenerator(
    rotation_range=10,  # Leicht reduzierte Rotation
    width_shift_range=0.1,  # Leicht reduzierte Verschiebung
    height_shift_range=0.1,  # Leicht reduzierte Verschiebung
    shear_range=0.1,  # Leicht reduzierte Verzerrung
    zoom_range=0.1,  # Leicht reduzierter Zoom
    horizontal_flip=True,  # Horizontales Spiegeln bleibt gleich
    fill_mode='nearest'
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
    
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy')  # Lernrate weiter reduziert
    return model

# Autoencoder-Modell erstellen
autoencoder = create_autoencoder()

# Lernraten-Scheduler hinzufügen
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# 4. Modell trainieren mit augmentierten Daten und Loss-Kurve darstellen
history = autoencoder.fit(
    datagen.flow(x_train, x_train, batch_size=64),  # Erhöhte Batchgröße auf 64
    steps_per_epoch=len(x_train) // 64,
    epochs=20,
    validation_data=(x_val, x_val),  # Validierungsdaten werden nicht augmentiert
    callbacks=[lr_scheduler],  # Lernraten-Scheduler hinzufügen
    verbose=1
)

plt.plot(history.history['loss'], label='Train Loss (with Augmentation)')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Optimized Loss Curve with Adjusted Data Augmentation and Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 5. Modellbewertung
test_loss = autoencoder.evaluate(x_val, x_val)  # Testen mit Validierungsdaten
print(f'Test Loss: {test_loss}')
