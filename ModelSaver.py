import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

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

# Pfade zu den Ordnern
train_image_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\bilder256'  # Trainingsdaten
val_image_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\val'  # Validierungsdaten

# Trainings- und Validierungsbilder laden
x_train = load_images(train_image_folder)
x_val = load_images(val_image_folder)

# 2. Datenaugmentation einrichten (leicht angepasst)
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 3. Autoencoder-Modell definieren mit zusätzlichem Dense Layer und Bottleneck
def create_autoencoder(input_shape=(256, 256, 1)):
    model = Sequential()
    
    # Encoder
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Flattening before Dense Layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.1))

    # Middle bottleneck layer with 2 neurons
    model.add(Dense(2))
    model.add(LeakyReLU(alpha=0.1))

    # Expand to the previous number of neurons before reshaping
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(65536))
    model.add(LeakyReLU(alpha=0.1))

    # Reshape zurück zu Feature Maps für den Decoder
    model.add(Reshape((32, 32, 64)))

    # Decoder
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))
    
    model.compile(optimizer=Nadam(learning_rate=0.00005), loss='binary_crossentropy')
    return model

# Autoencoder-Modell erstellen
autoencoder = create_autoencoder()

# Lernraten-Scheduler und EarlyStopping hinzufügen
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# 4. Modell trainieren mit augmentierten Daten
history = autoencoder.fit(
    datagen.flow(x_train, x_train, batch_size=64),  # Augmentierte Trainingsdaten
    steps_per_epoch=len(x_train) // 64,
    epochs=100,
    validation_data=datagen.flow(x_val, x_val, batch_size=64),  # Augmentierte Validierungsdaten
    validation_steps=len(x_val) // 64,
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

# 5. Modell speichern
autoencoder.save('autoencoder_model.h5')
