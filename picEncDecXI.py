import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# 1. Bilder laden und vorverarbeiten
def load_images(image_folder):
    images = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.jpg'):
            image = load_img(os.path.join(image_folder, file_name), target_size=(256, 256))
            image = img_to_array(image)
            image = preprocess_input(image)  # VGG16 Preprocessing
            images.append(image)
    images = np.array(images, dtype='float32')
    return images

# Pfade zu den Ordnern
train_image_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\bilder256'  # Trainingsdaten
val_image_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\val'  # Validierungsdaten

# Trainings- und Validierungsbilder laden
x_train = load_images(train_image_folder)
x_val = load_images(val_image_folder)

# 2. VGG16 Modell als Feature-Extractor verwenden (ohne Top-Schichten)
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# 3. Autoencoder-Modell definieren mit VGG16-Features als Input
def create_autoencoder(input_shape=(8, 8, 512)):  # Anpassung an die VGG16-Ausgabeform
    model = Sequential()
    
    # Encoder (die bereits extrahierten Features werden als Input verwendet)
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))

    # Bottleneck Layer mit 16 Neuronen
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.1))

    # Decoder
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(np.prod(input_shape)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Reshape(input_shape))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same'))
    
    model.compile(optimizer=Nadam(learning_rate=0.00005), loss='binary_crossentropy')
    return model

# 4. Autoencoder-Modell erstellen
autoencoder = create_autoencoder()

# 5. Lernraten-Scheduler und EarlyStopping hinzufügen
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

