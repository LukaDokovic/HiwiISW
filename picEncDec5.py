import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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
    rotation_range=10,  # Leicht reduzierte Rotation
    width_shift_range=0.1,  # Leicht reduzierte Verschiebung
    height_shift_range=0.1,  # Leicht reduzierte Verschiebung
    shear_range=0.1,  # Leicht reduzierte Verzerrung
    zoom_range=0.1,  # Leicht reduzierter Zoom
    horizontal_flip=True,  # Horizontales Spiegeln bleibt gleich
    fill_mode='nearest'
)

# 3. Modell-Erstellungs- und Trainingsfunktion für die Bayes'sche Optimierung
def build_and_train_model(params):
    model = Sequential()
    
    # Encoder
    model.add(Conv2D(int(params['filters_1']), kernel_size=(3, 3), activation='relu', padding='same', input_shape=(256, 256, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(int(params['filters_2']), kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(int(params['filters_3']), kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    
    # Decoder
    model.add(Conv2D(int(params['filters_3']), kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(int(params['filters_2']), kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(int(params['filters_1']), kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))
    
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='binary_crossentropy')
    
    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    # Training
    history = model.fit(
        datagen.flow(x_train, x_train, batch_size=int(params['batch_size'])),
        steps_per_epoch=len(x_train) // int(params['batch_size']),
        epochs=20,
        validation_data=(x_val, x_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Ziel: Minimieren des Validierungsverlusts
    val_loss = min(history.history['val_loss'])
    
    return {'loss': val_loss, 'status': STATUS_OK}

# 4. Hyperparameter-Raum definieren
space = {
    'filters_1': hp.choice('filters_1', [16, 32, 64]),
    'filters_2': hp.choice('filters_2', [32, 64, 128]),
    'filters_3': hp.choice('filters_3', [64, 128, 256]),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
    'batch_size': hp.choice('batch_size', [32, 64, 128])
}

# 5. Bayes'sche Optimierung ausführen
trials = Trials()
best = fmin(fn=build_and_train_model,
            space=space,
            algo=tpe.suggest,
            max_evals=20,  # Anzahl der Iterationen
            trials=trials)

print("Beste Hyperparameter: ", best)

# Optional: Trainingsverlauf des besten Modells visualisieren (aus den Trials)
# Hinweis: Der komplette Trainingsverlauf kann in den Trials gespeichert werden.
