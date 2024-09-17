import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

# 1. Bilder laden und vorverarbeiten
def load_images(image_folder, label):
    images = []
    labels = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.jpg'):  # Auf '.jpg' prüfen
            image = load_img(os.path.join(image_folder, file_name), target_size=(256, 256), color_mode='grayscale')
            image = img_to_array(image)
            images.append(image)
            labels.append(label)
    images = np.array(images, dtype='float32') / 255.0  # Normalisierung
    labels = np.array(labels)
    return images, labels

# Pfade zu den Ordnern
keytool_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\bilderMitKeytool'
nokeytool_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\bilderOhneKeytool'

# Bilder laden und Labels erstellen
x_keytool, y_keytool = load_images(keytool_folder, 1)
x_nokeytool, y_nokeytool = load_images(nokeytool_folder, 0)

# Zusammenführen von Bildern und Labels
x_data = np.concatenate([x_keytool, x_nokeytool], axis=0)
y_data = np.concatenate([y_keytool, y_nokeytool], axis=0)

# Labels in one-hot encodieren
y_data = to_categorical(y_data, num_classes=2)

# Aufteilen der Daten in Trainings- und Validierungsdaten
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# 2. Datenaugmentation einrichten (leicht angepasst)
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 3. Klassifikationsmodell definieren
def create_classifier(input_shape=(256, 256, 1)):
    model = Sequential()
    
    # Encoder
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    
    # Flattening before Dense Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    
    # Klassifikations-Output
    model.add(Dense(2, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Klassifikationsmodell erstellen
classifier = create_classifier()

# Lernraten-Scheduler und EarlyStopping hinzufügen
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# 4. Modell trainieren mit augmentierten Daten und Accuracy-Kurve darstellen
history = classifier.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    steps_per_epoch=len(x_train) // 64,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

# Genauigkeit anzeigen
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 5. Modellbewertung
test_loss, test_acc = classifier.evaluate(x_val, y_val)
print(f'Test Accuracy: {test_acc}')
