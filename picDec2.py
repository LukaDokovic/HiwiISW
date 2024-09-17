import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# 1. Bilder laden und vorverarbeiten
def load_images(image_folder, label):
    images = []
    labels = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.jpg'):
            image = load_img(os.path.join(image_folder, file_name), target_size=(256, 256))
            image = img_to_array(image)
            image = preprocess_input(image)  # VGG16 Preprocessing
            images.append(image)
            labels.append(label)
    images = np.array(images)
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
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

# 2. VGG16 Modell als Feature-Extractor verwenden
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# 3. Hinzufügen von eigenen Schichten
model = Sequential([
    vgg16_base,
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# 4. Modellkompilierung
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Lernraten-Scheduler und EarlyStopping hinzufügen
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 6. Modell trainieren mit augmentierten Daten
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=16),
    steps_per_epoch=len(x_train) // 16,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

# 7. Genauigkeit anzeigen
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 8. Modellbewertung
test_loss, test_acc = model.evaluate(x_val, y_val)
print(f'Test Accuracy: {test_acc}')
