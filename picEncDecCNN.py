import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical

# 1. Bilder laden und vorverarbeiten
def load_images(image_folder, label):
    images = []
    labels = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.jpg'):
            image = load_img(os.path.join(image_folder, file_name), target_size=(256, 256), color_mode='grayscale')
            image = img_to_array(image)
            image = np.repeat(image, 3, axis=-1)  # Convert grayscale to 3-channel image
            image = preprocess_input(image)  # VGG16 Preprocessing
            images.append(image)
            labels.append(label)
    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    return images, labels

# Pfade zu den Ordnern
keytool_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\bilderMitKeytool'
nokeytool_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\bilderOhneKeytool'
test_image_folder = 'C:\\Users\\dokov\\Documents\\Uni\\HiWi\\Keytool\\val'  # Testdaten

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

# 3. Hinzufügen von eigenen Schichten für die Klassifikation
model = Sequential([
    vgg16_base,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(2, activation='softmax')  # 2 Klassen (Keytool vorhanden oder nicht)
])

# 4. Modell kompilieren
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Lernraten-Scheduler und EarlyStopping hinzufügen
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# 6. Modell trainieren
history = model.fit(
    x_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

# 7. Genauigkeit und Verlust anzeigen
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
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

# 9. Testdatensatz laden und Vorhersagen treffen
x_test, test_labels = load_images(test_image_folder, None)  # Labels für den Test sind hier nicht relevant
predictions = model.predict(x_test)

# 10. Jedes Bild anzeigen und die Modellbeurteilung ausgeben
for i in range(len(x_test)):
    plt.imshow(x_test[i].reshape(256, 256, 3)[:,:,0], cmap='gray')  # Zeigt das Bild in Graustufen
    plt.axis('off')
    prediction = np.argmax(predictions[i])
    label = 'Keytool' if prediction == 1 else 'Kein Keytool'
    plt.title(f'Vorhersage: {label}')
    plt.show()
    print(f'Bild {i+1} - Vorhersage: {label}')
