import os
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pyads

# Pfade und ADS Einstellungen
image_folder = 'C:\\Pfad\\zum\\Ordner'
model_path = 'autoencoder_model.h5'
ams_net_id = '192.168.0.1.1.1'
ams_port = 851

# Modell laden
autoencoder = load_model(model_path)

# Verbindung zu TwinCAT ADS herstellen
plc = pyads.Connection(ams_net_id, ams_port)

def get_latest_image(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    if not files:
        return None
    latest_file = max([os.path.join(folder, f) for f in files], key=os.path.getctime)
    return latest_file

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(256, 256), color_mode='grayscale')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0
    return image

def calculate_loss(autoencoder, image):
    reconstructed = autoencoder.predict(image)
    loss = np.mean(np.square(image - reconstructed))
    return loss

def send_loss_to_sps(loss_value):
    plc.open()
    plc.write_by_name('MAIN.loss_value', loss_value, pyads.PLCTYPE_REAL)
    plc.close()

def main():
    last_processed_image = None
    while True:
        latest_image = get_latest_image(image_folder)
        if latest_image and latest_image != last_processed_image:
            print(f'Neues Bild gefunden: {latest_image}')
            image = preprocess_image(latest_image)
            loss = calculate_loss(autoencoder, image)
            print(f'Berechneter Loss: {loss}')
            send_loss_to_sps(loss)
            last_processed_image = latest_image
        time.sleep(5)

if __name__ == "__main__":
    main()
