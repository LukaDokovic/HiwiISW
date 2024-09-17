import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, target_size=(256, 256)):
    # Erstelle den Ausgabefolder, wenn er nicht existiert
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iteriere durch alle Dateien im Eingabefolder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):  # Dateitypen filtern
            img_path = os.path.join(input_folder, file_name)
            with Image.open(img_path) as img:
                # Bild skalieren
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                # Pfad zum Speichern des Bildes im Ausgabefolder
                output_path = os.path.join(output_folder, file_name)
                # Bild speichern
                img_resized.save(output_path)
                print(f'{file_name} resized and saved to {output_path}')

# Beispiel zur Anwendung:
input_folder = r'C:\Users\dokov\Documents\Uni\HiWi\Keytool\bilderRaw'
output_folder = r'C:\Users\dokov\Documents\Uni\HiWi\Keytool\bilderResized'

resize_images_in_folder(input_folder, output_folder)
