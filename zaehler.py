import cv2
import numpy as np

image_path = '/home/lukadokovic/workspace/HiwiISW/2_10-reo.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not loaded properly")

_, blobs = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(blobs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

min_radius = 20  # Minimaler Radius, um Blobs zu zählen
edge_distance = 5  # Mindestabstand vom Rand

def is_near_edge(cnt, img_shape, edge_dist):
    x, y, w, h = cv2.boundingRect(cnt)
    return x <= edge_dist or y <= edge_dist or x + w >= img_shape[1] - edge_dist or y + h >= img_shape[0] - edge_dist

# Filtere die Konturen, die groß genug sind und nicht zu nahe am Rand liegen
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= np.pi * min_radius**2 and not is_near_edge(cnt, image.shape, edge_distance)]

# Zähle und gebe die Blobs aus, die den Mindestradius überschreiten
blob_info = []
for cnt in filtered_contours:
    area = cv2.contourArea(cnt)
    radius = np.sqrt(area / np.pi)
    x, y, w, h = cv2.boundingRect(cnt)
    blob_info.append((x + w // 2, y + h // 2, radius))  # Mittelpunkt und Radius des Blobs

print(f"Anzahl der Blobs mit einem Radius ≥ {min_radius}: {len(blob_info)}")
for index, (x, y, radius) in enumerate(blob_info):
    print(f"Blob {index + 1}: Mittelpunkt bei ({x}, {y}), Radius = {radius:.2f}")

