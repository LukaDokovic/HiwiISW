import cv2
import numpy as np

image_path = '/home/lukadokovic/workspace/HiwiISW/2_10-reo.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not loaded properly")

_, blobs = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(blobs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

edge_distance = 100  # Mindestabstand vom Rand

# Funktion zum Beschneiden des Bildes basierend auf dem Randabstand
def crop_image(img, edge_dist):
    height, width = img.shape
    return img[edge_dist:height-edge_dist, edge_dist:width-edge_dist]

# Beschneide das Bild entlang des definierten Rands
cropped_image = crop_image(image, edge_distance)

# Speichere das beschnittene Bild (optional)
cv2.imwrite('/home/lukadokovic/workspace/HiwiISW/2_10-reo_cropped.png', cropped_image)

# Zeige das Original- und das beschnittene Bild an (optional)
cv2.imshow("Original Image", image)
cv2.imshow("Cropped Image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

