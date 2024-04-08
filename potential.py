import cv2
import numpy as np



image_path = '/home/lukadokovic/workspace/HiwiISW/2_10-reo.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


if image is None:
    raise ValueError("Image not loaded properly")


_, blobs = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


contours, _ = cv2.findContours(blobs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def sample(x, y):
    potential = 0
    for cnt in contours:
        closest_point = cv2.pointPolygonTest(cnt, (x, y), True)
        potential += 1.0 / (1 + closest_point ** 2) if closest_point > 0 else 0
    return potential


def potField(x_resolution, y_resolution):

    potential_field = np.zeros((y_resolution, x_resolution))

    x_scale = image.shape[1] / x_resolution
    y_scale = image.shape[0] / y_resolution

    for i in range(y_resolution):
        for j in range(x_resolution):

            x = int(j * x_scale)
            y = int(i * y_scale)
            potential_field[i, j] = sample(x, y)
    return potential_field


print(f"Sample potential at (100,100): {sample(100, 100)}")

# Generate a small potential field (10x10) for demonstration purposes
pot_field_demo = potField(10, 10)
print(f"Potential field (10x10) for demonstration:\n{pot_field_demo}")

