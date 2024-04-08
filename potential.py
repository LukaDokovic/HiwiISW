import cv2
import numpy as np


image_path = '/home/lukadokovic/workspace/HiwiISW/2_10-reo.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not loaded properly")


_, blobs = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


contours, _ = cv2.findContours(blobs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


min_radius = 2  
edge_distance = 5  

def is_near_edge(cnt, img_shape, edge_dist):
    x, y, w, h = cv2.boundingRect(cnt)
    if x <= edge_dist or y <= edge_dist:
        return True
    if x + w >= img_shape[1] - edge_dist or y + h >= img_shape[0] - edge_dist:
        return True
    return False


filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= np.pi * min_radius**2 and not is_near_edge(cnt, image.shape, edge_distance)]


def sample(x, y, influence_radius=200):
    potential = 0
    for cnt in filtered_contours:
        dist = cv2.pointPolygonTest(cnt, (x, y), True)
        if dist < influence_radius and dist >= 0:
            potential += 1.0 / (1 + dist**2)
    return potential


def potField(x_resolution, y_resolution, influence_radius=200):
    potential_field = np.zeros((y_resolution, x_resolution))

    x_scale = image.shape[1] / x_resolution
    y_scale = image.shape[0] / y_resolution

    for i in range(y_resolution):
        for j in range(x_resolution):
            x = int(j * x_scale)
            y = int(i * y_scale)
            potential_field[i, j] = sample(x, y, influence_radius)

    return potential_field


sample_potential = sample(100, 100)
print(f"Sample potential at (100,100): {sample_potential}")


pot_field_demo = potField(10, 10)
print(f"Potential field (10x10) for demonstration:\n{pot_field_demo}")
