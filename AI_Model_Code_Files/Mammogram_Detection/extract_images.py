import numpy as np
import cv2
from process import preprocess_data  # your preprocessing function
import os

# Load your preprocessed data
X_train, X_test, y_train, y_test, datagen = preprocess_data()

# Pick first 15 images from the test set
num_images_to_save = 15
images_to_save = X_test[:num_images_to_save]
labels_to_save = y_test[:num_images_to_save]

# Folder to save images
output_folder = "test_images/"
os.makedirs(output_folder, exist_ok=True)

# Save the images
for i, img in enumerate(images_to_save):
    # Convert back to 0-255 uint8
    img_uint8 = (img.squeeze() * 255).astype(np.uint8)
    # Determine label for filename
    label_str = "Normal" if labels_to_save[i].argmax() == 0 else "Abnormal"
    filename = os.path.join(output_folder, f"test_image_{i+1}_{label_str}.png")
    cv2.imwrite(filename, img_uint8)
    print(f"Saved {filename}")
