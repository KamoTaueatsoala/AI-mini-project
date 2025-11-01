import numpy as np
import cv2
from process import preprocess_data  # your preprocessing function
import os

# Load your preprocessed data
X_train, X_test, y_train, y_test, datagen = preprocess_data()

# Folder to save images
output_folder = "test_images/"
os.makedirs(output_folder, exist_ok=True)

# Save all test set images
for i, (img, label) in enumerate(zip(X_test, y_test)):
    # Convert back to 0-255 uint8
    img_uint8 = (img.squeeze() * 255).astype(np.uint8)
    # Determine label for filename
    label_str = "Normal" if label.argmax() == 0 else "Abnormal"
    filename = os.path.join(output_folder, f"test_image_{i+1}_{label_str}.png")
    cv2.imwrite(filename, img_uint8)
    print(f"Saved {filename}")

print(f"\n✅ All {len(X_test)} test images saved to '{output_folder}'")
