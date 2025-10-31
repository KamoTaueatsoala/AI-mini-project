import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, Label, Frame, Canvas, Scrollbar
from PIL import Image, ImageTk
import os

# Load trained model
model = load_model('mammogram_detector.h5')
classes = ['Normal', 'Abnormal']

# Function to select folder and predict images
def select_folder_and_predict():
    folder_path = filedialog.askdirectory()
    if not folder_path:
        return

    # Get all image files in folder
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) == 0:
        result_label.config(text="No images found in folder.")
        return

    # Clear previous images
    for widget in frame_inside_canvas.winfo_children():
        widget.destroy()

    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set max image size to fit the screen (divide width by number of columns)
    num_columns = min(5, len(image_files))
    max_img_width = screen_width // num_columns - 40
    max_img_height = 300  # reasonable height for each image

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized.astype('float32') / 255.0
        img_input = np.expand_dims(img_normalized, axis=(0, -1))

        # Predict
        pred = model.predict(img_input)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_prob = pred[0][pred_class]

        # Convert image for Tkinter
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Resize for display
        img_pil.thumbnail((max_img_width, max_img_height))
        img_tk = ImageTk.PhotoImage(img_pil)

        # Create frame for image + text
        frame = Frame(frame_inside_canvas, bd=2, relief=tk.RIDGE)
        frame.grid(row=i // num_columns, column=i % num_columns, padx=10, pady=10)

        # Image label
        label_img = Label(frame, image=img_tk)
        label_img.image = img_tk  # Keep reference
        label_img.pack()

        # Filename label
        label_name = Label(frame, text=img_file, font=("Helvetica", 10))
        label_name.pack()

        # Prediction label
        label_pred = Label(frame, text=f"{classes[pred_class]} ({pred_prob*100:.2f}%)",
                           font=("Helvetica", 12, "bold"))
        label_pred.pack()

# Tkinter window
root = tk.Tk()
root.title("Mammogram Classifier")

# Make fullscreen but allow closing
root.state('zoomed')

btn = tk.Button(root, text="Select Folder", command=select_folder_and_predict)
btn.pack(pady=10)

# Canvas with scrollbars
canvas = Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scrollbar.set)

frame_inside_canvas = Frame(canvas)
canvas.create_window((0, 0), window=frame_inside_canvas, anchor='nw')

def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox('all'))

frame_inside_canvas.bind("<Configure>", on_configure)

result_label = Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=10)

root.mainloop()
