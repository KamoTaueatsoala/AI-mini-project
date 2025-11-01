import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# Create output directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()
    print(f"✅ Training curves saved to {output_dir}/training_curves.png")

def evaluate_and_visualize(model, X_test, y_test):
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    # --- Classification report ---
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal'])
    print(report)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print(f"✅ Classification report saved to {output_dir}/classification_report.txt")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Abnormal'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print(f"✅ Confusion matrix saved to {output_dir}/confusion_matrix.png")

    # --- Save sample predictions ---
    sample_count = min(5, len(X_test))
    for i in range(sample_count):
        plt.imshow(X_test[i].squeeze(), cmap='gray')
        plt.title(f"True: {'Normal' if y_true[i]==0 else 'Abnormal'}, "
                  f"Pred: {'Normal' if y_pred[i]==0 else 'Abnormal'}")
        plt.axis('off')
        file_path = os.path.join(output_dir, f"sample_{i+1}_prediction.png")
        plt.savefig(file_path)
        plt.close()
    print(f"✅ Saved {sample_count} sample predictions to {output_dir}/")

def visualize_results(model_path, X_test, y_test):
    model = load_model(model_path)
    evaluate_and_visualize(model, X_test, y_test)
