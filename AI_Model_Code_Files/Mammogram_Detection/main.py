import numpy as np
from process import preprocess_data
from train import build_model, train_model
from visualise import plot_history, visualize_results
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # Processing
    X_train, X_test, y_train, y_test, datagen = preprocess_data()

    # ----- Training -----
    model = build_model()

    # Unpack all returned values from train_model (model, history, adaptive_cb)
    model, history = train_model(model, X_train, X_test, y_train, y_test)

    # Save the trained model
    model.save('mammogram_detector.h5')

    # Save training history
    np.save('training_history.npy', history.history)

    # Evaluation
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    test_accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f"Test accuracy: {test_accuracy:.2f}")

    # Visualization
    plot_history(history.history)
    visualize_results('mammogram_detector.h5', X_test, y_test)