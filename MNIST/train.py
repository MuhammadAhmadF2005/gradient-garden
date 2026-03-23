import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def main():
    print("Fetching MNIST dataset (this may take a moment)...")
    mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
    
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    
    print("Training SGDClassifier with StandardScaler on the full dataset...")
    # StandardScaler transforms values to mean 0, unit variance giving SGD a massive accuracy boost over plain 0-255 pixels
    model = make_pipeline(StandardScaler(), SGDClassifier(random_state=42))
    model.fit(X_train, y_train)
    
    model_path = "mnist_model.pkl"
    print(f"Saving the improved model to {model_path}...")
    joblib.dump(model, model_path)
    
    print("Training complete and improved model saved successfully.")

if __name__ == "__main__":
    main()
