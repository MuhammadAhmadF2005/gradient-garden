import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

def main():
    print("Fetching MNIST dataset (this may take a moment)...")
    # Using fetch_openml with cache=True is the recommended way
    # Explicitly set parser='auto' to avoid future warnings
    mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
    
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    
    # Standard MNIST split: first 60000 for training, rest for testing
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    
    print("Training SGDClassifier on the full dataset...")
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)
    
    model_path = "mnist_model.pkl"
    print(f"Saving the trained model to {model_path}...")
    joblib.dump(sgd_clf, model_path)
    
    print("Training complete and model saved successfully.")

if __name__ == "__main__":
    main()
