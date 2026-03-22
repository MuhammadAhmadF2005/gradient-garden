import argparse
import numpy as np
import joblib
from PIL import Image

def load_mnist_image(path, invert=False):
    """Load image and prepare it for MNIST prediction."""
    try:
        img = Image.open(path).convert("L")
    except Exception as e:
        print(f"Error loading image '{path}': {e}")
        return None
        
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.uint8)

    # MNIST digits are typically white on a black background (pixel values 0-255).
    # If the user provides an image with dark text on a light background, we should invert it.
    if invert:
        arr = 255 - arr
        
    # The SGDClassifier expects a 1D array of 784 features per sample
    arr = arr.reshape(1, 28*28)
    return arr

def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digit from an image using the trained MNIST model.")
    parser.add_argument("--image", required=True, help="Path to the user's image file.")
    parser.add_argument("--invert", action="store_true", help="Invert image colors (use if your image has dark text on a light background).")
    parser.add_argument("--model", default="mnist_model.pkl", help="Path to the trained model file (default: mnist_model.pkl).")
    
    args = parser.parse_args()
    
    # Load the trained model
    try:
        model = joblib.load(args.model)
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found.")
        print("Please run 'python train.py' first to generate and save the model.")
        return
        
    # Process the provided image
    x = load_mnist_image(args.image, invert=args.invert)
    if x is None:
        return
        
    # Make the prediction
    prediction = model.predict(x)
    print(f"Predicted digit for '{args.image}': {prediction[0]}")

if __name__ == "__main__":
    main()
