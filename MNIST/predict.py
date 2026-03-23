import argparse
import numpy as np
import joblib
from PIL import Image

def load_mnist_image(path_or_file, invert=False):
    """Load image from path or file-like object and prepare it for MNIST prediction."""
    try:
        if isinstance(path_or_file, str):
            img = Image.open(path_or_file).convert("L")
        else:
            # Assume it's a file-like object (like from Flask upload)
            img = Image.open(path_or_file).convert("L")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
        
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.uint8)

    # MNIST digits are typically white on a black background (pixel values 0-255).
    # If the user provides an image with dark text on a light background, we should invert it.
    if invert:
        arr = 255 - arr
        
    # The Model expects a 1D array of 784 features per sample
    arr = arr.reshape(1, 28*28)
    return arr

def predict_image(image_file, model_path="mnist_model.pkl", invert=False):
    """
    Function to be called by the Flask app or CLI.
    Expects a file path or file-like object.
    Returns a dictionary with result or error.
    """
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        return {"error": f"Model file '{model_path}' not found. Please run 'python train.py' first!"}
        
    x = load_mnist_image(image_file, invert=invert)
    if x is None:
        return {"error": "Failed to process image. Please upload a valid image file."}
        
    prediction = model.predict(x)
    return {"prediction": int(prediction[0])}

def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digit from an image using the trained MNIST model.")
    parser.add_argument("--image", required=True, help="Path to the user's image file.")
    parser.add_argument("--invert", action="store_true", help="Invert image colors (use if image is dark text on a light background).")
    parser.add_argument("--model", default="mnist_model.pkl", help="Path to the trained model file (default: mnist_model.pkl).")
    
    args = parser.parse_args()
    
    result = predict_image(args.image, args.model, args.invert)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Predicted digit for '{args.image}': {result['prediction']}")

if __name__ == "__main__":
    main()
