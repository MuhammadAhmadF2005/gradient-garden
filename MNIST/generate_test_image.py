from sklearn.datasets import fetch_openml
import numpy as np
from PIL import Image

def generate_sample():
    print("Loading one image from MNIST to test...")
    mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
    X = mnist["data"]
    
    # Extract the first image (which is a 5)
    img_array = X.iloc[0].to_numpy().reshape(28, 28).astype(np.uint8)
    
    # Save as sample_5.png
    img = Image.fromarray(img_array)
    img.save("sample_5.png")
    print("Saved sample_5.png successfully!")

if __name__ == "__main__":
    generate_sample()
