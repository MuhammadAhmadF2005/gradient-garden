# MNIST ML Pipeline & Web Application

This project transforms a standard Jupyter Notebook analysis into a fully reproducible, end-to-end Machine Learning pipeline with a premium Graphical User Interface.

## Architecture

The project consists of three main components:
1. **Model Training Generator (`train.py`)**: Responsible for securely fetching the MNIST dataset, preparing the images, executing a `StandardScaler`, and training a robust `SGDClassifier` linearly. It caches its results into a `.pkl` file.
2. **Command Line Predictor (`predict.py`)**: A command line utility containing modular functions to process images (via `Pillow`), invert colors, extract image tensors, and classify digits.
3. **Web Application (`app.py`, `templates/`, `static/`)**: A modern, responsive Flask web application that allows users to drag and drop images and receive model predictions in real-time without touching the terminal.

## Installation

Ensure you have Python installed, then install the necessary dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model
Before generating predictions, you must train the ML model and cache it locally:
```bash
python train.py
```
*This will fetch the dataset, fit the model on 60,000 data points, and generate the `mnist_model.pkl` artifact required for the predictor.*

### 2. Beautiful Web Interface (Recommended)
To launch the graphical user interface, start the Flask web server:
```bash
python app.py
```
Then, open your favorite browser and navigate to exactly **`http://localhost:5000/`**. 
You can **drag and drop** any image snippet containing a handwritten digit (0-9). 

*(Note: MNIST images are natively white digits on a black background. If your image has dark ink/pencil writing on a white/light background, make sure to toggle the "Invert Colors" switch in our UI!)*

### 3. Command Line Interface (CLI)
You can also use the backend predictor directly from your terminal:
```bash
python predict.py --image <path_to_your_image.png>
```
*Add the `--invert` flag if your image is dark text on a light background!*
