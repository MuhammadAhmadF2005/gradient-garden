from flask import Flask, render_template, request, jsonify
from predict import predict_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        invert = request.form.get('invert', 'false').lower() == 'true'
        result = predict_image(file, invert=invert)
        return jsonify(result)
        
    return jsonify({"error": "Unknown error occurred processing the request."}), 500

if __name__ == '__main__':
    # Start the app in debug mode on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
