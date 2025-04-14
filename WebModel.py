from flask import Flask, request, render_template, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
app.secret_key = 'supersecretkey' 
app.config['UPLOAD_FOLDER'] = 'uploads'
model_path =  'vggtest (1).keras'
target_size = (244, 244)

# Load the trained model
model = load_model(model_path)

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img = np.array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Expand dimensions to match batch size
        return img
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        raise

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Clear previous session data
        session.pop('filename', None)
        session.pop('prediction_text', None)
        session.pop('uploaded', None)
        session.pop('error', None)
        
        # Check if the post request has the file part
        if 'file' not in request.files:
            session['error'] = 'No file part'
            return redirect(url_for('upload_file'))

        file = request.files['file']

        if file.filename == '':
            session['error'] = 'No selected file'
            return redirect(url_for('upload_file'))

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the uploaded image
            try:
                processed_image = preprocess_image(filepath)

                # Make prediction
                prediction = model.predict(processed_image)
                predicted_class = np.argmax(prediction)

                # Define your class labels based on your training setup
                class_labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

                prediction_text = class_labels[predicted_class]
                session['prediction_text'] = prediction_text
                session['filename'] = filename
                session['uploaded'] = True
                return redirect(url_for('upload_file'))
            except Exception as e:
                session['error'] = f"Error in prediction: {e}"
                return redirect(url_for('upload_file'))

    return render_template('index.html', 
                           prediction_text=session.get('prediction_text'), 
                           filename=session.get('filename'), 
                           uploaded=session.get('uploaded'), 
                           error=session.get('error'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
