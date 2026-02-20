from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from dog_app import which_breed_from_upload

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        return redirect(url_for('thanks'))
    return render_template('contact.html')

@app.route('/thanks', methods=['POST', 'GET'])
def thanks():
    return render_template('thanks.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            with open(path, "rb") as img_file:
                result = which_breed_from_upload(img_file)

            return render_template('result.html',
                                   image_name=filename,
                                   result=result)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)