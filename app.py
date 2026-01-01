from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('model/alzheimers_model.h5')

# Class labels must match your training order
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Hospital info
HOSPITALS = [
    {"name": "Aarogya Hospital", "location": "Bidar", "link": "https://www.justdial.com/Bidar/Aarogya-Hospital-Bidar-Bazar/9999P8482-8482-180216083341-A5S8_BZDET"},
    {"name": "Vasu Hospital", "location": "Bidar", "link": "https://www.justdial.com/Bidar/Vasu-Hospital-Near-Mailoor-Cross-Bidar-Bazar/9999P8482-8482-140724110051-L4R8_BZDET"},
    {"name": "Gudge Hospital", "location": "Bidar", "link": "https://www.justdial.com/Bidar/Gudage-Hospital-Stadeam-Road-Bidar-Ho/9999P8482-8482-170111100849-Y7W4_BZDET"},
    {"name": "Yashosai Hospital", "location": "Bidar", "link": "https://www.justdial.com/Bidar/Yashosai-Hospital-Near-Hanuman-Temple-Adarsh-Colony/9999P8482-8482-211012220228-S1M3_BZDET"},
    {"name": "Wali Shree Hospital", "location": "Bidar", "link": "https://www.justdial.com/Bidar/Wali-Shree-Hospital-Super-Speciality-Centre-Behind-Megur-Eye-Care-Center-Devi-Colony/9999P8482-8482-180606200019-S1S5_BZDET"},
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        location = request.form['location']
        mri_file = request.files['mri']

        # Save MRI image
        upload_folder = os.path.join('static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        img_path = os.path.join(upload_folder, mri_file.filename)
        mri_file.save(img_path)

        # Preprocess image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        pred_index = np.argmax(prediction)
        pred_stage = CLASS_NAMES[pred_index]
        confidence = np.max(prediction) * 100

        nearby_hospitals = HOSPITALS[:4]

        return render_template(
            'result.html',
            result=pred_stage,
            confidence=f"{confidence:.2f}%",
            name=name,
            age=age,
            location=location,
            hospitals=nearby_hospitals,
            mri_image=f'uploads/{mri_file.filename}'
        )
    return render_template('predict.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/hospitals')
def hospitals():
    return render_template('hospitals.html', hospitals=HOSPITALS)

if __name__ == '__main__':
    app.run(debug=True)
