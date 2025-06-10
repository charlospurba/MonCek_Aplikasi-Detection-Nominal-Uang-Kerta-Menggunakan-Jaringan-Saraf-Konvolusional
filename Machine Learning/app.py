from flask import Flask, request, jsonify, send_file
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import io

app = Flask(__name__)

model = tf.keras.models.load_model('model.keras')

class_names = ['10 k', '100 RB', '1k', '20 K', '2k', '50 K', '5k']

# Folder Audio
audio_folder = './Audio'

# Folder untuk menyimpan gambar yang diunggah
upload_folder = './Uploaded_Images'
os.makedirs(upload_folder, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pastikan file diunggah
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        
        # Simpan file yang diunggah
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Load dan preprocess gambar
        img = image.load_img(file_path, target_size=(200, 200))  
        x = image.img_to_array(img)  
        x = np.expand_dims(x, axis=0)  
        
        # Lakukan prediksi
        prediction = model.predict(x)
        predicted_class = np.argmax(prediction)  
        predicted_label = class_names[predicted_class] 
        
        # Cari file audio sesuai nama kelas
        audio_file = os.path.join(audio_folder, f"{predicted_label}.mp3")
        if not os.path.exists(audio_file):
            audio_link = None
        else:
            audio_link = f"https://d7db-103-177-96-43.ngrok-free.app/audio/{predicted_label}.mp3"

        # Kembalikan hasil prediksi dan tautan audio
        return jsonify({
            "predicted_class": predicted_label,
            "audio_link": audio_link
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint untuk mengakses file audio
@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    audio_path = os.path.join(audio_folder, filename)
    if os.path.exists(audio_path):
        return send_file(audio_path)
    else:
        return jsonify({"error": "Audio file not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)

