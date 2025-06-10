import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Class Config
class Config:
    img_height = 200
    img_width = 200
    epochs = 50
    batch_size = 32
    learning_rate = 1e-3
    model_path = 'model.keras'  
    test_image_folder = 'test_images'  

# Daftar kelas sesuai model
currency_nominal = ['10 k', '100 RB', '1k', '20 K', '2k', '50 K', '5k']

# Memuat model
if not os.path.exists(Config.model_path):
    raise FileNotFoundError(f"Model file '{Config.model_path}' tidak ditemukan!")
model = load_model(Config.model_path)

# Fungsi untuk memproses gambar
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(Config.img_height, Config.img_width))
    input_arr = img_to_array(image) / 255.0  # Normalisasi
    return tf.expand_dims(input_arr, axis=0)

# Fungsi untuk memprediksi gambar
def predict_image(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    scores = tf.nn.softmax(predictions[0])
    predicted_class = currency_nominal[np.argmax(scores)]
    confidence = 100 * np.max(scores)
    return predicted_class, confidence

# Memproses semua gambar di folder uji
if os.path.exists(Config.test_image_folder):
    plt.figure(figsize=(10, 10))
    images = os.listdir(Config.test_image_folder)[:9]  # Batasi 9 gambar
    for i, image_file in enumerate(images):
        image_path = os.path.join(Config.test_image_folder, image_file)
        if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            predicted_class, confidence = predict_image(image_path)
            print(f"Gambar: {image_file} | Prediksi: {predicted_class} | Keyakinan: {confidence:.2f}%")

            # Visualisasi gambar
            ax = plt.subplot(3, 3, i + 1)
            image = load_img(image_path)
            plt.imshow(image)
            plt.title(f"{predicted_class} ({confidence:.2f}%)")
            plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print(f"Folder '{Config.test_image_folder}' tidak ditemukan.")
