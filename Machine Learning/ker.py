from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('currency_recognition_model_with_keras_augmentation.h5')

# Daftar nama kelas sesuai urutan indeks dalam model
class_names = ['10 k', '100 RB', '1k', '20 K', '2k', '50 K', '5k']

# Set the test image folder
test_image_folder = './test_images'

# Prepare lists for images and their predictions
images = []
titles = []

# Iterate over all images in the test folder
for image_name in os.listdir(test_image_folder):
    # Get the full path of the image
    image_path = os.path.join(test_image_folder, image_name)
    
    # Load and preprocess the image
    try:
        img = image.load_img(image_path, target_size=(200, 200))  # Resize to match the model's input size
        x = image.img_to_array(img)  # Convert to array
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(x)
        predicted_class = np.argmax(prediction)  # Get the index of the highest probability
        predicted_label = class_names[predicted_class]  # Get the class name
        
        # Append image and title
        images.append(img)
        titles.append(f"{image_name}\nPredicted: {predicted_label}")
    except Exception as e:
        print(f"Error processing {image_name}: {e}")

# Display all images in a single frame (grid)
num_images = len(images)
cols = 4  # Number of columns in the grid
rows = (num_images + cols - 1) // cols  # Calculate rows needed

plt.figure(figsize=(15, rows * 5))
for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i], fontsize=10)
    plt.axis('off')  # Hide axes
plt.tight_layout()
plt.show()
