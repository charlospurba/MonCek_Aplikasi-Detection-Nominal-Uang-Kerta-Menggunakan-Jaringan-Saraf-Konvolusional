import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configurations
class Config:
    img_height = 200
    img_width = 200
    epochs = 50
    batch_size = 16
    learning_rate = 1e-4 

# Load dataset paths
dir_path = './Uang_Baru'
image_paths = []
for dirname, _, filenames in os.walk(dir_path):
    for filename in filenames:
        image_path = os.path.join(dirname, filename)
        image_paths.append(image_path)

# Create DataFrame
df = pd.DataFrame({'path': image_paths})
df['filename'] = df.path.apply(lambda x: os.path.basename(x))
df['nominal'] = df.path.apply(lambda x: os.path.basename(os.path.dirname(x)))
df = df.sample(frac=1).reset_index(drop=True)

# Visualisasi distribusi kelas dalam dataset
filename_vc = df.nominal.value_counts()
plt.figure(figsize=(10, 5))
plt.bar(filename_vc.index, filename_vc.values, color='skyblue')
plt.xticks(rotation=45)
plt.xlabel('Class Labels')
plt.ylabel('Number of Images')
plt.title('Distribution of Images per Class')
plt.show()

# Compute Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['nominal']),
    y=df['nominal'].values
)
class_weight_dict = dict(zip(np.unique(df['nominal']), class_weights))
print("Class Weights Before Mapping:", class_weight_dict)

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    dir_path,
    subset='training',
    validation_split=0.2,
    seed=42,
    image_size=(Config.img_height, Config.img_width),
    batch_size=Config.batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    dir_path,
    subset='validation',
    validation_split=0.2,
    seed=42,
    image_size=(Config.img_height, Config.img_width),
    batch_size=Config.batch_size
)

# Get class names
currency_nominal = train_ds.class_names
num_of_classes = len(currency_nominal)
class_indices = {class_name: i for i, class_name in enumerate(currency_nominal)}
class_weight_dict = {class_indices[class_name]: weight for class_name, weight in class_weight_dict.items()}
print("Adjusted class weights:", class_weight_dict)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(Config.img_height, Config.img_width, 3)),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomBrightness(0.1)
])

# CNN Model with deeper architecture
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_of_classes, activation='softmax')  
])

# Compile model with Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=Config.learning_rate)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Callbacks (Early Stopping + Custom Callback)
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.90:
            print("\nStop training, accuracy > 0.90")
            self.model.stop_training = True

callback = myCallback()
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Train Model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=Config.epochs,
    callbacks=[callback, early_stopping],
    class_weight=class_weight_dict
)

# Plot Accuracy and Loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title('Accuracy')
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.legend()
ax2.set_title('Loss')
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.legend()
plt.show()

# Evaluate Model
loss, accuracy = model.evaluate(val_ds)
print('Accuracy:', accuracy)
print('Loss:', loss)

# Confusion Matrix
predictions = model.predict(val_ds)
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred = np.argmax(predictions, axis=1)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=currency_nominal, yticklabels=currency_nominal)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save Model
model.save('model.keras')