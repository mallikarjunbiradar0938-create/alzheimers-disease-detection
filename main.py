import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Use CPU only
tf.config.set_visible_devices([], 'GPU')

# Path to your dataset folder (update this path)
dataset_dir = r'AugmentedAlzheimerDataset'  # Folder containing 4 subfolders

# Image preprocessing and automatic train-test split
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2  # 80% training, 20% validation
)

# Training data generator
train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

# Validation (test) data generator
val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

# Model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Evaluate model
loss, acc = model.evaluate(val_data)
print(f"Validation Accuracy: {acc*100:.2f}%")

# Save model
model.save("alzheimers_model.h5")
print("Model saved as alzheimers_model.h5")
