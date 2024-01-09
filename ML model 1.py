import os
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Concatenate
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import random

# Set the path to your image dataset
image_dataset_path = r"D:\adhvik\adh\Hackathon\space hack\Data RR\hack code\picture dataset"

# Function to load and preprocess an image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

# Function to generate pairs of images (positive and negative pairs)
def generate_pairs(image_paths, labels):
    pairs = []
    pair_labels = []

    for i in range(len(image_paths)):
        # Ensure there are enough images remaining to create a pair
        if i + 1 < len(image_paths):
            pairs.append([[image_paths[i], labels[i]], [image_paths[i + 1], labels[i + 1]]])
            pair_labels.append(labels[i] == labels[i + 1])

        # Ensure there are enough images remaining to create a negative pair
        if i + 1 < len(image_paths) and i + 2 < len(image_paths):
            j = random.choice([idx for idx in range(len(image_paths)) if idx != i and idx != i + 1])
            pairs.append([[image_paths[i], labels[i]], [image_paths[j], labels[j]]])
            pair_labels.append(labels[i] != labels[j])

    return np.array(pairs), np.array(pair_labels)

# Get a list of all image files in the dataset path
image_files = [os.path.join(image_dataset_path, file) for file in os.listdir(image_dataset_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))]

# Assign labels to images (for simplicity, using integer labels here)
labels = [i // 2 for i in range(len(image_files))]

# Generate pairs and labels
pairs, labels = generate_pairs(image_files, labels)

# Split the data into training and testing sets
train_pairs, test_pairs, train_labels, test_labels = train_test_split(pairs, labels, test_size=0.2, random_state=42)

# Define the base MobileNetV2 model (pre-trained on ImageNet)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Create the Siamese model architecture
input_a = Input(shape=(224, 224, 3))
input_b = Input(shape=(224, 224, 3))

processed_a = base_model(input_a)
processed_b = base_model(input_b)

flattened_a = Flatten()(processed_a)
flattened_b = Flatten()(processed_b)

merged_output = Concatenate()([flattened_a, flattened_b])
dense1 = Dense(128, activation='relu')(merged_output)
output_layer = Dense(1, activation='sigmoid')(dense1)

siamese_model = Model(inputs=[input_a, input_b], outputs=output_layer)

# Compile the Siamese model
siamese_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the Siamese model
# Train the Siamese model
siamese_model.fit(
    [load_and_preprocess_image(pair[0][0]) for pair in train_pairs],
    train_labels,
    epochs=10,
    batch_size=32,
    validation_data=(
        [load_and_preprocess_image(pair[0][0]) for pair in test_pairs],
        test_labels
    )
)
