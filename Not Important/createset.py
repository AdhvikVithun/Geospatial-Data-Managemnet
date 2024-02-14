import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from keras.optimizers import Adam
from keras import backend as K

# Function to load images
def load_images(image_path):
    image_files = os.listdir(image_path)
    images = [cv2.imread(os.path.join(image_path, file)) for file in tqdm(image_files, desc="Loading Images")]
    return images

# Function to apply distortions to images
def apply_distortions(image):
    angle = np.random.uniform(-10, 10)
    rotated_image = rotate_image(image, angle)
    return rotated_image

# Function to rotate an image
def rotate_image(image, angle):
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

# Function to create Siamese model
def create_siamese_model(input_shape):
    input_anchor = Input(shape=input_shape)
    input_positive = Input(shape=input_shape)
    input_negative = Input(shape=input_shape)

    # Shared Convolutional Neural Network (CNN) layers
    shared_cnn = create_shared_cnn(input_shape)

    encoded_anchor = shared_cnn(input_anchor)
    encoded_positive = shared_cnn(input_positive)
    encoded_negative = shared_cnn(input_negative)

    # Euclidean distance between the anchor and positive encoded representations
    positive_distance = Lambda(euclidean_distance, name='positive_distance')([encoded_anchor, encoded_positive])
    # Euclidean distance between the anchor and negative encoded representations
    negative_distance = Lambda(euclidean_distance, name='negative_distance')([encoded_anchor, encoded_negative])

    # Concatenate the distances
    outputs = [positive_distance, negative_distance]

    siamese_model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=outputs)

    return siamese_model

# Function to create shared CNN for encoding images
def create_shared_cnn(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    return model

# Function to calculate Euclidean distance between two vectors
def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# Function to generate batches of training data
def generate_batch(anchor_set, positive_set, negative_set, batch_size):
    while True:
        # Randomly select indices for a batch
        indices = np.random.choice(len(anchor_set), batch_size, replace=False)

        # Generate batches of anchor, positive, and negative images
        anchor_batch = [anchor_set[i] for i in indices]
        positive_batch = [positive_set[i] for i in indices]
        negative_batch = [negative_set[i] for i in indices]

        # Yield the batches
        yield ([np.array(anchor_batch), np.array(positive_batch), np.array(negative_batch)], 
               [np.zeros(batch_size), np.ones(batch_size)])

# Function to save image pairs
def save_image_pairs(args):
    anchor_set, paired_set, output_path, label, i = args
    pair = np.concatenate((anchor_set[i], paired_set[i]), axis=1)
    np.save(os.path.join(output_path, f"{label}_pair_{i + 1}.npy"), pair)


def main():
    image_path = r"D:\adhvik\adh\Hackathon\space hack\siamese data lulc\imagedata"
    output_path = r"D:\adhvik\adh\Hackathon\space hack\siamese data lulc\outdata"

    images = load_images(image_path)

    np.random.seed(42)

    # Select indices of a subset of unique images as anchor set
    anchor_set_size = 1000
    indices = np.random.choice(len(images), anchor_set_size, replace=False)
    anchor_set = [images[i] for i in indices]

    # Generate positive set with slight distortions using parallel processing
    with Pool(cpu_count()) as pool:
        positive_set = pool.map(apply_distortions, anchor_set)

    # Generate negative set by reversing the order of the anchor set
    negative_set = anchor_set[::-1]

    # Save image pairs (anchor, positive) and (anchor, negative) using parallel processing
    with Pool(cpu_count()) as pool:
        args_positive = [(anchor_set, positive_set, output_path, "positive", i) for i in range(len(anchor_set))]
        args_negative = [(anchor_set, negative_set, output_path, "negative", i) for i in range(len(anchor_set))]

        pool.map(save_image_pairs, args_positive)
        pool.map(save_image_pairs, args_negative)

    # Load saved pairs
    anchor_set = np.load(os.path.join(output_path, "positive_pair_1.npy"))
    positive_set = np.load(os.path.join(output_path, "positive_pair_1.npy"))
    negative_set = np.load(os.path.join(output_path, "negative_pair_1.npy"))

    # Specify input shape based on your image dimensions
    input_shape = anchor_set[0].shape

    # Create a Siamese model
    siamese_model = create_siamese_model(input_shape)

    # Compile the model
    siamese_model.compile(optimizer=Adam(), loss=['mean_squared_error', 'mean_squared_error'])

    # Specify batch size and number of epochs
    batch_size = 32
    epochs = 10

    # Train the Siamese model
    siamese_model.fit(generate_batch(anchor_set, positive_set, negative_set, batch_size),
                      steps_per_epoch=len(anchor_set)//batch_size,
                      epochs=epochs)

    # Save the trained Siamese model
    siamese_model.save("siamese_model.h5")

if __name__ == "__main__":
    main()
