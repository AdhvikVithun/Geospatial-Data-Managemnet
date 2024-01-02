import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.losses import triplet_semihard_loss


# Load dataset into a DataFrame
df = pd.read_csv('siamese_dataset.csv')

# Get list of all file paths
filepaths = df['File1'].tolist() + df['File2'].tolist()
filepaths = list(set(filepaths))

# Dictionary to store file hash and path
file_dict = {}

# Calculate hash for each file
for filepath in filepaths:
    with open(filepath, 'rb') as f:
        filehash = hashlib.md5(f.read()).hexdigest()
    file_dict[filepath] = filehash

# Create triplet samples
anchor_paths = []
positive_paths = []
negative_paths = []

for index, row in df.iterrows():
    hash1 = row['HashFile1']
    hash2 = row['HashFile2']

    if hash1 == hash2:
        # Files are duplicates
        anchor_paths.append(row['File1'])
        positive_paths.append(row['File2'])
    else:
        # Files are non-duplicates
        anchor_paths.append(row['File1'])
        negative_paths.append(row['File2'])

# Split data into training and validation sets
anchor_train, anchor_val, positive_train, positive_val = train_test_split(
    anchor_paths, positive_paths, test_size=0.2, random_state=42
)

# Placeholder function for hash network
def create_hash_network(input_size):
    input_layer = Input(shape=(input_size,))
    hidden_layer = Dense(64, activation='relu')(input_layer)
    output_layer = Dense(32, activation='sigmoid')(hidden_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create and compile the hash network
hash_network = create_hash_network(input_size=32)  # You need to adjust input_size based on your hash length
hash_network.compile(optimizer=Adam(), loss='binary_crossentropy')

# Function to generate triplet data
def generate_triplet_data(anchor_paths, positive_paths, negative_paths, file_dict, batch_size=32):
    while True:
        batch_anchor = np.zeros((batch_size, 32))  # You need to adjust dimensions based on your hash length
        batch_positive = np.zeros((batch_size, 32))
        batch_negative = np.zeros((batch_size, 32))

        for i in range(batch_size):
            anchor_path = np.random.choice(anchor_paths)
            positive_path = np.random.choice(positive_paths)
            negative_path = np.random.choice(negative_paths)

            batch_anchor[i, :] = file_dict[anchor_path]
            batch_positive[i, :] = file_dict[positive_path]
            batch_negative[i, :] = file_dict[negative_path]

        yield [batch_anchor, batch_positive, batch_negative], []

# Train triplet network
triplet_model = create_triplet_model(embedding_size=32)  # You need to adjust embedding_size based on your hash length
triplet_model.compile(optimizer=Adam(), loss=triplet_semihard_loss())
triplet_model.fit(
    generate_triplet_data(anchor_train, positive_train, negative_paths, file_dict),
    steps_per_epoch=len(anchor_train) // 32,
    epochs=10,
    validation_data=generate_triplet_data(anchor_val, positive_val, negative_paths, file_dict),
    validation_steps=len(anchor_val) // 32
)

# Evaluate network on file hashes to find duplicates
duplicates = []
for filepath1 in filepaths:
    hash1 = file_dict[filepath1]
    for filepath2 in filepaths:
        if filepath1 == filepath2:
            continue

        hash2 = file_dict[filepath2]

        # Evaluate network
        anchor_embedding = hash_network.predict(np.array([hash1]))
        positive_embedding = hash_network.predict(np.array([hash2]))

        distance = np.linalg.norm(anchor_embedding - positive_embedding)

        # Set a threshold for considering duplicates
        if distance < 0.5:
            duplicates.append((filepath1, filepath2))

print(duplicates)
