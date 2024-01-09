import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def load_images(image_path):
    image_files = os.listdir(image_path)
    images = [cv2.imread(os.path.join(image_path, file)) for file in tqdm(image_files, desc="Loading Images")]
    return images

def apply_distortions(image):
    angle = np.random.uniform(-10, 10)
    rotated_image = rotate_image(image, angle)
    return rotated_image

def rotate_image(image, angle):
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def save_image_pairs(args):
    anchor_set, paired_set, output_path, label, i = args
    pair = np.concatenate((anchor_set[i], paired_set[i]), axis=1)
    cv2.imwrite(os.path.join(output_path, f"{label}_pair_{i + 1}.png"), pair)

def generate_positive_set(args):
    anchor_set, output_path = args
    return [apply_distortions(image) for image in tqdm(anchor_set, desc="Generating Positive Set")]

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

if __name__ == "__main__":
    main()
