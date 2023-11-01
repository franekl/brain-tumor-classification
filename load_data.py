import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def get_image_data(path, width, height):
    img = cv2.imread(path)  # load
    return cv2.resize(img, (width, height))  # resize


def load_dataset(IMG_WIDTH=225, IMG_HEIGHT=225):
    base_path = "data"
    images, labels = [], []
    label_names = os.listdir(base_path)
    label_map = {name: idx for idx, name in enumerate(label_names)}

    for label_name, label_idx in label_map.items():
        folder_path = os.path.join(base_path, label_name)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = get_image_data(img_path, IMG_WIDTH, IMG_HEIGHT)
            images.append(img)
            labels.append(label_idx)

    return np.array(images), np.array(labels), label_map


def augment_data(images, labels, augmentations=2):
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        fill_mode="nearest",
    )

    augmented_images, augmented_labels = [], []
    for img, label in zip(images, labels):
        augmented_images.append(img)
        augmented_labels.append(label)
        for _ in range(augmentations):
            augmented = datagen.flow(np.expand_dims(img, 0), batch_size=1).next()[0]
            augmented_images.append(np.squeeze(augmented).astype(np.uint8))
            augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)


def split_and_augment_dataset(images, labels, eval_set=False, random_state=None):
    if eval_set:
        # Split into train+eval and test
        (
            train_eval_images,
            test_images,
            train_eval_labels,
            test_labels,
        ) = train_test_split(
            images, labels, test_size=0.2, random_state=random_state, stratify=labels
        )
        # Split train+eval into train and eval
        train_images, eval_images, train_labels, eval_labels = train_test_split(
            train_eval_images,
            train_eval_labels,
            test_size=0.25,
            random_state=random_state,
            stratify=train_eval_labels,
        )
        # Augment only the training data
        augmented_train_images, augmented_train_labels = augment_data(
            train_images, train_labels
        )
        return (
            augmented_train_images,
            augmented_train_labels,
            eval_images,
            eval_labels,
            test_images,
            test_labels,
        )
    else:
        # Split into train and test
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.2, random_state=random_state, stratify=labels
        )
        # Augment only the training data
        augmented_train_images, augmented_train_labels = augment_data(
            train_images, train_labels
        )
        return augmented_train_images, augmented_train_labels, test_images, test_labels
