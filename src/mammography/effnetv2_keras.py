# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras_cv
import math
BATCH_SIZE = 256
EPOCHS = 100
IMG_SIZE = 299
CHANNELS = 1

def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label_normal': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
    image = tf.reshape(image, [299, 299, 1])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0

    label = parsed_features['label']

    label = tf.one_hot(label, 5, dtype=tf.float32)

    return image, label
def create_dataset(tfrecord_files, batch_size=BATCH_SIZE):
    """Create dataset from TFRecord files"""
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
def create_numpy_dataset(data_path, labels_path, batch_size=BATCH_SIZE, is_training=False):
    if isinstance(data_path, str):
        data = np.load(data_path, mmap_mode='r')
        labels = np.load(labels_path, mmap_mode='r')
    else:
        data = data_path
        labels = labels_path

    def generator():
        for i in range(len(data)):
            # Process image
            image = data[i].reshape(299, 299, 1)
            image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
            #image = image.astype('uint8')
            image = tf.cast(image, tf.float32) / 255.0

            label_index = labels[i]
            one_hot_label = np.zeros(5, dtype=np.float32)
            one_hot_label[label_index] = 1.0

            yield image, one_hot_label

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(5,), dtype=tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
print("Loading training dataset...")
train_files = [f'/kaggle/input/ddsm-mammography/training10_{i}/training10_{i}.tfrecords' for i in range(5)]
train_dataset = create_dataset(train_files, BATCH_SIZE)
import matplotlib.pyplot as plt
import numpy as np


def display_batch(dataset, num_images=8):
    """
    Display images from a batch of the dataset
    """
    # Get one batch
    for images, labels in dataset.take(1):
        plt.figure(figsize=(15, 8))

        # Display up to num_images from the batch
        for i in range(min(num_images, len(images))):
            plt.subplot(2, 4, i + 1)

            # Convert tensor to numpy and display
            img = images[i].numpy()
            label = np.argmax(labels[i].numpy())

            # Map label index to class name
            label_names = ['Normal', 'Abnormal Type 1', 'Abnormal Type 2',
                           'Abnormal Type 3', 'Abnormal Type 4']
            title = f'Label: {label_names[label]}'

            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')

        plt.tight_layout()
        plt.show()


# Display images from your dataset
display_batch(train_dataset)

# To display multiple batches, you can use:
# for batch in train_dataset.take(10):
#    display_batch(train_dataset)
#Validation et Test Data needs to be combined to mix Masses and Calcifications
print("Loading and combining validation/test data...")
test_data = np.load('/kaggle/input/ddsm-mammography/test10_data/test10_data.npy', mmap_mode='r')
test_labels = np.load('/kaggle/input/ddsm-mammography/test10_labels.npy', mmap_mode='r')

cv_data = np.load('/kaggle/input/ddsm-mammography/cv10_data/cv10_data.npy', mmap_mode='r')
cv_labels = np.load('/kaggle/input/ddsm-mammography/cv10_labels.npy', mmap_mode='r')

combined_data = np.concatenate([test_data, cv_data])
combined_labels = np.concatenate([test_labels, cv_labels])

#Shuffling the Data
#np.random.shuffle(combined_data)
indices = np.random.permutation(len(combined_data))
combined_data = combined_data[indices]
combined_labels = combined_labels[indices]

val_split = int(len(combined_data) * 0.5)
val_dataset = create_numpy_dataset(
    combined_data[:val_split],
    combined_labels[:val_split],
    BATCH_SIZE,
    is_training=False
)

test_dataset = create_numpy_dataset(
    combined_data[val_split:],
    combined_labels[val_split:],
    BATCH_SIZE,
    is_training=False
)

class_names = ['Negative', 'Benign Calcification', 'Benign Mass',
               'Malignant Calcification', 'Malignant Mass']

print(train_dataset.take(1))
print(val_dataset.take(2))

def print_label_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    print("\nLabel distribution:")
    for label_idx, count in dist.items():
        print(f"{class_names[label_idx]}: {count} samples ({count/len(labels)*100:.2f}%)")

print("\nValidation set:")
print_label_distribution(combined_labels[:val_split])
print("\nTest set:")
print_label_distribution(combined_labels[val_split:])


def calculate_class_weights(train_dataset, strategy='basic'):
    """
    Calculate class weights with advanced boosting strategies for minority classes
    """
    class_counts = np.zeros(5)
    total_samples = 0

    # Count samples per class
    for _, labels in train_dataset:
        batch_labels = labels.numpy()
        class_counts += np.sum(batch_labels, axis=0)
        total_samples += len(batch_labels)

    # Calculate base frequencies
    epsilon = 1e-7
    class_frequencies = class_counts / total_samples

    if strategy == 'basic':
        # Original inverse frequency weighting
        class_weights = 1 / (class_frequencies + epsilon)
    elif strategy == 'custom':
        class_weights = 1 / (class_frequencies + epsilon)

        boost_factors = np.array([
            1.0,  # Normal - no boost needed
            2.0,  # Benign Calc - significant boost
            1.8,  # Benign Mass - moderate boost
            2.5,  # Malignant Calc - highest boost (critical class)
            2.0  # Malignant Mass - significant boost
        ])

        class_weights = class_weights * boost_factors

    # Handle zero-count classes
    class_weights[class_counts == 0] = 0.0

    # Normalize weights
    if np.sum(class_weights) > 0:
        class_weights = class_weights * len(class_counts) / np.sum(class_weights)

        # Ensure minimum weight is 1.0
        min_weight = np.min(class_weights[class_weights > 0])
        class_weights = class_weights / min_weight

    class_weights_dict = dict(enumerate(class_weights))

    # Print detailed statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {total_samples}")
    print("\nClass Distribution:")
    label_names = ['Normal', 'Benign Calc', 'Benign Mass',
                   'Malignant Calc', 'Malignant Mass']

    print("\nDetailed Class Analysis:")
    print(f"{'Class':<15} {'Count':>8} {'Frequency':>12} {'Weight':>10}")
    print("-" * 45)

    for i, (count, freq, weight) in enumerate(zip(class_counts,
                                                  class_frequencies,
                                                  class_weights)):
        print(f"{label_names[i]:<15} {int(count):8d} {freq:12.4f} {weight:10.4f}")

    print("\nWeight Statistics:")
    print(f"Mean weight: {np.mean(class_weights):.4f}")
    print(f"Max/Min ratio: {np.max(class_weights) / np.min(class_weights[class_weights > 0]):.4f}")

    return class_weights_dict


class_weights = calculate_class_weights(train_dataset, strategy='custom')

import tensorflow as tf


def create_custom_efficientnet_v2(img_size=IMG_SIZE, num_classes=5):
    """
    Creates a custom EfficientNetV2 model for breast imaging classification
    with additional regularization and custom top layers.

    Parameters:
      img_size: int, the height and width of the input image.
      num_classes: int, number of classes for classification.

    Returns:
      A tf.keras.Model instance.
    """
    # Load the base EfficientNetV2S model without its top layers.
    # It expects a 3-channel input so we set input_shape accordingly.
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )

    # Optionally freeze the base model during training.
    base_model.trainable = False

    # Define the input layer for single channel images.
    inputs = tf.keras.Input(shape=(img_size, img_size, 1))

    # Convert single channel input to 3 channels using a 1x1 convolution.
    x = tf.keras.layers.Conv2D(3, (1, 1), padding='same')(inputs)

    # Pass through the EfficientNetV2 base.
    x = base_model(x, training=False)

    # Global average pooling.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # First dense block with L2 regularization, batch normalization and dropout.
    x = tf.keras.layers.Dense(512, activation='leaky_relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Second dense block.
    x = tf.keras.layers.Dense(256, activation='leaky_relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Output layer with softmax activation.
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create and return the model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def compile_and_prepare_model(model):
    # Learning rate schedule
    initial_learning_rate = 2e-3
    decay_steps = 1000
    decay_rate = 0.9

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    # Optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=initial_learning_rate,
        weight_decay=0.01
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )

    return model

model = create_custom_efficientnet_v2()
model = compile_and_prepare_model(model)

print(model.summary())

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,                  # Number of epochs with no improvement after which training will be stopped
    min_delta=1e-7,              # Minimum change in the monitored quantity to be considered an improvement
    restore_best_weights=True,
)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,           # Factor by which the learning rate will be reduced (new_lr = lr * factor)
    patience=5,
    min_delta=1e-7,
    cooldown=0,           # Number of epochs to wait before resuming normal operation after learning rate reduction
    verbose=1
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

history = model.fit(train_dataset,
    validation_data=val_dataset,
    epochs = 120,
    batch_size = BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stopping, plateau, model_checkpoint])

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Just get test results
results = model.evaluate(test_dataset)
print("\nTest Results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# Get predictions for confusion matrix
y_pred = np.argmax(model.predict(test_dataset), axis=1)
y_true = np.concatenate([np.argmax(labels, axis=1)
                        for _, labels in test_dataset])

# Plot confusion matrix
class_names = ['Normal', 'Benign Calc', 'Benign Mass',
               'Malignant Calc', 'Malignant Mass']
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()