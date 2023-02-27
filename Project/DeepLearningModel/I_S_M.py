#Image Synthesize Model Prototype

import os
import random
import shutil

# Define paths to labeled data and directories for train/test data
labeled_data_dir = '/path/to/labeled_data'
train_dir = '/path/to/train_dir'
test_dir = '/path/to/test_dir'

# Define the ratio of data to use for training and testing
train_ratio = 0.8
test_ratio = 0.2

# Get a list of all labeled images in the labeled_data_dir
labeled_images = os.listdir(labeled_data_dir)

# Shuffle the list of labeled images
random.shuffle(labeled_images)

# Calculate the number of images to use for training and testing
num_train = int(len(labeled_images) * train_ratio)
num_test = len(labeled_images) - num_train

# Create the train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Move the first num_train images to the train directory
for img in labeled_images[:num_train]:
    src_path = os.path.join(labeled_data_dir, img)
    dst_path = os.path.join(train_dir, img)
    shutil.copy(src_path, dst_path)

# Move the remaining images to the test directory
for img in labeled_images[num_train:]:
    src_path = os.path.join(labeled_data_dir, img)
    dst_path = os.path.join(test_dir, img)
    shutil.copy(src_path, dst_path)

    
import cv2
import numpy as np

# Define the paths to the labeled data and directories for train/test data
train_dir = '/path/to/train_dir'
test_dir = '/path/to/test_dir'

# Define the size of the images to resize to
img_size = (224, 224)

# Define the function to pre-process an image
def preprocess_img(img_path):
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    
    # Resize the image to img_size
    img = cv2.resize(img, img_size)
    
    # Convert the image to RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert the image to a float between 0 and 1
    img = img.astype(np.float32) / 255.0
    
    return img

# Define the function to load and pre-process all images in a directory
def load_images_from_dir(dir_path):
    # Get a list of all image paths in the directory
    img_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jpg')]
    
    # Load and pre-process each image using preprocess_img
    imgs = [preprocess_img(path) for path in img_paths]
    
    return np.array(imgs)

# Load and pre-process the training and testing images
train_imgs = load_images_from_dir(train_dir)
test_imgs = load_images_from_dir(test_dir)

# Define the labels for the training and testing data
train_labels = np.array([1] * len(os.listdir(os.path.join(train_dir, 'positive'))) + 
                        [0] * len(os.listdir(os.path.join(train_dir, 'negative'))))
test_labels = np.array([1] * len(os.listdir(os.path.join(test_dir, 'positive'))) + 
                       [0] * len(os.listdir(os.path.join(test_dir, 'negative'))))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# Load the pre-trained VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False)

# Define the input shape of the model
input_shape = (224, 224, 3)

# Create a new model using the pre-trained VGG16 as a base and adding a Global Average Pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
model = Model(inputs=base_model.input, outputs=x)

# Preprocess the data and load it using an ImageDataGenerator
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=input_shape[:2],
        batch_size=32,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        'test',
        target_size=input_shape[:2],
        batch_size=32,
        class_mode='binary')

# Extract features from the data using the pre-trained VGG16 model
train_features = model.predict(train_generator)
test_features = model.predict(test_generator)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train a logistic regression classifier on the extracted features
logreg = LogisticRegression()
logreg.fit(train_features, train_labels)

# Evaluate the classifier on the test data
y_pred = logreg.predict(test_features)
accuracy = accuracy_score(test_labels, y_pred)
print("Accuracy: {:.2f}".format(accuracy))

from tensorflow.keras.optimizers import SGD

# Fine-tune the pre-trained VGG16 model on the training data
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=train_steps,
                    epochs=10,
                    validation_data=test_generator,
                    validation_steps=test_steps)

# Evaluate the fine-tuned model on the test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
print("Test accuracy: {:.2f}".format(test_accuracy))

model.save('my_model.h5')

from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('my_model.h5')

# Use the model to make predictions on new data
predictions = model.predict(new_data)
