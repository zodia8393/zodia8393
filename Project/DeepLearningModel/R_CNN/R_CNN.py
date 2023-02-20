# Install necessary libraries
!pip install numpy
!pip install opencv-python
!pip install keras
!pip install tensorflow
!pip install scikit-learn

import cv2
import numpy as np
import os

# Set paths to your data directories
train_data_path = 'path/to/train/data'
test_data_path = 'path/to/test/data'

#Image Segmentation Using DCGAN (Warning! Dont try this code Without Data)

import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define the DCGAN generator model
def generator_model(input_shape=(100,)):
    model = Sequential()
    model.add(Dense(256 * 4 * 4, input_shape=input_shape))
    model.add(Reshape((4, 4, 256)))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'))
    return model

# Define the DCGAN discriminator model
def discriminator_model(input_shape=(64, 64, 3)):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Combine the generator and discriminator to form the DCGAN
def dcgan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

# Load the real images from the dataset
def load_real_images(dataset_path, n_samples):
    images = []
    for filename in os.listdir(dataset_path):
        if len(images) >= n_samples:
            break
        image = Image.open(os.path.join(dataset_path, filename))
        image = np.asarray(image)
        images.append(image)
    return np.asarray(images)

# Generate n fake images using the DCGAN generator
def generate_fake_images(generator, latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    images = generator.predict(x_input)
    return (images + 1) / 2.0

# Train the DCGAN
def train_dcgan(generator, discriminator, dcgan, dataset_path, n_epochs=100, n_batch=128, latent_dim=100):
    # Load the real images
    X_real = load_real_images(dataset_path, n_batch)
    y_real = np.ones((n_batch, 1))
    # Train the discriminator on real images
    discriminator.train_on_batch(X_real, y_real)
    # Train the generator and discriminator together
    for i in range(n_epochs):
        # Prepare fake images and labels
        X_fake = generate_fake_images(generator, latent_dim, n_batch)
        y_fake = np.zeros((n_batch, 1))
        # Train the discriminator on fake images
        discriminator.train_on_batch(X_fake, y_fake)
        # Train the generator via the discriminator's error
        X_gan = np.random.randn(latent_dim * n_batch)
        X_gan = X_gan.reshape(n_batch, latent_dim)
        y_gan = np.ones((n_batch, 1))
        dcgan.train_on_batch(X_gan, y_gan)
    return generator, discriminator, dcgan
  
# Define the dataset path and number of images to use
dataset_path = 'path/to/dataset'
n_samples = 10000

# Define the hyperparameters
n_epochs = 100
n_batch = 128
latent_dim = 100

# Create the generator and discriminator models
generator = generator_model()
discriminator = discriminator_model()

# Create the DCGAN model
dcgan = dcgan(generator, discriminator)

# Train the DCGAN on the dataset
generator, discriminator, dcgan = train_dcgan(generator, discriminator, dcgan, dataset_path, n_epochs=n_epochs, n_batch=n_batch, latent_dim=latent_dim)

# Generate some fake images and display them
fake_images = generate_fake_images(generator, latent_dim, 10)
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(fake_images[i])
plt.show()


# Preprocess train data
train_images = []
train_labels = []

for class_dir in os.listdir(train_data_path):
    class_path = os.path.join(train_data_path, class_dir)
    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # resize the image to 224x224
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB format
        train_images.append(image)
        train_labels.append(class_dir)
        
# Preprocess test data
test_images = []
test_labels = []

for image_file in os.listdir(test_data_path):
    image_path = os.path.join(test_data_path, image_file)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # resize the image to 224x224
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB format
    test_images.append(image)
    test_labels.append(class_dir)
    
# Convert the lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Normalize the pixel values of the images
train_images = train_images / 255.0
test_images = test_images / 255.0

from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

# Define the input shape for the model
input_shape = (224, 224, 3)

# Define the backbone network (e.g. ResNet50)
backbone = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')

# Define the region proposal network (RPN)
rpn = Conv2D(256, (3, 3), padding='same', activation='relu', name='rpn_conv')(backbone.output)

# Define the region of interest (ROI) pooling layer
roi_pooling = RoiPoolingConv(7, 1)([rpn, input_rois])

# Define the mask head
mask_fc_1 = Dense(1024, activation='relu', name='mask_fc_1')(roi_pooling)
mask_fc_2 = Dense(1024, activation='relu', name='mask_fc_2')(mask_fc_1)
mask_fc_3 = Dense(1024, activation='relu', name='mask_fc_3')(mask_fc_2)
mask_logits = Dense(num_classes, activation='softmax', name='mask')(mask_fc_3)

# Define the full MASK R-CNN model
model = Model(inputs=[backbone.input, input_rois], outputs=[rpn, mask_logits])

from keras.optimizers import Adam

# Compile the model with loss functions and metrics
model.compile(loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors), mask_loss()],
              optimizer=Adam(lr=1e-5),
              metrics={'rpn_cls': cls_accuracy(num_anchors), 'rpn_regr': rpn_regr_accuracy(num_anchors), 'mask': mask_accuracy()})

# Train the model for a specified number of epochs
num_epochs = 50
batch_size = 8
model.fit(x=[train_images, train_rois], y=[y_rpn_cls, y_rpn_regr, y_mask],
          validation_data=([test_images, test_rois], [y_rpn_cls_test, y_rpn_regr_test, y_mask_test]),
          batch_size=batch_size, epochs=num_epochs)

# Evaluate the model on the test dataset
scores = model.evaluate([test_images, test_rois], [y_rpn_cls_test, y_rpn_regr_test, y_mask_test], verbose=1)
print('Test loss:', scores[0])
print('RPN classification accuracy:', scores[1])
print('RPN regression accuracy:', scores[2])
print('Mask accuracy:', scores[3])

import matplotlib.pyplot as plt
import numpy as np

# Use the trained model to make predictions on new images
test_image_path = 'test_image.jpg'
test_image = preprocess_image(test_image_path)

rpn_cls, rpn_regr, mask = model.predict([test_image, np.expand_dims(anchor_boxes, axis=0)])

# Visualize the predicted bounding boxes and masks on the test image
visualize_predictions(test_image, rpn_cls, rpn_regr, mask, anchor_boxes)

# Compute the precision, recall, and F1 score for the object detection task
y_true = np.array([1, 0, 1, 0, 1])
y_pred = np.array([1, 1, 0, 1, 1])
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1_score)

import seaborn as sns

# Visualize the distribution of object sizes in the dataset
object_sizes = [calculate_object_size(mask) for mask in masks]
sns.histplot(object_sizes, kde=False, bins=50)
plt.title('Distribution of Object Sizes')
plt.xlabel('Object Size (in pixels)')
plt.ylabel('Frequency')
plt.show()

# Plot the precision-recall curve for the object detection task
precision, recall, _ = precision_recall_curve(y_true, y_scores)
plt.plot(recall, precision, color='blue', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Object Detection')
plt.legend()
plt.show()

# Save the predicted bounding boxes and masks for later use
np.savez('predictions.npz', rpn_cls=rpn_cls, rpn_regr=rpn_regr, mask=mask)



