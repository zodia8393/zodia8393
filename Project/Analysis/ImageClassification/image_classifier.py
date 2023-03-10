import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model


def load_model(model_name, num_classes):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def classify_images(model, image_dir, class_labels):
    predictions = []
    if os.path.exists(image_dir):
        for image_file in os.listdir(image_dir):
            try:
                img_path = os.path.join(image_dir, image_file)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
                img_batch = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_batch)
                prediction_index = np.argmax(prediction)
                prediction_label = class_labels[prediction_index]
                predictions.append((image_file, prediction_label))
            except Exception as e:
                print(f'Failed to classify {img_path}. Reason: {e}')
                continue
    else:
        print(f'{image_dir} does not exist')
    return predictions


def delete_trained_data(data_dir):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    else:
        print(f'{data_dir} does not exist')
