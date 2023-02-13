#CNN Model For Image Classification

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load the image data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define the model
def create_model(optimizer, learning_rate, dropout_rate):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the grid search parameters
param_grid = {
    'optimizer': [tf.keras.optimizers.Adam, tf.keras.optimizers.SGD],
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout_rate': [0.0, 0.5, 0.8]
}

# Create the grid search object
grid = GridSearchCV(keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0),
                    param_grid=param_grid,
                    cv=3)

# Run the grid search
grid.fit(x_train, y_train)

# Print the best parameters
print("Best parameters found: ", grid.best_params_)

# Evaluate the best model on the test data
best_model = grid.best_estimator_
y_pred = best_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print("Accuracy on test data: ", accuracy_score(y_test, y_pred))
