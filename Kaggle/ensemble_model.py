from keras.models import Model
from keras.layers import Input, Concatenate, Dense

def create_ensemble_model(lstm_model, cnn_model, input_shape):
    lstm_input = Input(shape=input_shape)
    cnn_input = Input(shape=input_shape)

    # LSTM model
    lstm_output = lstm_model(lstm_input)

    # CNN model
    cnn_output = cnn_model(cnn_input)

    # Fully connected layer for fusion
    fusion_layer = Concatenate()([lstm_output, cnn_output])
    fusion_layer = Dense(16, activation="relu")(fusion_layer)

    # Output layer
    output_layer = Dense(1, activation="linear")(fusion_layer)

    model = Model(inputs=[lstm_input, cnn_input], outputs=output_layer)
    model.compile(optimizer="adam", loss="mae")
    return model
