from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Embedding, Concatenate
from keras.layers import MultiHeadAttention, GlobalAveragePooling1D, LayerNormalization

def create_transformer_model(input_shape):
    num_heads = 4
    feed_forward_dim = 128
    dropout_rate = 0.2

    inputs = Input(shape=input_shape)

    # Embedding layer
    embedding_layer = Embedding(input_dim=input_shape[1], output_dim=num_heads)(inputs)

    # Multi-Head Attention layer
    attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1])(embedding_layer, embedding_layer)

    # Skip connection and Layer Normalization
    attention_layer = LayerNormalization(epsilon=1e-6)(inputs + attention_layer)

    # Feed-Forward layer
    feed_forward_layer = Dense(feed_forward_dim, activation="relu")(attention_layer)
    feed_forward_layer = Dropout(dropout_rate)(feed_forward_layer)
    feed_forward_layer = Dense(input_shape[1])(feed_forward_layer)

    # Skip connection and Layer Normalization
    transformer_output = LayerNormalization(epsilon=1e-6)(attention_layer + feed_forward_layer)

    # Global Average Pooling layer
    output = GlobalAveragePooling1D()(transformer_output)

    model = Sequential()
    model.add(inputs)
    model.add(embedding_layer)
    model.add(attention_layer)
    model.add(feed_forward_layer)
    model.add(transformer_output)
    model.add(output)
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mae")
    return model
