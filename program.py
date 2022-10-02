### Transformer Based Text Classification
import numpy, tensorflow
maxlen = 200 #number of first words taken from each review
embed_dim = 32 # layer size Embedding for every word
transformer_layers = 1 # number of transformer blocks

## Downloading and preparing a vectorized dataset
word_index = tensorflow.keras.datasets.imdb.get_word_index( path="imdb_word_index.json")
vocab_size = len(word_index) # dictionary size
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = tensorflow.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tensorflow.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

## Creating a text classifier model
inputs = tensorflow.keras.layers.Input(shape=(maxlen,))
# layer Embedding and positional encryption
positions = tensorflow.keras.backend.arange(start=0, stop= maxlen, step=1)
token_embeddings = tensorflow.keras.layers.Embedding( input_dim=vocab_size, output_dim=embed_dim)(inputs)
position_embeddings = tensorflow.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)(positions)
embedding_layer = tensorflow.keras.layers.Add()([token_embeddings, position_embeddings[numpy.newaxis,::]]) 
# transformer block set
for _ in range(transformer_layers):
    # multihead attention, addition and normalization
    attn_output = tensorflow.keras.layers.MultiHeadAttention(num_heads=2, key_dim=embed_dim)(embedding_layer, embedding_layer)
    merge = tensorflow.keras.layers.Add()([attn_output, embedding_layer])
    out1 = tensorflow.keras.layers.LayerNormalization(epsilon=1e-6) (merge)
    # two-layer perceptron, addition and normalization
    x = tensorflow.keras.layers.Dense(units=embed_dim, activation="relu")(out1)
    ffn_output = tensorflow.keras.layers.Dense(units=embed_dim)(x)
    merge = tensorflow.keras.layers.Add()([ffn_output, out1])
    embedding_layer = tensorflow.keras.layers.LayerNormalization(epsilon=1e-6) (merge)
# global pooling layer
x = tensorflow.keras.layers.GlobalAveragePooling1D()(embedding_layer)
# double layer perceptron
x = tensorflow.keras.layers.Dense(units=embed_dim, activation="relu")(x)
outputs = tensorflow.keras.layers.Dense(units=1, activation="sigmoid")(x)
transformer = tensorflow.keras.Model(inputs=inputs, outputs=outputs)
transformer.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
transformer.summary()

## Model training and evaluation
transformer.fit(x=x_train, y=y_train, batch_size=64, epochs=10, validation_split=0.2, verbose=1, shuffle=True, initial_epoch=0, steps_per_epoch=None, validation_batch_size=64, validation_freq=1)
test_scores = transformer.evaluate(x=x_test, y=y_test, batch_size=64, verbose=1, return_dict=False)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

## Classification by model
y_model = transformer.predict(x=x_test, batch_size=64, verbose=0)
print("y_test[0:10]:", y_test[0:10])
print("y_model[0:10]:", numpy.argmax(y_model[0:10], axis=1))
