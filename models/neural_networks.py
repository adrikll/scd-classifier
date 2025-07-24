from tensorflow import keras
from tensorflow.keras import layers

def create_mlp(optimizer_name='adam', learning_rate=0.001, input_shape=(175,)):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Dense(64, activation='relu'), 
        layers.Dropout(0.3), 
        layers.Dense(32, activation='relu'), 
        layers.Dropout(0.3), 
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = keras.optimizers.get({
        'class_name': optimizer_name,
        'config': {'learning_rate': learning_rate}
    })
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn(optimizer_name='adam', learning_rate=0.001, input_shape=(175,)):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Reshape((input_shape[0], 1)),
        layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'), # Reduzido de 32
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'), # Reduzido de 64
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'), 
        layers.Dropout(0.2), 
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = keras.optimizers.get({
        'class_name': optimizer_name,
        'config': {'learning_rate': learning_rate}
    })
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Resumo do Modelo MLP----------------")
    mlp_model = create_mlp()
    mlp_model.summary()
    print("\n")

    print("Resumo do Modelo CNN----------------")
    cnn_model = create_cnn()
    cnn_model.summary()

