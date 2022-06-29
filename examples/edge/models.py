from tensorflow import keras
from tensorflow.keras import layers

xavier_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05)

# ===
# Dense model
# ===


def Discriminator(dim):
    x_mb = keras.Input(shape=(dim,), name="x_mb")
    m_mb = keras.Input(shape=(dim,), name="m_mb")

    concat = layers.Concatenate(axis=1)([x_mb, m_mb])

    x = layers.Dense(dim, activation='relu', kernel_initializer=xavier_init)(concat)
    x = layers.Dense(dim, activation='relu', kernel_initializer=xavier_init)(x)
    x = layers.Dense(dim, activation='sigmoid', kernel_initializer=xavier_init, name="output")(x)

    return keras.Model(inputs=[x_mb, m_mb], outputs=x)


def Generator_dense(dim):
    x_mb = keras.Input(shape=(dim,), name="x_mb")
    m_mb = keras.Input(shape=(dim,), name="m_mb")

    concat = layers.Concatenate(axis=1)([x_mb, m_mb])

    x = layers.Dense(dim, activation='relu', kernel_initializer=xavier_init)(concat)
    x = layers.Dense(dim, activation='relu', kernel_initializer=xavier_init)(x)
    x = layers.Dense(dim, activation='sigmoid', kernel_initializer=xavier_init, name="output")(x)

    return keras.Model(inputs=[x_mb, m_mb], outputs=x)

# ===
# CNN model
# ===

def Generator(dim):
    x_mb = keras.Input(shape=(dim,), name="x_mb")
    m_mb = keras.Input(shape=(dim,), name="m_mb")

    x_mb = layers.Reshape((dim, -1))(x_mb)
    m_mb = layers.Reshape((dim, -1))(m_mb)

    concat = layers.Concatenate(axis=2)([x_mb, m_mb])

    x = layers.Conv1D(32, 5, activation='relu', kernel_initializer=xavier_init)(concat)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, 10, activation='relu', kernel_initializer=xavier_init)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, 20, activation='relu', kernel_initializer=xavier_init)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, 10, activation='relu', kernel_initializer=xavier_init)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(32, 5, activation='relu', kernel_initializer=xavier_init)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dim, activation='sigmoid', kernel_initializer=xavier_init)(x)

    return keras.Model(inputs=[x_mb, m_mb], outputs=x)


def Generator_map(dim):
    x_mb = keras.Input(shape=(dim,), name="x_mb")
    m_mb = keras.Input(shape=(dim,), name="m_mb")
    map_mb = keras.Input(shape=(dim,), name="map_mb")

    x_mb = layers.Reshape((dim, -1))(x_mb)
    m_mb = layers.Reshape((dim, -1))(m_mb)
    map_mb = layers.Reshape((dim, -1))(map_mb)

    concat = layers.Concatenate(axis=2)([x_mb, m_mb, map_mb])

    x = layers.Conv1D(32, 5, activation='relu', kernel_initializer=xavier_init)(concat)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, 10, activation='relu', kernel_initializer=xavier_init)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, 20, activation='relu', kernel_initializer=xavier_init)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, 10, activation='relu', kernel_initializer=xavier_init)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(32, 5, activation='relu', kernel_initializer=xavier_init)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dim, activation='sigmoid', kernel_initializer=xavier_init)(x)

    return keras.Model(inputs=[x_mb, m_mb, map_mb], outputs=x)