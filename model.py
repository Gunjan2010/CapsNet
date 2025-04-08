import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Lambda, Reshape


epsilon = 1e-7
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1. + s_squared_norm) / tf.sqrt(s_squared_norm + epsilon)
    return scale * vectors

class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(
            shape=[1, self.input_num_capsule, self.num_capsule, self.input_dim_capsule, self.dim_capsule],
            initializer='glorot_uniform',
            trainable=True,
            name='W')

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs_expand = tf.expand_dims(inputs, 2)
        inputs_expand = tf.expand_dims(inputs_expand, -1)
        W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1])
        W_tiled = tf.transpose(W_tiled, [0, 1, 2, 4, 3])
        u_hat = tf.matmul(W_tiled, inputs_expand)
        u_hat = tf.squeeze(u_hat, axis=-1)
        b = tf.zeros_like(u_hat[..., 0])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            c = tf.expand_dims(c, -1)
            s = tf.reduce_sum(c * u_hat, axis=5)
            v = squash(s)
            if i < self.routings - 1:
                v_expand = tf.expand_dims(v, 1)
                b += tf.reduce_sum(u_hat * v_expand, axis=-1)
        return v

def PrimaryCaps(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    x = layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size,
                      strides=strides, padding=padding, activation='relu')(inputs)
    x = Reshape(target_shape=[-1, dim_capsule + 2])(x)



train_dir = '/kaggle/input/nail-disease-detection-dataset/data/train'
val_dir = '/kaggle/input/nail-disease-detection-dataset/data/validation'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1.0/255)


labels = sorted(os.listdir(train_dir))
num_classes = len(labels)


def create_caps_model():
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False

    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = PrimaryCaps(x, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='same')
    x = CapsuleLayer(num_capsule=num_classes, dim_capsule=16)(x)
    outputs = Lambda(lambda z: tf.sqrt(tf.reduce_sum(tf.square(z), axis=2)))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0003),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_scratch_caps_model():
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # PrimaryCaps + Capsule Layer
    primary_caps = PrimaryCaps(x, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='same')
    digit_caps = CapsuleLayer(num_capsule=num_classes, dim_capsule=16)(primary_caps)
    output = Lambda(lambda z: tf.sqrt(tf.reduce_sum(tf.square(z), axis=2)))(digit_caps)

    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=0.0003),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


model = create_caps_model()
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator
)


val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")
