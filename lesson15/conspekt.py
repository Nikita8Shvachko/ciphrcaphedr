# # Нейронные сети
# - сверточные (конволюционные - CNN - Convolutional Neural Network) нейроныные сети - компьютерное зрение классификация изображений
# - рекуррентные (RNN - Recurrent Neural Network) нейроныные сети - обработка текста, естественный язык
# - генеративные (GAN - Generative Adversarial Network) нейроныные сети - генерация изображений, текста
# - многослойные перцептроны (MLP - Multi-Layer Perceptron) нейроныные сети - классификация, регрессия


import math as m

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    decode_predictions,
    preprocess_input,
)
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

TRAIN_DATA_DIR = ".../dataset/train"  # Contains cats and dogs subdirectories
VALIDATION_DATA_DIR = "/lesson15/dataset/train"  # Same for validation
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 100
NUM_CLASSES = 2
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64

train_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
)
val_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=1234,
    class_mode="categorical",
)
val_generator = val_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode="categorical",
)

from tensorflow.keras.models import Model


def model_maker():
    base_model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    for layer in base_model.layers:
        layer.trainable = False

    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation="relu")(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    custom_model = Dense(NUM_CLASSES, activation="softmax")(custom_model)
    return Model(inputs=input, outputs=custom_model)


model = model_maker()
model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["acc"],
)


num_steps = m.ceil(TRAIN_SAMPLES / BATCH_SIZE)

model.fit(
    train_generator,
    steps_per_epoch=num_steps,
    epochs=10,
    validation_data=val_generator,
    validation_steps=num_steps,
)

print(val_generator.class_indices)
model.save("./lesson19/model.h5")

# img_path = "./lesson19/dataset/train/dog.1.jpg"
# img = image.load_img(img_path, target_size=(224, 224))
# img_array = image.img_to_array(img)
# img_array = preprocess_input(img_array)
# img_array = np.expand_dims(img_array, axis=0)
# model = ResNet50()
# prediction = model.predict(img_array)
# print(decode_predictions(prediction, top=3)[0])
# plt.imshow(img)
# plt.show()
