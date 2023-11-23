import tensorflow as tf
import os
import numpy as np
from PIL import Image

GLOBAL_PATH = "/content/drive/MyDrive/Colab Notebooks/Type of Clouds/new_clouds/"

CUMULUS_CLOUDS = GLOBAL_PATH + "train/Кучевые облака"
CIRRUS_CUMULUS_CLOUDS = GLOBAL_PATH + "train/Перисто-кучевые облака"
CIRRUS_CLOUDS = GLOBAL_PATH + "train/Перистые облака"
LAYERED_CLOUDS = GLOBAL_PATH + "train/Слоистые облака"
CLEAR_SKY = GLOBAL_PATH + "train/Чистое небо"
CLOUDS_TYPES = {
    "Кучевые облака": np.array([1, 0, 0, 0, 0]),
    "Перисто-кучевые облака": np.array([0, 1, 0, 0, 0]),
    "Перистые облака": np.array([0, 0, 1, 0, 0]),
    "Слоистые облака": np.array([0, 0, 0, 1, 0]),
    "Чистое небо": np.array([0, 0, 0, 0, 1]),
}



IMAGE_PATH = "/content/drive/MyDrive/Colab Notebooks/Type of Clouds/new_clouds/"
CUMULUS_CLOUDS_T = GLOBAL_PATH + "test/Кучевые облака"
CIRRUS_CUMULUS_CLOUDS_T = GLOBAL_PATH + "test/Перисто-кучевые облака"
CIRRUS_CLOUDS_T = GLOBAL_PATH + "test/Перистые облака"
LAYERED_CLOUDS_T = GLOBAL_PATH + "test/Слоистые облака"
CLEAR_SKY_T = GLOBAL_PATH + "test/Чистое небо"

MODEL_PATH = GLOBAL_PATH + "lonet-5_new_clouds_v2.h5"
MODELS_LIST = {
    "lonet-5_new_clouds_v2": "/content/drive/MyDrive/Colab Notebooks/Type of Clouds/new_clouds/lonet-5_new_clouds_v2.h5",
}



def start_train():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(400, 300, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)), # 48x35
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # Компилируем модель
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # Создаём собственный коллбек
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epochs, logs={}):
            if logs.get('val_accuracy') >= 0.84 or logs.get('val_loss') <= 0.4 or logs.get('accuracy') >= 0.9 or logs.get('loss') <= 0.2:
                print("\nДостигнута точность на валидационной выборке в 85%. Прерывание обучения.")
                self.model.stop_training = True

    # Предобработка данных и обучение
    arr_CUMULUS_CLOUDS = prepare_images_for_training(CUMULUS_CLOUDS)
    arr_CIRRUS_CUMULUS_CLOUDS = prepare_images_for_training(CIRRUS_CUMULUS_CLOUDS)
    arr_CIRRUS_CLOUDS = prepare_images_for_training(CIRRUS_CLOUDS)
    arr_LAYERED_CLOUDS = prepare_images_for_training(LAYERED_CLOUDS)
    arr_CLEAR_SKY = prepare_images_for_training(CLEAR_SKY)

    train_x = np.array([*arr_CUMULUS_CLOUDS,
                        *arr_CIRRUS_CUMULUS_CLOUDS,
                        *arr_CIRRUS_CLOUDS,
                        *arr_LAYERED_CLOUDS,
                        *arr_CLEAR_SKY,
                        ])
    train_y = np.array([*np.full(shape=(len(arr_CUMULUS_CLOUDS), 5), fill_value=CLOUDS_TYPES["Кучевые облака"]),
                        *np.full(shape=(len(arr_CIRRUS_CUMULUS_CLOUDS), 5), fill_value=CLOUDS_TYPES["Перисто-кучевые облака"]),
                        *np.full(shape=(len(arr_CIRRUS_CLOUDS), 5), fill_value=CLOUDS_TYPES["Перистые облака"]),
                        *np.full(shape=(len(arr_LAYERED_CLOUDS), 5), fill_value=CLOUDS_TYPES["Слоистые облака"]),
                        *np.full(shape=(len(arr_CLEAR_SKY), 5), fill_value=CLOUDS_TYPES["Чистое небо"])
                        ])
    train_x, train_y = shuffle_arrays(train_x, train_y)
    model.fit(train_x, train_y, epochs=50, callbacks=[CustomCallback()], validation_split=0.2)


    # Сохраняем состояние модели
    model.save(GLOBAL_PATH + 'lonet-5_new_clouds_v2.h5')
    print("Обучение было завершенно успешно!")
    return

def prepare_images_for_training(folder_path):
    image_array = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = image.convert("L")
            image = image.resize((400, 300), resample=Image.BILINEAR)
            image = tf.keras.preprocessing.image.img_to_array(image) / 255
            image_array.append(image.reshape(400, 300, 1))
    return image_array

def shuffle_arrays(train_x, train_y):
    assert len(train_x) == len(train_y), "Массивы должны быть одинакового размера"
    print("!!! train_x и train_y равно: ", len(train_x), len(train_y))

    # Создаем случайный индекс для перемешивания
    indices = np.random.permutation(len(train_x))

    # Перемешиваем оба массива с использованием одного и того же индекса
    return train_x[indices], train_y[indices]



# ------ ------
# Далее идёт часть с проверкой сети
# ------ ------



def start_test():
    valid_x, valid_y = get_valid_data()
    for name, model_path in MODELS_LIST.items():
      print("Model -", name)
      model = tf.keras.models.load_model(model_path)
      model.summary()
      predict_res = model.evaluate(valid_x, valid_y)
      print("Потери:", predict_res[0])
      print("Точность:", predict_res[1])
      print("# --- --- +")
    '''
    model = tf.keras.models.load_model(MODEL_PATH)
    test_image = np.array([[*ppr(IMAGE_PATH)]])
    prediction = model.predict(test_image)
    model.summary()
    print(prediction)
    print(prediction.shape)
    print(prediction[0][0])
    '''
    return

def get_valid_data():
    arr_CUMULUS_CLOUDS = prepare_images_for_training(CUMULUS_CLOUDS)
    arr_CIRRUS_CUMULUS_CLOUDS = prepare_images_for_training(CIRRUS_CUMULUS_CLOUDS)
    arr_CIRRUS_CLOUDS = prepare_images_for_training(CIRRUS_CLOUDS)
    arr_LAYERED_CLOUDS = prepare_images_for_training(LAYERED_CLOUDS)
    arr_CLEAR_SKY = prepare_images_for_training(CLEAR_SKY)

    train_x = np.array([*arr_CUMULUS_CLOUDS,
                        *arr_CIRRUS_CUMULUS_CLOUDS,
                        *arr_CIRRUS_CLOUDS,
                        *arr_LAYERED_CLOUDS,
                        *arr_CLEAR_SKY,
                        ])
    train_y = np.array([*np.full(shape=(len(arr_CUMULUS_CLOUDS), 5), fill_value=CLOUDS_TYPES["Кучевые облака"]),
                        *np.full(shape=(len(arr_CIRRUS_CUMULUS_CLOUDS), 5), fill_value=CLOUDS_TYPES["Перисто-кучевые облака"]),
                        *np.full(shape=(len(arr_CIRRUS_CLOUDS), 5), fill_value=CLOUDS_TYPES["Перистые облака"]),
                        *np.full(shape=(len(arr_LAYERED_CLOUDS), 5), fill_value=CLOUDS_TYPES["Слоистые облака"]),
                        *np.full(shape=(len(arr_CLEAR_SKY), 5), fill_value=CLOUDS_TYPES["Чистое небо"])
                        ])
    return shuffle_arrays(train_x, train_y)

def ppr(image_path):
    image_path = os.path.join(os.getcwd(), image_path)
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.resize((400, 300))  # Resize the image to the desired dimensions
    image = tf.keras.preprocessing.image.img_to_array(image) / 255
    return image.reshape(400, 300, 1)
