import numpy as np
import asyncio
import logging
import os
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram import Bot, Dispatcher, types

load_dotenv()
MODEL_PATH = "lonet-5_new_clouds_v2.h5"
TG_IMAGE_PATH = "images_tg"
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

logging.basicConfig(level=logging.INFO)
bot = Bot(token=os.getenv("API_KEY"))
dp = Dispatcher()



@dp.message()
async def handle_photo(message: types.Message):
    await bot.download(
        message.photo[-1],
        destination=f"{TG_IMAGE_PATH}/{message.photo[-1].file_id}.jpg"
    )

    image = ppr(cut_square(f"{TG_IMAGE_PATH}/{message.photo[-1].file_id}.jpg", message.photo[-1].file_id, 400, 300))
    prediction = model.predict(image)
    print(prediction)
    index = np.argmax(prediction)
    pred_val = prediction[0][index]
    match index:
        case 0:
            await message.reply(f"На изображении - Кучевые облака\nВероятность: {round(pred_val * 100)}%")
        case 1:
            await message.reply(f"На изображении - Перисто-кучевые облака\nВероятность: {round(pred_val * 100)}%")
        case 2:
            await message.reply(f"На изображении - Перистые облака\nВероятность: {round(pred_val * 100)}%")
        case 3:
            await message.reply(f"На изображении - Слоистые облака\nВероятность: {round(pred_val * 100)}%")
        case 4:
            await message.reply(f"На изображении - Чистое небо\nВероятность: {round(pred_val * 100)}%")
        case _:
            await message.reply("Что-то пошло не так :(")
    os.remove(f"{TG_IMAGE_PATH}/{message.photo[-1].file_id}.jpg")
    os.remove(f"{TG_IMAGE_PATH}/square_{message.photo[-1].file_id}.jpg")

def ppr(image_path):
    image_path = os.path.join(os.getcwd(), image_path)
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.resize((400, 300))  # Resize the image to the desired dimensions
    image = tf.keras.preprocessing.image.img_to_array(image) / 255
    return np.array([[*image.reshape(400, 300, 1)]])

def cut_square(image_path, file_id, width_x, height_y):
    # Открываем изображение
    image = Image.open(image_path)

    # Получаем размеры изображения
    width, height = image.size

    # Вычисляем координаты для вырезания квадрата
    left = (width - width_x) // 2
    top = (height - height_y) // 2
    right = left + width_x
    bottom = top + height_y

    # Вырезаем квадрат из центра изображения
    square = image.crop((left, top, right, bottom))

    # Генерируем относительный путь для сохранения квадрата
    square_path = f"{TG_IMAGE_PATH}/square_{file_id}.jpg"

    # Сохраняем квадрат
    square.save(square_path)

    # Возвращаем относительный путь к квадрату
    return square_path

# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Hello!")

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())