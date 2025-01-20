import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from aglib import AudioGuru
import os
import requests

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

audio_guru = AudioGuru()

WELCOME_TEXT = (
    "Приветствую тебя, о путник в мире мелодий! Я — премудрый AudioGuru,\n"
    "и сегодня я поделюсь с тобой твоими музыкальными изысканиями.\n"
    "Функционал:\n"
    "- Разметка аудиофайлов форматов mp3 и wav.\n"
    "\nИнструкция:\n"
    "1. Отправьте мне аудиофайл в формате mp3 или wav.\n"
    "2. Я начну обработку и пришлю результат."
)

IMAGE_PATH = "./telegram_attachments/welcome.jpg"

RESPONSE_END = "\nНе удивляйся, если что-то не совпало с твоими ожиданиями — даже премудрого AudioGuru иногда подводит слух!"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет визитную карточку на команду /start с изображением."""
    await update.message.reply_text(WELCOME_TEXT)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=IMAGE_PATH)


async def info_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет документацию для библиотеки."""
    info_text = "Документация:\n" "Lorem ipsum."
    await update.message.reply_text(info_text)


async def github_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет условия пользования и информацию о материалах."""
    license_text = (
        "GitHub:\n"
        "https://github.com/ilyamikhailov16/Audio_Guru.\n\n"
        "Pypip:\n"
        "Lorem ipsum."
    )
    await update.message.reply_text(license_text)


async def credits_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет информацию о команде и контактные данные."""
    credits_text = (
        "Наша команда:\n"
        "- Крылов Илья - Backend-разработчик\n"
        "- Гатин Ленар - Тимлид/ML-инженер\n"
        "- Михайлов Илья - ML-инженер\n"
        "- Барановский Никита - Дизайнер\n\n"
        "Контакты для обратной связи:\n"
        "- Крылов Илья: krylovilyusha@gmail.com\n"
        "- Гатин Ленар: lgatin1711@gmail.com\n"
        "- Михайлов Илья: ilyamikhailov2006@gmail.com\n"
        "- Барановский Никита: baranovckiivl@gmail.com"
    )
    await update.message.reply_text(credits_text)


async def fallback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ответ на любые другие текстовые сообщения."""
    await update.message.reply_text(
        "Извините, я не понимаю это сообщение. Пожалуйста, используйте команду /start."
    )


async def audio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка аудиофайлов .mp3."""
    audio_file = update.message.audio
    if audio_file:
        file_extension = audio_file.file_name.split(".")[-1].lower()
        if file_extension in ["mp3", "wav"]:
            await update.message.reply_text("Идет обработка...")
            logger.info(f"Receiving audio file: {audio_file.file_name}")

            try:
                file_url = (await context.bot.get_file(audio_file.file_id)).file_path

                file_path = os.path.join(os.getcwd(), audio_file.file_name)
                response = requests.get(file_url)

                logger.info(f"Get audio file: {audio_file.file_name}")

                with open(file_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"Add to directory audio file: {audio_file.file_name}")

                mood, genre, tempo = audio_guru(file_path)

                name = file_path.replace("\\", ".")
                name = name.split(".")[-2]

                output_tag_information = (
                    "Ты подал мне аудиотрек, а я, как великий мастер звуковых искусств, разметил его.\n"
                    "\nВот, что получилось:\n\n"
                )

                mood_pattern = ""
                mood_pattern += f"   {mood[0][0]} - {int(mood[0][1]*100)}%\n\n"
                if len(mood) > 1:
                    mood_pattern += f"   {mood[1][0]} - {int(mood[1][1]*100)}%\n\n"
                if len(mood) > 2:
                    mood_pattern += f"   {mood[2][0]} - {int(mood[2][1]*100)}%\n\n"

                genre_pattern = ""
                genre_pattern += f"   {genre[0][0]} - {int(genre[0][1]*100)}%\n\n"
                if len(genre) > 1:
                    genre_pattern += f"   {genre[1][0]} - {int(genre[1][1]*100)}%\n\n"
                if len(genre) > 2:
                    genre_pattern += f"   {genre[2][0]} - {int(genre[2][1]*100)}%\n\n"

                pattern = (
                    f"- Название трека: {name}\n\n"
                    f"- Распознанные настроения:\n\n{mood_pattern}"
                    f"- Распознанные жанры:\n\n{genre_pattern}"
                    f"- Темп: {tempo}\n"
                    # f'- Голос: {voice}\n'
                )

                output_tag_information += f"{pattern}"

                output_tag_information += RESPONSE_END

                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=output_tag_information,
                )

                logger.info(f"Processed audio file: {audio_file.file_name}")

                os.remove(file_path)

                logger.info(f"Delete audio file: {audio_file.file_name}")

            except Exception as e:
                logger.error(f"Error: {e}")
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"Произошла ошибка при обработке файла {audio_file.file_name}.",
                )
        else:
            await update.message.reply_text(
                "Поддерживаются только файлы в формате .mp3."
            )


async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка изображений."""
    await update.message.reply_text(
        "Ошибка: я не обрабатываю изображения. Пожалуйста, отправьте аудиофайл в формате .mp3."
    )


def main() -> None:
    """Запуск бота."""
    application = ApplicationBuilder().token("").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("documentation", info_handler))
    application.add_handler(CommandHandler("about", github_handler))
    application.add_handler(CommandHandler("credits", credits_handler))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, fallback_handler)
    )
    application.add_handler(MessageHandler(filters.AUDIO, audio_handler))
    application.add_handler(MessageHandler(filters.PHOTO, image_handler))

    application.run_polling()


if __name__ == "__main__":
    main()
