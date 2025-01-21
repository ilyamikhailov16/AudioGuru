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

WELCOME_IMAGE_PATH = "./telegram_attachments/welcome.jpg"

RESPONSE_END = "\nНе удивляйся, если что-то не совпало с твоими ожиданиями — даже премудрого AudioGuru иногда подводит слух!"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет визитную карточку на команду /start с изображением."""
    await update.message.reply_text(WELCOME_TEXT)
    await context.bot.send_photo(
        chat_id=update.effective_chat.id, photo=WELCOME_IMAGE_PATH
    )


async def info_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет документацию, условия пользования и информацию о материалах."""
    documentation_license_text = (
        "Документация проекта:\n"
        "https://ganjamember.github.io/audio-guru-documentation/\n\n"
        "GitHub(вся информация в README):\n"
        "https://github.com/ilyamikhailov16/Audio_Guru."
    )

    await update.message.reply_text(documentation_license_text)


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

                mood, genre, tempo = audio_guru(file_path, mode_tag=False)

                name = file_path.replace("\\", ".")
                name = name.split(".")[-2]

                output_tag_information = "Вот, что получилось:\n\n"

                mood_pattern = ""
                genre_pattern = ""

                for i in range(len(mood)):
                    mood_pattern += f"   {mood[i][0]} - {int(mood[i][1]*100)}%\n\n"

                for i in range(len(genre)):
                    genre_pattern += f"   {genre[i][0]} - {int(genre[i][1]*100)}%\n\n"

                pattern = (
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
    application = (
        ApplicationBuilder()
        .token("")
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("about", info_handler))
    application.add_handler(CommandHandler("credits", credits_handler))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, fallback_handler)
    )
    application.add_handler(MessageHandler(filters.AUDIO, audio_handler))
    application.add_handler(MessageHandler(filters.PHOTO, image_handler))

    application.run_polling()


if __name__ == "__main__":
    main()
