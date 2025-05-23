# bot.py

import logging
import requests
import os
import tempfile
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Updater, MessageHandler, Filters, CallbackContext,
    CallbackQueryHandler, CommandHandler
)

import os
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = os.getenv("API_URL", "http://ragbot:8000")

logging.basicConfig(level=logging.INFO)

# --- Стартовое приветствие ---
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "👋 Привет!\n"
        "Я RAG-бот. Задай мне вопрос по базе знаний или пришли PDF — и я отвечу максимально по делу!\n\n"
        "— Просто напиши свой вопрос или\n"
        "— Пришли PDF-файл для пополнения базы.\n\n"
        "После ответа можешь посмотреть использованный контекст или оценить качество генерации! 👍👎"
    )

# --- Обработка PDF ---
def handle_document(update: Update, context: CallbackContext):
    tmp_dir = tempfile.gettempdir()
    file_path = os.path.join(tmp_dir, update.message.document.file_name)
    file = update.message.document.get_file()
    file.download(custom_path=file_path)
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{API_URL}/upload_pdf",
            files={"file": (update.message.document.file_name, f, "application/pdf")}
        )
    try:
        status = resp.json().get("status", "error")
    except Exception:
        update.message.reply_text("❗️Ошибка на сервере при загрузке PDF. Проверь, что FastAPI работает.")
        logging.error(f"Ошибка загрузки PDF: {resp.status_code} {resp.text}")
        return
    update.message.reply_text(f"📄 PDF загружен: {status}\nТеперь можешь задать вопрос по этому документу!")

# --- Обработка вопросов ---
def handle_text(update: Update, context: CallbackContext):
    query = update.message.text.strip()
    if query.startswith('/'):
        return  # не отвечаем на команды, кроме /start
    data = {"query": query, "top_k": 2}
    resp = requests.post(f"{API_URL}/ask", data=data)
    if resp.status_code != 200:
        update.message.reply_text("❗️Ошибка на сервере, попробуй позже.")
        return

    result = resp.json()
    context_text = "\n\n".join(result.get("chunks", []))
    answer = result.get("answer", "Нет ответа")

    # Кнопки: посмотреть контекст, оценить ответ 👍👎
    buttons = [
        [InlineKeyboardButton("Показать контекст", callback_data="show_ctx")],
        [
            InlineKeyboardButton("👍 Хороший", callback_data="rate_good"),
            InlineKeyboardButton("👎 Плохой", callback_data="rate_bad")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(buttons)

    # Сохраняем нужные данные для callback
    context.user_data["last_context"] = context_text
    context.user_data["last_answer"] = answer
    context.user_data["last_question"] = query

    update.message.reply_text(
        f"<b>Ответ:</b>\n{answer}",
        parse_mode="HTML",
        reply_markup=reply_markup
    )

# --- Обработка нажатий на кнопки ---
def button_callback(update: Update, context: CallbackContext):
    query_obj = update.callback_query
    query_obj.answer()
    cb_data = query_obj.data

    if cb_data == "show_ctx":
        ctx = context.user_data.get("last_context", "Контекст не сохранён")
        query_obj.message.reply_text(
            f"<b>Контекст ответа:</b>\n{ctx}", parse_mode="HTML"
        )

    elif cb_data == "rate_good":
        question = context.user_data.get("last_question", "")
        answer = context.user_data.get("last_answer", "")
        # Здесь можно добавить сохранение оценки куда хочешь: в файл, БД и т.п.
        query_obj.message.reply_text(
            "Спасибо за оценку! 👍\n\nТвой положительный фидбек сохранён."
        )
        logging.info(f"[GOOD] {question} ||| {answer}")

    elif cb_data == "rate_bad":
        question = context.user_data.get("last_question", "")
        answer = context.user_data.get("last_answer", "")
        query_obj.message.reply_text(
            "Спасибо за честность! 👎\n\nТвой отрицательный фидбек сохранён."
        )
        logging.info(f"[BAD] {question} ||| {answer}")

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    # Хендлеры
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.document.mime_type("application/pdf"), handle_document))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))
    dp.add_handler(CallbackQueryHandler(button_callback, pattern="^(show_ctx|rate_good|rate_bad)$"))

    updater.start_polling()
    logging.info("Бот запущен на polling...")
    updater.idle()

if __name__ == "__main__":
    main()
