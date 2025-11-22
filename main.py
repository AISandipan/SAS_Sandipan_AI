import os
from fastapi import FastAPI, Request
from telegram import Bot, Update
from telegram.ext import Dispatcher, MessageHandler, Filters

from agent import agent_executor
from tools import transcribe_audio
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = Bot(token=TELEGRAM_TOKEN)

app = FastAPI()

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    update = Update.de_json(data, bot)

    if update.message.voice:
        # 1) Get file
        file_id = update.message.voice.file_id
        file = bot.get_file(file_id)
        file.download("audio.ogg")

        # 2) Transcribe
        text = transcribe_audio("audio.ogg")

    else:
        text = update.message.text

    # 3) Send text to the Agent
    result = agent_executor.invoke({"input": text})

    # 4) Reply back
    bot.send_message(chat_id=update.effective_chat.id, text=result["output"])

    return {"status": "ok"}
