from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import os
from dotenv import load_dotenv
import logging
import sys

load_dotenv()

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

app.logger.setLevel(logging.INFO)
if not app.logger.handlers:
    app.logger.addHandler(logging.StreamHandler(sys.stdout))


# @app.route("/test-logger", methods=["POST"])
# def testLogger():
#     app.logger.info(f"Request body: {request.get_data(as_text=True)}")

#     return "OK"


@app.route("/webhook", methods=["POST"])
def webhook():
    # Get X-Line-Signature header
    signature = request.headers.get("X-Line-Signature")

    # Get request body
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message_text = event.message.text
    source_type = event.source.type
    if source_type == "user":
        reply_text = f"你說的是：{event.message.text}"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
    elif source_type == "group":
        bot_name = "@linebot name"
        if bot_name in message_text and "@all" not in message_text:
            reply_text = f"你提到我了！你說的是：{message_text.replace(bot_name, '').strip()}"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
