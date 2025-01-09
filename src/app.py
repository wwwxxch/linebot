import os
import sys
from dotenv import load_dotenv
import logging
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain.chains import RetrievalQA

import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage


load_dotenv()

app = Flask(__name__)


# LINE configuration
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# LLM configuration
class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


# Setup logging
app.logger.setLevel(logging.INFO)
if not app.logger.handlers:
    app.logger.addHandler(logging.StreamHandler(sys.stdout))


# LLM configuration
def llmConfig():
    try:
        llmProvider = os.getenv("LLM_PROVIDER", LLMProvider.GEMINI.value)

        if llmProvider == LLMProvider.OPENAI.value:
            return ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="..."
            )
        elif llmProvider == LLMProvider.GEMINI.value:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
    except Exception as e:
        logging.error(f"llmConfig Error: {e}")
        return None


# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize vector store
db_directory = os.path.join(os.getcwd(), "cat_care_db")
# vector_store = Chroma(persist_directory=db_directory, embeddin_function=embeddings)

try:
    vector_store = Chroma(persist_directory=db_directory, embedding_function=embeddings)
except Exception as e:
    logging.exception(f"Failed to initialize vector database: {e}")
    sys.exit(1)


def getQAChain():
    workflow = StateGraph(state_schema=MessagesState)
    model = llmConfig()

    def callModel(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    workflow.add_edge(START, "model")
    workflow.add_node("model", callModel)
    memory = MemorySaver()

    langGraphApp = workflow.compile(checkpointer=memory)
    return langGraphApp


def generate_system_prompt():
    return """你是一個專業的貓咪照護顧問，請根據用戶的提問提供專業的建議。
    回答時請注意：
    1. 使用友善且專業的口吻
    2. 回答要簡潔明瞭
    3. 如果問題涉及貓咪健康，建議諮詢獸醫
    4. 回答要以繁體中文回覆
    """


def getResFromAI(user_id, user_message):
    """Generate response using RAG system"""
    try:
        langGraphApp = getQAChain()
        thread_id = uuid.uuid4()

        config = {"configurable": {"thread_id": thread_id}}
        inputMessage = HumanMessage(content=user_message)
        full_prompt = f"{generate_system_prompt()}\n\n用戶問題：{inputMessage}"

        for event in langGraphApp.stream({"messages": [full_prompt]}, config, stream_mode="values"):
            response = event["messages"][-1].content
        return response

    except Exception as e:
        app.logger.error(f"Error generating response: {str(e)}")
        return "抱歉，我現在無法正確回答您的問題。請稍後再試。"


# ================================================================================================
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
        app.logger.error("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    app.logger.info(f"event: {event}")
    # responseMessage = f"你說的是：{event.message.text}"
    sourceType = event.source.type
    userId = event.source.user_id

    # Get mention flag
    mentionees = event.message.mention.mentionees if event.message.mention else []
    mention = any(getattr(mentionee, "is_self", False) for mentionee in mentionees)
    if sourceType == "user" or (sourceType == "group" and mention):
        responseMessage = getResFromAI(userId, event.message.text)
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token, messages=[TextMessage(text=responseMessage)]
                )
            )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
