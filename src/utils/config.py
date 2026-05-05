from dotenv import load_dotenv
import os

load_dotenv()

MOLIT_API_KEY       = os.getenv("MOLIT_API_KEY")
KAPT_API_KEY        = os.getenv("KAPT_API_KEY")
NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
