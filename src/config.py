from dotenv import load_dotenv
import os 
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
NEWS_API_KEY=os.getenv("NEWS_API_KEY")
SECRET_KEY=os.getenv("SECRET_KEY")
HF_TOKEN=os.getenv("HF_TOKEN")
CURRENTS_API_KEY=os.getenv("CURRENTS_API_KEY")
NEWSDATA_API_KEY=os.getenv("NEWSDATA_API_KEY")