import os
import requests
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
res = requests.get('https://api.groq.com/openai/v1/models', headers={'Authorization': f'Bearer {groq_key}'})
print("Groq models:", [m['id'] for m in res.json().get('data', [])])

or_key = os.getenv("OPENROUTER_API_KEY")
res2 = requests.get('https://openrouter.ai/api/v1/models')
print("OpenRouter free models:", [m['id'] for m in res2.json().get('data', []) if "free" in m['id'].lower()])
