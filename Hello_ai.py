from dotenv import load_dotenv
from google import genai

load_dotenv()  # Reads GEMINI_API_KEY from your .env file
client = genai.Client()  # Automatically picks up GEMINI_API_KEY

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Give me a 3-sentence pep talk for Day 1 of becoming an AI Engineer.",
)

print(response.text)