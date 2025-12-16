import os
from dotenv import load_dotenv

# מנסה לטעון את הקובץ
loaded = load_dotenv()

print(f"Did .env file load? {loaded}")

llmod_key = os.getenv("LLMOD_API_KEY")
pine_key = os.getenv("PINECONE_API_KEY")

# מדפיס רק את 5 התווים הראשונים ואת האורך, לא את המפתח המלא (לביטחון)
if llmod_key:
    print(f"LLMOD Key found: {llmod_key[:5]}... (Length: {len(llmod_key)})")
else:
    print("ERROR: LLMOD_API_KEY is missing or None!")

if pine_key:
    print(f"Pinecone Key found: {pine_key[:5]}... (Length: {len(pine_key)})")
else:
    print("ERROR: PINECONE_API_KEY is missing or None!")