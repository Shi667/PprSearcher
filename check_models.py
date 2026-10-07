from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

print("Modèles disponibles sur Groq :\n")
for m in client.models.list().data:
    print(f"{m.id} - actif: {m.active}")