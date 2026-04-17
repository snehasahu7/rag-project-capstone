# from groq import Groq
# from app.core.config import settings

# client = Groq(api_key=settings.GROQ_API_KEY)


# def generate_answer(prompt: str) -> str:
#     response = client.chat.completions.create(
#         model=settings.GROQ_MODEL,
#         messages=[
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a strict document QA assistant.\n"
#                     "Only answer using provided context.\n"
#                     "Do NOT add assumptions.\n"
#                     "Do NOT summarize beyond context.\n"
#                     "If answer not found, say: Not found in document."
#                 ),
#             },
#             {"role": "user", "content": prompt},
#         ],
#         temperature=0.0,   # 🔥 IMPORTANT (removes hallucination)
#     )

#     return response.choices[0].message.content.strip()