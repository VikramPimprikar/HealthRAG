# ============================================================
# LLM EXPLAINER
# ============================================================

import os

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENV VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ============================================================
# GENERATE EXPLANATION
# ============================================================

def generate_explanation(
    prediction,
    severity,
    context,
    patient_data
):

    try:

        prompt = f"""
You are a medical AI assistant.

Patient Data:
{patient_data}

Prediction:
{severity}

Medical Context:
{context}

Generate a short human-readable medical explanation.
"""

        response = client.chat.completions.create(

            model="gpt-4o-mini",

            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical AI assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,
            max_tokens=200
        )

        return response.choices[0].message.content

    except Exception as e:

        print(f"LLM Error: {e}")

        return f"""
Prediction: {severity}

The patient may require additional cardiovascular evaluation
based on the supplied clinical parameters.
"""