import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def run_model(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 10,
    temperature: float = 0
):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content


'''client = OpenAI(
  api_key = os.getenv("OPENAI_API_KEY")
)

response = client.responses.create(
  model="gpt-4.1-mini",
  input="if you read this, say 'hello world'",
  store=True,
)

print(response.output_text)
'''
