import openai
import os

client = openai.OpenAI(
  base_url="https://mistral-7b.lepton.run/api/v1/",
  api_key="BTmCPY2Xbr1vZ9jhRAqafqzLLjR3KzTL"
)
response = client.chat.completions.create(
  model="mistral-7b",
  messages=[{"role": "user", "content": "Hello, world!"}],
  max_tokens=128,
  stream=True
)

for chunk in response:
  print(chunk.choices[0].delta.content)