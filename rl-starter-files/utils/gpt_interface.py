# This script asks GPT-3.5-Turbo for reward of a state.

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_reward_from_gpt(prompt):
    print("Asking GPT about it...")
    output = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a helpful virtual assistant who follows the user's instructions."},
        {"role": "user", "content": prompt}
      ]
    )
    print("GPT responsed...")
    return output.choices[0].message['content']



