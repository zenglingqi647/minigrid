# This script asks Mistral-7B-v0.1 to evaluate the reward

import requests
import re

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def truncate_string(string):
    # Find all occurrences of "USER:"
    user_indices = [match.start() for match in re.finditer(r"USER:", string)]

    # If there are at least two occurrences
    if len(user_indices) >= 2:
        # Find the index of the second occurrence of "USER:"
        second_user_index = user_indices[1]

        # Truncate the string after the second "USER:" occurrence
        truncated_string = string[:second_user_index]

        return truncated_string
    else:
        return string  # No second "USER:" found, return the original string
	
def start_repeating(string):
     # Find all occurrences of "USER:"
    user_indices = [match.start() for match in re.finditer(r"USER:", string)]
    return len(user_indices) >= 2
     
input_str = '''
A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {input}
ASSISTANT:
'''

input_str = input_str.format(input='''You are an agent in a Minigrid environment in reinforcement learning. The task of the agent is to open door in a maze. Available actions may include: explore, go to <object>, pick up <object>, toggle <object>. Please infer action for Q. For each state, provide a reward from -1 to 1 based on how helpful the state is to reach the reward. Examples:
                             
Q: [observed blue door, carry blue key]
Value: 0.5
''')

output = query({
	"inputs": input_str,
	"options" : {
		"wait_for_model" : True
    },
	"parameters": {
		"max_new_tokens" : 256
    }
})
output_str = output[0]['generated_text']

# Continue to query until we start repeating.
while not start_repeating(output_str):
    output_str = query({
        "inputs": output_str,
        "options" : {
            "wait_for_model" : True
        },
        "parameters": {
            "max_new_tokens" : 256
        }
    })[0]['generated_text']

print(truncate_string(output_str))