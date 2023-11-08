from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", token="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")
print(pipe)