import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "meta-llama/Llama-2-13b-chat-hf"

tokenizer, model, path = None, None, None

def interact_with_llama(prompt):
    global tokenizer, model, path
    if tokenizer is None:
        print("Model loading started")
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16
        )
        print("Model loading finished")
        pipe = pipeline("text-generation", model=model.to(device), tokenizer=tokenizer, device=device)
    input_str = f'''You are a helpful virtual assistant who gives polite and helpful answers to user question Question: {prompt} Answer:'''
    returned = pipe(input_str)
    return returned.replace(input_str, "")

if __name__ == "__main__":
    interact_with_llama("What is the capital of France?")