# Load model directly
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# ** unpacks the dictionary
print(tokenizer.decode(model.generate(**tokenizer("hey I was good at basketball but ", return_tensors="pt"))[0]))