import torch 
from cornell_gpt import GPTLanguageModel
from transformers import GPT2Tokenizer
from functions import clean_text, remove_stopwords

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize the model 
model = GPTLanguageModel()
m = model.to(device)
m.load_state_dict(torch.load('./models/edition1.pth'))
# m.load_state_dict(torch.load('./models/tuned_gpt.pth'))
m.eval() 

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({  "pad_token" : "<pad>",
                                "bos_token" : "<sos>",
                                "eos_token" : "<eos>"})
tokenizer.add_tokens(["<context>", "<question>", "<personality>"])

# inference 
print("Inference File")
input = remove_stopwords(clean_text("I used my artistic skills to teach children how to create beautiful and creative works of art. I don't have a specific exercise routine, but I try to get some activity in most days of the week. i m also a fulltime student studying radiology at local college. i ve never been to the beach. I play in a band that my parents don't know about."))
# input = "<personality>enfp<personality><context>" + input + "<context><question>what is your favourite hobby<question>"
input = "<context>" + input + "<context><question>what is the meaning of life<question>"
hi = [] 
hi.append(input)
data_ = tokenizer(hi, padding=True, truncation=True, max_length=128, return_tensors="pt")
# print("Inference Data", data_)

for x in range(5):
    print(tokenizer.decode(m.generate(data_['input_ids'].to(device),max_new_tokens=50)[0]))

