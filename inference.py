import torch 
from cornell_gpt import GPTLanguageModel
from transformers import GPT2Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# with open('./datasets/vocab.txt', 'r', encoding = 'utf-8') as f:
#     vocab = f.read().split('\n')
#     # 4888
#     vocab_size = len(vocab)
#     print(vocab_size)

# # create a mapping from characters to integers
# word_to_idx = { word:idx for idx,word in enumerate(vocab) }
# idx_to_word = { idx:word for idx,word in enumerate(vocab) }
# # 0 acts as the UNKOWN token
# encode = lambda s: [word_to_idx.get(c, 0) for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ' '.join([idx_to_word.get(i, '<UNK>') for i in l]) # decoder: take a list of integers, output a string

# initialize the model 
model = GPTLanguageModel()
m = model.to(device)
# m.load_state_dict(torch.load('./models/cornell_gpt.pth'))
m.load_state_dict(torch.load('./models/tuned_gpt.pth'))
m.eval() 

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = '<PAD>'

# inference 
print("Inference File")
hi = [] 
input = 'how is the weather today'
hi.append(input)
max_length = 20
# data_ = tokenizer(hi, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
data_ = tokenizer(hi, return_tensors="pt")
print("Tokenized Data:", data_)
data_ = data_['input_ids'][0].reshape(5,1)
# print(decode(m.generate(data,max_new_tokens=100)[0].tolist()))
print("Data size:", data_.size())
print("Data reshaped:", data_)
print("Decoded Data:", tokenizer.decode(data_[0].tolist()))

for i in range(4):
    x = m.generate(data_.to(device),max_new_tokens=10)
    print("Inference First Generated Text:" + tokenizer.decode( x[0], skip_special_tokens=True))
    print("Inference Second Generated Text:" + tokenizer.decode( x[1], skip_special_tokens=True))
    print("Inference Third Generated Text:" + tokenizer.decode( x[2], skip_special_tokens=True))
    print("Inference Fourth Generated Text:" + tokenizer.decode( x[3], skip_special_tokens=True))
    print("Inference Fifth Generated Text:" + tokenizer.decode( x[4], skip_special_tokens=True))


# data = torch.tensor(encode(input.split()), dtype=torch.long, device = device)
# data = data.reshape(len(input.split()),1)
# print("Inference Generated Text: " + decode(m.generate(data,max_new_tokens=100)[0].tolist()))