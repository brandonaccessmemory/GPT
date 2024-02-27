import torch
import torch.nn as nn
from cornell_gpt import GPTLanguageModel
from transformers import GPT2Tokenizer
# #mid hyperparameters
# batch_size = 64 # how many independent sequences will we process in parallel?
# block_size = 256 # what is the maximum context length for predictions?
# max_iters = 300
# eval_interval = 50
# learning_rate = 1e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 100
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.4
# personality = 'INFJ'

# #small hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 20 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 8
n_layer = 6
dropout = 0.6
# ------------
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data

    # create a tensor of size batch_size and generate a random index to sample from the data set 
    # question statement is at odd index
    ix = torch.randint(int((len(data) - block_size)/2), (batch_size,)) * 2 -1
    x = torch.stack([data[i][:block_size-1] for i in ix])
    # target sentence 
    y = torch.stack([data[i+1][:block_size-1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# exclude calculation of gradient 
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    print("Hyperparameters for this model: ")
    print(f"Batch Size: {batch_size}")
    print(f"Block Size: {block_size}")
    print(f"Max Iterations: {max_iters}")
    print(f"Evaluation Interval: {eval_interval}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"Evaluation Iterations: {eval_iters}")
    print(f"Number of Embeddings: {n_embd}")
    print(f"Number of Heads: {n_head}")
    print(f"Number of Layers: {n_layer}")
    print(f"Dropout: {dropout} \n")
    
    with open('./datasets/dialogue.txt', 'r', encoding = 'utf-8') as f: 
        text = f.read().split('\n')

    sentence = [] 
    for line in text: 
        sentence.append(line)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = '<PAD>'
    # Train and test splits
    data = tokenizer(sentence, padding=True, truncation=True, max_length=20, return_tensors="pt")
    print("Tokenized Data", data[:5])

    n = int(0.8*len(data['input_ids'])) # first 80% will be train, rest validation
    train_data = data['input_ids'][:n]
    val_data = data['input_ids'][n:]
    vocab_size = tokenizer.vocab_size

    # with open(f'./MBTI/{personality}.txt', 'r', encoding = 'utf-8') as f: 
    #     text = f.read()

    # with open('./datasets/vocab.txt', 'r', encoding = 'utf-8') as f:
    #     vocab = f.read().split('\n')
    #     # 7079
    #     vocab_size = len(vocab)
    #     print(vocab_size)

    # # create a mapping from words to integers
    # word_to_idx = { word:idx for idx,word in enumerate(vocab) }
    # idx_to_word = { idx:word for idx,word in enumerate(vocab) }
    # # 0 acts as the UNKOWN token
    # encode = lambda s: [word_to_idx.get(c, 0) for c in s] # encoder: take a string, output a list of integers
    # decode = lambda l: ' '.join([idx_to_word.get(i, '<UNK>') for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    # data = torch.tensor(encode(text.split()), dtype=torch.long)
    # n = int(0.9*len(data)) # first 90% will be train, rest validation
    # train_data = data[:n]
    # val_data = data[n:]

    # initialize the model 
    model = GPTLanguageModel()
    m = model.to(device)
    m.load_state_dict(torch.load('./models/cornell_gpt.pth'))

    # create a PyTorch optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay =0.01)

    # start training 
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        # update the model
        loss.backward()
        optimizer.step()

    # Save the model
    torch.save(m.state_dict(), './models/tuned_gpt.pth')

    print("\nInference")
    # input = "hello how are you"
    # data = torch.tensor(encode(input.split()), dtype=torch.long, device=device)
    # data = data.reshape(len(input.split()),1)
    # print("Tuned Generated Text: " + decode(m.generate(data,max_new_tokens=100)[0].tolist()))   
    hi = "i am doing great today"
    data_ = tokenizer(hi, padding=True, truncation=True, max_length=20, return_tensors="pt")
    data_ = data_['input_ids'][0].reshape(5,1)
    print(data_)
    # data_ = data_['input_ids'].reshape(len(input.split()),1)
    # print(decode(m.generate(data,max_new_tokens=100)[0].tolist()))
    print(tokenizer.decode(m.generate(data_.to(device),max_new_tokens=10)[0]))


