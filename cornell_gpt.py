import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from torch.nn import functional as F

# # huge hyperparameters ( ran out of memory )
# batch_size = 128 # how many independent sequences will we process in parallel?
# block_size = 512 # what is the maximum context length for predictions?
# max_iters = 8000
# eval_interval = 500
# learning_rate = 1e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 1024
# n_head = 16
# n_layer = 12
# dropout = 0.3
# # ------------

#mid hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 12
n_layer = 12
dropout = 0.6
# ------------

# #small hyperparameters
# batch_size = 32 # how many independent sequences will we process in parallel?
# block_size = 128 # what is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 8
# n_layer = 8
# dropout = 0.5
# #------------

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({  "pad_token" : "<pad>",
                                "bos_token" : "<sos>",
                                "eos_token" : "<eos>"})
tokenizer.add_tokens(["<context>", "<question>", "<personality>"])
vocab_size = len(tokenizer)

# # data loading
# def get_batch(split):
#     # generate a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data

#     # create a tensor of size batch_size and generate a random index to sample from the data set 
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data

    # create a tensor of size batch_size and generate a random index to sample from the data set 
    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i][:block_size-1] for i in ix])
    y = torch.stack([data[i][1:block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# exclude calculation of gradient 
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    # calculate loss for training set and validation set
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # forward pass
            logits, loss = model(X, Y)
            # get lossess for each iteration
            losses[k] = loss.item()
        # average loss for eval_iters iteration
        out[split] = losses.mean()
    model.train()
    return out

# positional encoding will be 256 (block size) , 384 (no of embeddings for a token)
# input vector to multi head attention is 256,384 , split into Q(Query) K(Key) V(Value)
# Q * K^T , result in a 256 x 256 vector, this lets us learn the the importance of each character to each other character
# normalise the matrix , then apply softmax so that each value is between 0 to 1 , multiply it with Value and this is the attention
# each head have access to a portion of the characters , if 4 heads then each head is 256,384/4
# multi head attention = head * weight ( gives us importance of each character with every other character )

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # torch.tril - triangular matrix 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # to avoid overfitting 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        # T = token , C = embedding size
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        weight = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weight = F.softmax(weight, dim=-1) # (B, T, T)
        weight = self.dropout(weight)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = weight @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) 
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # +1 due to accomodate UNK tokens
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    # X,Y
    def forward(self, idx, targets=None):
        # 32, 19
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # calculate loss using cross_entropy function
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            idx_cond = idx_cond.to(device)
        
            # get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution, output is different each time 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

            if tokenizer.decode(idx_next[0]) == tokenizer.eos_token:
                print("Finish generating sentence")
                break

        return idx

# model = GPTLanguageModel()
# m = model.to(device)
# # print the number of parameters in the model
# print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# # create a PyTorch optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# for iter in range(max_iters):

#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0 or iter == max_iters - 1:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#     # sample a batch of data
#     xb, yb = get_batch('train')

#     # evaluate the loss
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     # update the model
#     loss.backward()
#     optimizer.step()

# # Save the model
# torch.save(m.state_dict(), 'cornell_gpt.pth')
 
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

    with open('./datasets/persona_chat.txt', 'r', encoding = 'utf-8') as f: 
        text = f.read().split('\n')

    sentence = [] 
    for line in text: 
        sentence.append(line)

    # Train and test splits
    data = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt")
    n = int(0.8*len(data['input_ids'])) # first 80% will be train, rest validation
    train_data = data['input_ids'][:n]
    val_data = data['input_ids'][n:]
    print("Vocab Size", vocab_size)


    model = GPTLanguageModel()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # start training 
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data, yb is targets
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        # update the model
        loss.backward()
        optimizer.step()

    # Save the model
    torch.save(m.state_dict(), './models/edition2.pth')

    # print("\nInference")
    # input = clean_text("I used my artistic skills to teach children how to create beautiful and creative works of art. I don't have a specific exercise routine, but I try to get some activity in most days of the week. i m also a fulltime student studying radiology at local college. i ve never been to the beach. I play in a band that my parents don't know about.")
    # input = "<personality>enfp<personality><context>" + input + "<context><question>how are you doing today<question>"
    # hi = [] 
    # hi.append(input)
    # data_ = tokenizer(hi, padding=True, truncation=True, max_length=128, return_tensors="pt")
    # print("Inference Data", data_)
    # # data_ = data_['input_ids'].reshape(len(input.split()),1)
    # # print(decode(m.generate(data,max_new_tokens=100)[0].tolist()))
    # print(tokenizer.decode(m.generate(data_['input_ids'].to(device),max_new_tokens=30)[0]))

# Load the model
# model = YourTransformerModelClass()
# model.load_state_dict(torch.load('transformer_chatbot_model.pth'))
# model.eval()  # Set the model to evaluation mode
# nltk, spacy for tokenizers 
# online training, ongoing training datasets for LLM 
# personality filter at the end of the output of the chatbot, tune hyperparameters more or more layers 
# <personality> <personality><context> <context><question> <question><sos> <eos> 
