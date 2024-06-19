# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Token_Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.emb_table = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.emb_table(x)



class FixedAbsolutePositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    # given token embedding, add positional embedding
    def forward(self, x):
        batch_size, block_size, d_model = x.shape 
        num_of_token = block_size
        token_dim = d_model

        pos_emb = torch.ones((num_of_token, token_dim), dtype=torch.float)
        for i in range(num_of_token):
            for j in range(token_dim):
                if j % 2 == 0:
                    temp = torch.tensor(i / (10000**(j / token_dim)))
                    pos_emb[i, j] = torch.sin(temp)
                else:
                    temp = torch.tensor(i / (10000**( (j - 1) / token_dim)))
                    pos_emb[i, j] = torch.cos(temp)

        return x + pos_emb


class LearnableAbsolutePositionalEmbedding(nn.Module):
    def __init__(self, block_size: int, d_model: int):
        super().__init__()

        self.pos_emb = nn.Embedding(block_size, d_model)

        self.block_indices = torch.tensor([i for i in range(block_size)]).to(device)

    # given token embedding, add positional embedding
    def forward(self, x):
        return x + self.pos_emb(self.block_indices)



class SingleHead(nn.Module):
    def __init__(self, d_model: int, head_dim: int, needMask: bool, attentionMode: list):
        super().__init__()
        self.key_converter = nn.Linear(d_model, head_dim, bias=False)
        self.value_converter = nn.Linear(d_model, head_dim, bias=False)
        self.query_converter = nn.Linear(d_model, head_dim, bias=False)
        self.d_model = d_model
        self.needMask = needMask
        self.atten_mat = None

        assert attentionMode[0] in ['full', 'atrous', 'local', 'sparse']
        self.attentionMode = attentionMode[0]

        self.k = attentionMode[1]

    def atrousAttention(self, query, key):
        batch_size, block_size, head_dim = query.shape
        logits = torch.full((batch_size, block_size, block_size), float('-inf')).to(device)
        
        for idx in range(block_size):

            # get indices
            firsthalf = list(range(idx, 0, - self.k - 1))
            firsthalf.reverse()
            firsthalf = firsthalf[0:-1]
            secondhalf = list(range(idx, block_size, self.k + 1))
            indices = firsthalf + secondhalf

            # calculate logits
            q = query[:, idx, :]
            k = key[:, indices, :]
            res = torch.matmul(k, q.view(batch_size, head_dim, 1))
            res = res.transpose(-2, -1).squeeze(dim = 1)
            logits[:, idx, indices] = res 

        return logits / math.sqrt(self.d_model)

    def localAttention(self, query, key):
        batch_size, block_size, head_dim = query.shape
        logits = torch.full((batch_size, block_size, block_size), float('-inf')).to(device)
        
        for idx in range(block_size):

            # get indices
            start = max([0, idx - 2])
            end = min([idx + 3, block_size])
            indices = list(range(start, end))

            # calculate logits
            q = query[:, idx, :]
            k = key[:, indices, :]
            res = torch.matmul(k, q.view(batch_size, head_dim, 1))
            res = res.transpose(-2, -1).squeeze(dim = 1)
            logits[:, idx, indices] = res 

        return logits / math.sqrt(self.d_model)

    def sparseAttention(self, query, key):
        batch_size, block_size, head_dim = query.shape
        logits = torch.full((batch_size, block_size, block_size), float('-inf')).to(device)

        for idx in range(block_size):

            # get indices
            start = max([0, idx - 2])
            end = min([idx + 3, block_size])
            indices1 = list(range(start, end))

            # get indices
            firsthalf = list(range(idx, 0, - self.k - 1))
            firsthalf.reverse()
            firsthalf = firsthalf[0:-1]
            secondhalf = list(range(idx, block_size, self.k + 1))
            indices2 = firsthalf + secondhalf

            set1 = set(indices1) ; set2 = set(indices2)
            union_set = set1.union(set2)
            indices = list(union_set)

            # calculate logits
            q = query[:, idx, :]
            k = key[:, indices, :]
            res = torch.matmul(k, q.view(batch_size, head_dim, 1))
            res = res.transpose(-2, -1).squeeze(dim = 1)
            logits[:, idx, indices] = res 

        return logits / math.sqrt(self.d_model)

    def fullAttention(self, query, key):
        return torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)

    def forward(self, x):
        key = self.key_converter(x)
        value = self.value_converter(x)
        query = self.query_converter(x)

        # compute mask
        if self.needMask:
            _, block_size, _ = key.shape
            mask = torch.tril(torch.ones(block_size, block_size))
            mask = mask.to(device)
            mask = mask[:block_size, :block_size] == 0

        # compute logit scores
        if self.attentionMode == 'full':
            logit_scores = self.fullAttention(query=query, key=key)
        elif self.attentionMode == 'atrous':
            logit_scores = self.atrousAttention(query=query, key=key)
        elif self.attentionMode == 'local':
            logit_scores = self.localAttention(query=query, key=key)
        elif self.attentionMode == 'sparse':
            logit_scores = self.sparseAttention(query=query, key=key)


        # compute softmax score
        if self.needMask:
            masked_logit_socres = logit_scores.masked_fill(mask, float('-inf'))
            softmax_scores = masked_logit_socres.softmax(dim=-1)
        else:
            softmax_scores = logit_scores.softmax(dim=-1)  
        
        # store attention matrix
        self.atten_mat = softmax_scores
        
        # compute output
        out = torch.matmul(softmax_scores, value)

        return out
    
    def getAttenMap(self):
        assert self.atten_mat is not None
        return self.atten_mat



class MultiHead(nn.Module):
    def __init__(self, d_model: int, number_of_heads: int, dropout_p: float, needMask: bool, attentionMode: str):
        super().__init__()
        assert d_model % number_of_heads == 0
        single_head_size = int(d_model / number_of_heads)
        self.heads = nn.ModuleList([SingleHead(d_model=d_model, 
                                               head_dim=single_head_size, 
                                               needMask=needMask, 
                                               attentionMode=attentionMode) for _ in range(number_of_heads)])

        self.drop_layer = nn.Dropout(dropout_p)

        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x):
        out_list = [] 
        for h in self.heads:
            out_list.append(h(x))
        out = torch.cat(out_list, dim=-1)
        out = self.linear_layer(out)
        out = self.drop_layer(out)
        return out


class SingleLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, ff_hidden_size: int, dropout_p: float, needMask: bool, preLN: bool, attentionMode: str):
        super().__init__()

        # create multi head attention
        self.multiheads = MultiHead(d_model=d_model , number_of_heads=n_heads, 
                                    dropout_p=dropout_p, needMask=needMask, 
                                    attentionMode=attentionMode)

        # layer norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # feedforward
        self.ff = nn.Sequential(nn.Linear(d_model, ff_hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(ff_hidden_size, d_model), 
                                 nn.Dropout(dropout_p))

        self.preLN = preLN

    def forward(self, x):
        
        # this is from the youtube video
        if self.preLN:
            x = x + self.multiheads(self.ln1(x))
            x = x + self.ff(self.ln2(x))
        # this is from the original paper
        else:
            x = self.ln1(x + self.multiheads(x))
            x = self.ln2(x + self.ff(x))

        return x


class TransformerLayers(nn.Module):

    def __init__(self, d_model: int, n_heads: int, ff_hidden_size: int, 
                 n_single_blocks: int, dropout_p: int,
                 token_embedder: nn.Module, positional_embedder: nn.Module, needMask: bool, 
                 preLN: bool, attentionMode: str):
        super().__init__()
        
        self.token_embedder = token_embedder
        self.positional_embedder = positional_embedder

        self.net = nn.Sequential()
        self.atten_mat_lists = []

        for _ in range(n_single_blocks):
            self.net.append( SingleLayer(d_model=d_model, n_heads=n_heads, 
                                         ff_hidden_size=ff_hidden_size, 
                                         dropout_p=dropout_p, needMask=needMask, 
                                         preLN=preLN, attentionMode=attentionMode) )

    def forward(self, x):
        
        # convert to token embedding
        x = self.token_embedder(x)

        # add positional encoding
        x = self.positional_embedder(x)

        # forward 
        out = self.net(x)

        # extract attention maps
        for single_encoder in self.net:
            for singlehead in single_encoder.multiheads.heads:
                self.atten_mat_lists.append(singlehead.getAttenMap())

        return out , self.atten_mat_lists 
    
    def cleanAttenMaps(self):
        self.atten_mat_lists = []



class Classifier(nn.Module):
    def __init__(self, transformer_blocks: nn.Module, output_head: nn.Module):
        super().__init__()
        
        # transformer blocks
        self.transformer_blocks = transformer_blocks
        
        # output head
        self.output_head = output_head
        

    def forward(self, x):
        
        # forward
        x, _ = self.transformer_blocks(x)
        self.transformer_blocks.cleanAttenMaps()
        x = torch.mean(x, dim=1)
        x = self.output_head(x)

        return x



class LMGenerator(nn.Module):
    def __init__(self, transformer_blocks: nn.Module, output_head: nn.Module, loss_fn: nn.modules.loss):
        super().__init__()
        
        # transformer blocks
        self.transformer_blocks = transformer_blocks
        
        # output head
        self.output_head = output_head

        # define loss function
        self.loss_fn = loss_fn

    def forward(self, x, y):

        # forward
        x, _ = self.transformer_blocks(x)
        self.transformer_blocks.cleanAttenMaps()
        x = self.output_head(x)

        # check shapes
        xbatch_size, xblock_size, vocab_size = x.shape
        ybatch_size, yblock_size = y.shape
        assert xbatch_size == ybatch_size
        assert xblock_size == yblock_size

        # reshape
        x = x.view(xbatch_size*xblock_size, vocab_size)
        y = y.view(ybatch_size*yblock_size)

        # calculate loss
        loss = self.loss_fn(x, y)

        return loss