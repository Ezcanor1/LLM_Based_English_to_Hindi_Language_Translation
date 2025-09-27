import torch
import os
import urllib.request
torch.manual_seed(123)
import tiktoken
from torch.utils.data import Dataset, DataLoader
import time

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPT_CONFIG_123M={
    "vocab_size":50257,
    "context_length":1024,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False
}

import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert(d_out%num_heads==0),\
        "d_out must be divisible by num_heads."
        self.d_out=d_out
        self.num_heads=num_heads
        self.w_query=torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad=True)
        self.w_key=torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad=True)
        self.w_value=torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad=True)
        self.out_proj=torch.nn.Parameter(torch.rand(d_in,d_out))
        self.dropout=nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length), diagonal=1)
        )
    def forward(self,x):
        b,num_tokens,d_in=x.shape
        keys=x@self.w_key
        queries=x@self.w_query
        values=x@self.w_value
        self.head_dim=self.d_out//self.num_heads
        keys=keys.view(b,num_tokens,self.num_heads,self.head_dim)
        queries=queries.view(b,num_tokens,self.num_heads,self.head_dim)
        values=values.view(b,num_tokens,self.num_heads,self.head_dim)
        
        keys=keys.transpose(1,2)
        queries=queries.transpose(1,2)
        values=values.transpose(1,2)

        attn_scores=queries@keys.transpose(2,3)
        mask_bool=self.mask.bool()[:num_tokens,:num_tokens]
        attn_scores.masked_fill_(mask_bool,-torch.inf)

        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
        attn_weights=self.dropout(attn_weights)
        
        context_vec=(attn_weights@values).transpose(1,2)

        context_vec=context_vec.contiguous().view(b,num_tokens,self.d_out)
        context_vec=context_vec@self.out_proj

        return context_vec
    
class layerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps=1e-5
        self.shift=nn.Parameter(torch.zeros(emb_dim))
        self.scale=nn.Parameter(torch.ones(emb_dim))
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True)
        output=(x-mean)/torch.sqrt(var+self.eps)
        return self.scale*output+self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x,3))))    

class feedforward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"]))
    def forward(self,x):
        return self.layers(x)
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att=MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]            
        )
        self.ff=feedforward(cfg)
        self.norm1=layerNorm(cfg["emb_dim"])
        self.norm2=layerNorm(cfg["emb_dim"])
        self.drop_shortcut=nn.Dropout(cfg["drop_rate"])
    def forward(self,x):
        shortcut=x
        x=self.norm1(x)
        x=self.att(x)
        x=self.drop_shortcut(x)
        x=x+shortcut

        shortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        x=self.drop_shortcut(x)
        x=x+shortcut
        
        return x
        
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = layerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size,seq_len=in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        pos_embeds=self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x=tok_embeds+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits=self.out_head(x)
        return logits



model=GPTModel(GPT_CONFIG_123M)







file_path="the-verdict.txt"
with open(file_path,"r",encoding="Utf-8") as file:
        text_data=file.read()
        
train_rate=0.90
split_idx=int(train_rate*len(text_data))
train_data=text_data[:split_idx]
val_data=text_data[split_idx:]
print("train loader length",len(train_data))
print("val loader length:",len(val_data))




class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids=[]
        self.target_ids=[]

        token_ids=tokenizer.encode(txt,allowed_special={"<endoftext>"})

        for i in range(0, len(token_ids)-max_length,stride):
            input_chunk=token_ids[i:i+max_length]
            target_chunk=token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]
def create_dataloader_v1(txt,batch_size=2,max_length=256,
                         stride=128,shuffle=True,drop_last=True,
                         num_workers=0):
    tokenizer=tiktoken.get_encoding("gpt2")
    dataset=GPTDatasetV1(txt,tokenizer,max_length,stride)
    dataloader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

train_loader=create_dataloader_v1(
  train_data,
  batch_size=2,
  max_length=GPT_CONFIG_123M["context_length"],
  stride=GPT_CONFIG_123M["context_length"],
  drop_last=True,
  shuffle=True,
  num_workers=0
)

val_loader=create_dataloader_v1(
  val_data,
  batch_size=2,
  max_length=GPT_CONFIG_123M["context_length"],
  stride=GPT_CONFIG_123M["context_length"],
  drop_last=False,
  shuffle=False,
  num_workers=0
)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch,target_batch=input_batch.to(device),target_batch.to(device)
    logits=model(input_batch)
    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
    return loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss=0
    if len(data_loader)==0:
        return float("nan")
    elif num_batches is  None:
        num_batches=len(data_loader)
    else:
        num_batches=min(num_batches, len(data_loader))
    for i,(input_batch,target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss=calc_loss_batch(input_batch,target_batch,model,device)
            total_loss+=loss.item()
        else:
            break
    return total_loss/num_batches

def text_to_token_ids(text,tokenizer):
    encoded=tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encoded_tensor=torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor
def token_ids_to_text(token_ids,tokenizer):
    flat=token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits=logits[:,-1,:]
        probas=torch.softmax(logits,dim=-1)
        idx_next=torch.argmax(probas,dim=-1,keepdim=True)
        idx=torch.cat((idx,idx_next),dim=-1)
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size=model.pos_emb.weight.shape[0]
    encoded=text_to_token_ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids=generate_text_simple(
            model=model,idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text=token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n"," "))
    model.train()

def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss=calc_loss_loader(train_loader,model,device,num_batches=eval_iter)
        val_loss=calc_loss_loader(val_loader,model,device,num_batches=eval_iter)
    model.train()
    return train_loss,val_loss



def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses,val_losses,track_token_seen=[],[],[]
    tokens_seen,global_step=0,-1
    for epoch in range(num_epochs):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss=calc_loss_batch(input_batch,target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen+=input_batch.numel()
            global_step+=1
            
            if global_step%eval_freq==0:
                train_loss,val_loss=evaluate_model(
                    model, train_loader,val_loader,device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (step {global_step:06d}): "
                      f"train loss {train_loss:3f}, val loss {val_loss:.3f}")
        generate_and_print_sample(
            model,tokenizer,device,start_context
        )
    return train_losses,val_losses,track_token_seen

tokenizer=tiktoken.get_encoding("gpt2")
start_time=time.time()
torch.manual_seed(123)
model=GPTModel(GPT_CONFIG_123M)
model.to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=0.0004,weight_decay=0.1)

