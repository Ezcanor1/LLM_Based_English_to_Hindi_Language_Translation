import pandas as pd
dataset=pd.read_csv("Dataset_English_Hindi.csv")


dataset=dataset[:50000]


print(len(dataset))


train_portion=int(len(dataset)*0.80)
test_portion=int(len(dataset)*0.20)

train_data=dataset[:train_portion]
test_data=dataset[train_portion:]

print(len(train_data))
print(len(test_data))

print(train_data['Hindi'])


print(train_data['English'])


def format_input(entry):
    return f"### Instruction:\nTranslate the given English sentence to Hindi.\n\n### Input:\n{entry['English']}\n\n### Response:\n"

# def format_input(entry):
#     instruction_text = (
#         f"### Instruction:\nTranslate the given English sentence to Hindi."
        
#     )
#     response_tag = "\n\n### Response:\n"  
#     input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
#     return instruction_text + input_text + response_tag

# def format_input(entry):
#     instruction_text = "### Instruction:\nTranslate the given English sentence to Hindi."
#     input_text = f"\n\n### Input:\n{entry['English']}"
#     response_tag = "\n\n### Response:\n"  # Include this if you're going to add target Hindi
#     return instruction_text + input_text + response_tag




import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token  
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)



def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size=model.transformer.wpe.weight.shape[0]
    encoded=text_to_token_ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids=generate_text_simple(
            model=model,idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text=token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n"," "))
    model.train()



def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size=model.transformer.wpe.weight.shape[0]
    encoded=text_to_token_ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids=generate_text_simple(
            model=model,idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text=token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n"," "))
    model.train()


    import torch

device=torch.device("cuda")
model = model.to(device)
print(device)
import torch.nn.functional as F
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch,target_batch=input_batch.to(device),target_batch.to(device)
    logits=model(input_batch).logits
    loss = F.cross_entropy(
    logits.flatten(0,1),
    target_batch.flatten()
)
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
def train_model_simple(model, train_loader, test_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses,test_losses,track_token_seen=[],[],[]
    tokens_seen,global_step=0,0
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss=calc_loss_batch(input_batch,target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen+=input_batch.numel()
            global_step+=1
            
            if global_step%eval_freq==0:
                train_loss,test_loss=evaluate_model(
                    model, train_loader,test_loader,device,eval_iter)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                track_token_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (step {global_step:06d}): "
                      f"train loss {train_loss:3f}, val loss {test_loss:.3f}")
        generate_and_print_sample(
            model,tokenizer,device,start_context
        )
    return train_losses,test_losses,track_token_seen


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from functools import partial
import torch
import torch.nn.functional as F
    

def custom_collate_fn(
    batch,
    pad_token_id=50256,  
    ignore_index=-100,
    allowed_max_length=1024,
    device="cpu"
):
    inputs_lst, targets_lst = [], []

    for item in batch:
        item = item.copy()
        
        input_ids = item[:-1]
        target_ids = item[1:]

        if allowed_max_length:
            input_ids = input_ids[:allowed_max_length]
            target_ids = target_ids[:allowed_max_length]

        inputs_lst.append(torch.tensor(input_ids, dtype=torch.long))
        targets_lst.append(torch.tensor(target_ids, dtype=torch.long))

    max_len = max(seq.size(0) for seq in inputs_lst)
    inputs_padded = [F.pad(seq, (0, max_len - seq.size(0)), value=pad_token_id) for seq in inputs_lst]
    targets_padded = [F.pad(seq, (0, max_len - seq.size(0)), value=ignore_index) for seq in targets_lst]

    inputs_tensor = torch.stack(inputs_padded).to(device)
    targets_tensor = torch.stack(targets_padded).to(device)

    return inputs_tensor, targets_tensor

customized_collate_fn = partial(
    custom_collate_fn, 
    device=device, 
    allowed_max_length=1024,
    pad_token_id=tokenizer.pad_token_id
)



print(train_data['Hindi'])


import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []  
        
        for _,entry in data.iterrows():
            instruction_plus_input = format_input(entry)
            response_text = str(entry['Hindi'])
            full_text = instruction_plus_input + response_text + tokenizer.eos_token
            
            encoded = tokenizer.encode(full_text, add_special_tokens=False)
            self.encoded_texts.append(encoded)
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
    


def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss=calc_loss_loader(train_loader,model,device,num_batches=eval_iter)
        val_loss=calc_loss_loader(val_loader,model,device,num_batches=eval_iter)
    model.train()
    return train_loss,val_loss




print(type(train_data))



from torch.utils.data import DataLoader
num_workers=0
batch_size=2
torch.manual_seed(123)
train_dataset=InstructionDataset(train_data,tokenizer)
train_loader=DataLoader(
    train_dataset,
    batch_size=2,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

# val_dataset=InstructionDataset(val_data,tokenizer)
# val_loader=DataLoader(
#     val_dataset,
#     batch_size=batch_size,
#     collate_fn=customized_collate_fn,
#     shuffle=False,
#     drop_last=False,
#     num_workers=num_workers
# )

test_dataset=InstructionDataset(test_data,tokenizer)
test_loader=DataLoader(
    test_dataset,
    batch_size=2,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)



def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())




num_epochs = 5  
lr = 0.00005

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=0.7):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond).logits
        logits = logits[:, -1, :] / temperature
        probas = torch.softmax(logits, dim=-1)
        
        idx_next = torch.multinomial(probas, num_samples=1)
        
        if idx_next.item() == tokenizer.eos_token_id:
            break
            
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx



import torch
torch.cuda.empty_cache()



optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)




import time


start_time=time.time()
torch.manual_seed(123)
optimizer=torch.optim.AdamW(model.parameters(),lr=0.00005,weight_decay=0.01)
num_epochs=3
train_losses, test_losses,tokens_seen=train_model_simple(
    model, train_loader,test_loader, optimizer, device, num_epochs=num_epochs,eval_freq=100, eval_iter=5,
    start_context=format_input(test_data.iloc[5]), tokenizer=tokenizer
)
end_time=time.time()
execution_time_minutes=(end_time-start_time)/60
print(f"training completed in {execution_time_minutes:.2f} minutes.")



def generate(model, idx, max_new_tokens,context_size,temperature=0.1, top_k=None, eos_id=None):
    device = next(model.parameters()).device
    idx=idx.to(device)
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            output = model(idx_cond)
            logits = output.logits  # extract logits from ModelOutput
        logits = logits[:, -1, :]


        if top_k is not None:
            top_logits,_=torch.topk(logits,top_k)
            min_val=top_logits[:,-1]
            logits=torch.where(logits<min_val,torch.tensor(float("-inf")).to(logits.device),logits)
        if temperature>0.0:
            logits=logits/temperature
            probs=torch.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
        else:
            idx_next=torch.argmax(logits,dim=-1,keepdim=True)
        if idx_next==eos_id:
            break
        idx=torch.cat((idx,idx_next),dim=-1)
    return idx



torch.manual_seed(123)

for _, entry in test_data[:3].iterrows():
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=Base_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['Hindi']}")
    print(f"\nModel response:\n>> {response_text.strip()}")



from langchain_community.textspliter import charectortextsplitter
from langchain_community.vectorspace import chroma
from langchain.document_loaders import TextLoader
from  langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAi


data=TextLoader("sample.txt")

loader=data.load()

text_splitter=charectortextsplitter(chunk=1000)

embeddings=OpenAIEmbeddings(open_ai_key)

vectordb=chroma.from_documents(
    doc=texts,
    embedings=embeddings,
    presist_directory="./chroma_db"
)

llm=ChatOpenAi(temp=0,openai=openai)

qa_chain=RetrivalQA.from_chain_type(
    llm=llm,
    retriver=vectordb.as_retriever(),
    return_scource_documents=True
)

q1="what is the capital of india"
print(qa_chan.invoke({"query":q1}))


from langchain_community.chains import ReteivalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import chroma



api="id","key"

data=charectortextsplitter("sample.txt")

loader=data.load()

textsplit=charectortextsplitter(chunks=1000)

embeddings=OpenAIEmbeddings(ky=key)

vectordb=chroma(
    doc=text,
    emb=emb,
    dirc="./doc"
)

llm=openai(api=api)

qa_chain=RetrievalQA(
    llm=llm,
    retrival=vectordb.as_retriver(),
    return_source_documents=True
)

q1="hello"
print(qa_chain.invoke(q1))
