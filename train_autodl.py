import torch
from tqdm import tqdm
import torch.nn as nn
import math
import warnings
from pathlib import Path
from torch.utils.data import Dataset,DataLoader,random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit, Split, Sequence
from tokenizers.decoders import Fuse
from torch.utils.tensorboard import SummaryWriter
class InputEmbeddings(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self,d_model,seq_len,dropout):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        pe=torch.zeros(seq_len,d_model)

        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps:float=10**-6):
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model)
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model,h,dropout):

        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h==0, "d_model is not divisible by h"

        self.d_k=d_model//h
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k=query.shape[-1]
        attention_scores=(query@key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores=attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores=dropout(attention_scores)
        return (attention_scores@value),attention_scores
    def forward(self,q,k,v,mask):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)

        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)


        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        return self.w_o(x)


class REsidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout):

        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connection=nn.ModuleList([REsidualConnection(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x= self.residual_connection[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connection[1](x,self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)



class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout):

        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connection=nn.ModuleList([REsidualConnection(dropout)for _ in range(3)])

    def forward(self,x ,encoder_output,src_mask,tgt_mask):
        x=self.residual_connection[0](x,lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connection[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connection[2](x,self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)

        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model,vocab_size):
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)
    


class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projection_layer

    def encode(self,src,src_mask):
        src=self.src_embed(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt=self.tgt_embed(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size,tgt_vocab_size,src_seq_len,tgt_seq_len,d_model:int=512,N:int=6,h:int=8,dropout:float=0.1,d_ff:int=2048):
    src_embed=InputEmbeddings(d_model,src_vocab_size)
    tgt_embed=InputEmbeddings(d_model,tgt_vocab_size)

    src_pos=PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos=PositionalEncoding(d_model,tgt_seq_len,dropout)

    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block=DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))

    projection_layer=ProjectionLayer(d_model,tgt_vocab_size)

    transformer=Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return transformer

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len):

        super().__init__()
        self.seq_len=seq_len
        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        self.sos_token=torch.tensor([tokenizer_tgt.token_to_id('[SOS]')],dtype=torch.int64)
        self.eos_token=torch.tensor([tokenizer_tgt.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad_token=torch.tensor([tokenizer_tgt.token_to_id('[PAD]')],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair=self.ds[index]
        src_text=src_target_pair['translation'][self.src_lang]
        tgt_text=src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens=self.tokenizer_src.encode(src_text).ids
        dec_input_tokens=self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens)-2
        dec_num_padding_tokens= self.seq_len - len(dec_input_tokens)-1

        if enc_num_padding_tokens<0 or dec_num_padding_tokens<0:
            raise ValueError('Sentence is too long')

        encoder_input=torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens,dtype=torch.int64)

            ]
        )

        decoder_input=torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
            ]
        )

        label=torch.cat(
            [
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
            ]
        )
        assert encoder_input.size(0)==self.seq_len
        assert decoder_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len


        return {
            "encoder_input":encoder_input,
            "decoder_input":decoder_input,
            "encoder_mask":(encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask":(decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int()& causal_mask(decoder_input.size(0)),
            "label":label,
            "src_text":src_text,
            "tgt_text":tgt_text

        }

def causal_mask(size):
    mask=torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask==0
    

def get_config():
    return{
        "batch_size":8,
        "num_epochs":20,
        "lr":10**-4,
        "seq_len":350,
        "d_model":512,
        "lang_src":"en",
        "lang_tgt":"zh",
        "data_dir":"starter_data/opus20k",
        "model_folder":"weights_opus100_en_zh_char_v1",
        "model_basename":"tmodel_",
        "preload":None,
        "tokenizer_file":"tokenizer_opus100_char_v1_{0}.json",
        "experiment_name":"runs/opus100_en_zh_char_v1"
    }
def get_weights_file_path(config,epoch):
    model_folder=config['model_folder']
    model_basename=config['model_basename']
    model_filename=f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path =Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
        if lang=='zh':
            tokenizer.pre_tokenizer=Sequence([WhitespaceSplit(),Split('',behavior='isolated')])
            tokenizer.decoder=Fuse()
        else:
            tokenizer.pre_tokenizer=Whitespace()
        trainer=WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=1 if lang=='zh' else 2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer_path.parent.mkdir(parents=True,exist_ok=True)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def load_parallel_data(config):
    data_dir=Path(config['data_dir']).expanduser()
    if not data_dir.is_absolute():
        candidates=[root/data_dir for root in (Path.cwd(),*Path.cwd().parents)]
        candidates.append(Path.cwd()/'english-chinese-transformer'/data_dir)
        data_dir=next((path for path in candidates if (path/'train.jsonl').is_file()),data_dir)
    data_files={split:str(data_dir/f'{split}.jsonl') for split in ('train','validation','test')}
    for path in data_files.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f'Data file not found: {path}')
    ds_raw=load_dataset('json',data_files=data_files)
    return ds_raw.map(
        lambda item:{'translation':{config['lang_src']:item[config['lang_src']],config['lang_tgt']:item[config['lang_tgt']]}},
        remove_columns=ds_raw['train'].column_names
    )

def get_ds(config):
    ds_raw=load_parallel_data(config)
    train_ds_raw=ds_raw['train']
    val_ds_raw=ds_raw['validation']

    tokenizer_src=get_or_build_tokenizer(config,train_ds_raw,config['lang_src'])
    tokenizer_tgt=get_or_build_tokenizer(config,train_ds_raw,config['lang_tgt'])

    max_len_src=0
    max_len_tgt=0

    for item in train_ds_raw:
        src_ids=tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids=tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src=max(max_len_src,len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))

    print(f'Max length of source sentence:{max_len_src}')
    print(f'Max length of target sentence:{max_len_tgt}')

    def within_length(item):
        src_len=len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids)
        tgt_len=len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids)
        return src_len+2<=config['seq_len'] and tgt_len+1<=config['seq_len']

    train_count=len(train_ds_raw)
    val_count=len(val_ds_raw)
    train_ds_raw=train_ds_raw.filter(within_length)
    val_ds_raw=val_ds_raw.filter(within_length)
    print(f'Train sentences: {train_count} -> {len(train_ds_raw)}')
    print(f'Validation sentences: {val_count} -> {len(val_ds_raw)}')

    train_ds=BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds=BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])

    train_dataloader=DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_dataloader=DataLoader(val_ds,batch_size=1,shuffle=True)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt
def get_model(config,vocab_src_len,vocab_tgt_len):
    model=build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model

def train_model(config):
    device=torch.device('cuda' if torch.cuda.is_available()else 'cpu')
    print(f'using device{device}')

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)
    model=get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    writer=SummaryWriter(config['experiment_name'])

    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)

    initial_epoch=0
    global_step=0

    if config['preload']:
        model_filename=get_weights_file_path(config,config['preload']) 
        print(f'Preloading model{model_filename}')
        state=torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch=state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step=state['global_step']

    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()
        batch_iterator=tqdm(train_dataloader,desc=f'Processing epoch{epoch:02d}')
        for batch in batch_iterator:
            encoder_input=batch['encoder_input'].to(device)
            decoder_input=batch['decoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)
            decoder_mask=batch['decoder_mask'].to(device)

            encoder_output=model.encode(encoder_input,encoder_mask)
            decoder_output=model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            proj_output=model.project(decoder_output)
            label=batch['label'].to(device)

            loss=loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))

            batch_iterator.set_postfix({f"loss":f"{loss.item():6.3f}"})

            writer.add_scalar('train loss',loss.item(),global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step+=1

        model_filename=get_weights_file_path(config,f'{epoch:02d}')
        torch.save(
            {
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'global_step':global_step
            },model_filename
        )

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    config=get_config()
    train_model(config)

import torchmetrics

def greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt,max_len,device):
    sos_idx=tokenizer_tgt.token_to_id('[SOS]')
    eos_idx=tokenizer_tgt.token_to_id('[EOS]')
    encoder_output=model.encode(source,source_mask)
    decoder_input=torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1)>=max_len:
            break
        decoder_mask=causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out=model.decode(encoder_output,source_mask,decoder_input,decoder_mask)
        prob=model.project(out[:,-1])
        _,next_word=torch.max(prob,dim=1)
        decoder_input=torch.cat([decoder_input,torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)],dim=1)
        if next_word.item()==eos_idx:
            break
    return decoder_input.squeeze(0)

def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,max_len,device,print_msg=print,global_step=0,writer=None,num_examples=2):
    model.eval()
    expected=[]
    predicted=[]
    results=[]
    with torch.no_grad():
        for batch in validation_ds:
            encoder_input=batch['encoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)
            assert encoder_input.size(0)==1,'Batch size must be 1 for validation'
            model_out=greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device)
            source_text=batch['src_text'][0]
            target_text=batch['tgt_text'][0]
            model_out_text=tokenizer_tgt.decode(model_out.detach().cpu().tolist(),skip_special_tokens=True)
            expected.append(target_text)
            predicted.append(model_out_text)
            results.append({'source':source_text,'target':target_text,'predicted':model_out_text})
            print_msg('-'*80)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')
            if len(results)>=num_examples:
                break
    if writer is not None and results:
        expected_chars=[''.join(text.split()) for text in expected]
        predicted_chars=[''.join(text.split()) for text in predicted]
        cer=torchmetrics.CharErrorRate()(predicted_chars,expected_chars)
        bleu=torchmetrics.BLEUScore()([' '.join(text) for text in predicted_chars],[[' '.join(text)] for text in expected_chars])
        writer.add_scalar('validation cer',cer,global_step)
        writer.add_scalar('validation char bleu',bleu,global_step)
        writer.flush()
    return results

def load_translation_model(config,epoch='latest'):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer_src=Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt=Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))
    if epoch=='latest':
        weights_files=sorted(Path(config['model_folder']).glob(f"{config['model_basename']}*.pt"))
        if not weights_files:
            raise FileNotFoundError('No model checkpoint found')
        model_filename=str(weights_files[-1])
    else:
        model_filename=get_weights_file_path(config,epoch)
        if not Path(model_filename).is_file():
            raise FileNotFoundError(f'Model checkpoint not found: {model_filename}')
    model=get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    state=torch.load(model_filename,map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    print(f'Loaded model {model_filename}, epoch {state["epoch"]+1}, device {device}')
    return model,tokenizer_src,tokenizer_tgt,device

def translate(sentence,model,tokenizer_src,tokenizer_tgt,config,device,max_len=None):
    sentence=sentence.strip()
    if not sentence:
        raise ValueError('Source sentence is empty')
    source_tokens=tokenizer_src.encode(sentence).ids
    seq_len=config['seq_len']
    if len(source_tokens)+2>seq_len:
        raise ValueError(f'Source sentence is too long: {len(source_tokens)} tokens, maximum {seq_len-2}')
    max_len=seq_len if max_len is None else max_len
    if not 2<=max_len<=seq_len:
        raise ValueError(f'max_len must be between 2 and {seq_len}')
    sos_idx=tokenizer_src.token_to_id('[SOS]')
    eos_idx=tokenizer_src.token_to_id('[EOS]')
    pad_idx=tokenizer_src.token_to_id('[PAD]')
    source=torch.tensor([sos_idx]+source_tokens+[eos_idx]+[pad_idx]*(seq_len-len(source_tokens)-2),dtype=torch.int64,device=device).unsqueeze(0)
    source_mask=(source!=pad_idx).unsqueeze(1).unsqueeze(1).int()
    model.eval()
    with torch.no_grad():
        model_out=greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt,max_len,device)
    return tokenizer_tgt.decode(model_out.detach().cpu().tolist(),skip_special_tokens=True)

config=get_config()
model,tokenizer_src,tokenizer_tgt,device=load_translation_model(config,epoch=f"{config['num_epochs']-1:02d}")

train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)
validation_results=run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,config['seq_len'],device,num_examples=3)

sentences=["I love you.","Thank you.","Where are you?"]
for sentence in sentences:
    translation=translate(sentence,model,tokenizer_src,tokenizer_tgt,config,device,max_len=min(100,config['seq_len']))
    print(f'SOURCE: {sentence}')
    print(f'PREDICTED: {translation}')
