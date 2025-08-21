import os
import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # 분산을 일정하게 초기화하기 위한 일종의 플래그(flag)
                                           # 더 좋은 방법이 있을거라 하셨지만, 자기는 모르겠댘ㅋㅋㅋㅋㅋ
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming through
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):   
        # multi-head attention
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query(q), key(k), values(v) for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # (B, nh, T, T)
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        # 근사 버전을 안써도 좋지만, gpt2는 tanh 근사 버전을 쓴다
        self.gelu   = nn.GELU(approximate='tanh') # relu랑 비슷한데 완벽하게 flat하지 않다
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # Feedforward network
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # add & norm이 attn, mlp 앞에 들어간다! 
        # add를 통해 gradient flow를 원활하게 한다
        x = x + self.attn(self.ln_1(x)) # residual! 분산이 증가한다
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass # create classes that primarily store data
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension
    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # transformer decoder of gpt2
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # h(hidden): hidden layer의 weight, bias
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # ln_f: additional final layer norm
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # lm_head: linear model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing scheme
        # 초기화도 두번되지만 문제없다
        self.transformer.wte.weight = self.lm_head.weight # 파라미터 절약 (768 * 50257 = 38M)
        
        # init params
        # nn.Module.apply(fn): 현재 모듈과 그 안에 포함된 모든 하위 모듈(layer) 에 대해 fn(module)을 재귀적으로 호출
        self.apply(self._init_weights) # apply the weight init function to all parameters
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 # 0.02로 설정. 모델 크기에 따라 다르게 설정하는 것이 합리적이지만, GPT-2 논문에서는 0.02로 설정
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # 1/root(N) scale the std by the number of layers
                                                         # MLP의 각 블록마다 2개의 residual connection이 있기 때문에
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # normal distribution
            if module.bias is not None:
                # pytorch에서는 uniform이 기본이기 때문에, bias는 0으로 초기화
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # 논문에서는 0.01이었는데 그냥 0.02로
        
    # STEP2: implement forward pass
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        # T: time dimension
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
                                                                      # idx와 같은 device에 있어야 한다. 불일치가 없도록
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # (B*T, vocab_size), (B*T,)
        return logits, loss
        
    # STEP1: load weights from huggingface - NO HAVE TO READ THIS
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that requre grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused) # AdamW optimizer, hyperparameters: gpt3 paper
        return optimizer
    
# -------------------------------------------------------------------------
import tiktoken # encoding 해주는 library
import numpy as np

def load_tokens(filename):
    npt = np.load(filename) 
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B # B: batch size
        self.T = T # T: sequence length
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank # 각 프로세스의 데이터 시작 위치
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1] 
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes # 다음 배치로 넘어간다
        # if loading the next batch would be out of bounds, advance to next shard
        # 넘어가면 다음 shard에서 시작한다
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel)
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # 프로세스의 번호
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # 한 노드(서버)안에서의 프로세스 번호, 단일 노드라면 신경쓸 필요 없음
    ddp_world_size = int(os.environ['WORLD_SIZE']) # GPU 개수
    device = f'cuda:{ddp_local_rank}' 
    torch.cuda.set_device(device) 
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run, single GPU training
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
    print(f"using device: {device}")

# set the random seed for reproducibility(재현성)
# 초기 weight가 동일하게 초기화되도록 -> 일관되게 재현 가능
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

# data loader (B값은 자기 GPU의 메모리 크기에 따라 조정!)
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 4 # micro batch size
T = 1024 # sequence length, gpt2 maximum sequence length = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process: # master process가 출력을 맡는다 (한번만 출력되어야 하니까)
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# set the matmul precision to high for better performance
torch.set_float32_matmul_precision('high') # "highest": float32, "high": tensorfloat32, "medium": bfloat16

# create the model
model = GPT(GPTConfig(vocab_size=50304)) # from-scratch initialized model = randomly initialized
                                         # 해보면 완전히 별로다 - 완전 랜덤하게 초기화하였기 때문에 (학습x)
                                         # vocab_size=50304: nice number!
model.to(device) # 해당 device로 모델을 옮긴다
model = torch.compile(model) # torch.compile: PyTorch 2.0에서 제공하는 컴파일러로, 모델을 최적화하여 실행 속도를 높인다
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) # wrap the model, 모든 프로세스의 gradient를 동기화하고, 평균을 내준다
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# learning rate scheduler: gpt3 paper
max_lr = 6e-4 
min_lr = max_lr * 0.1
warmup_steps = 715 # first 375M tokens, 375e6 / 2**19
max_steps = 19073 # 10e9 / 2**19
def get_lr(it): 
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learing rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1 # decay_ratio should be in [0, 1]
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device) # 직접 구현한 optimizer

for step in range(max_steps):
    t0 = time.time() # 시작 시간
    
    # once in a while evaluate our validation loss
    if step % 1000 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad(): # no gradient calculation for validation
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device) 
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            
    # once in a while generate from the model (except step 0, which is noise)
    # disabled because torch.complie throws a scary error i can't solve rn
    # if you disable torch.compile, this code works fine
    if step > 0 and step % 100 == 0 and False:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device) # rng: random number generator
        sample_rng.manual_seed(42 + ddp_rank)
        with xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad(): # no backward = gradient tracking 비활성화 -> 메모리 절약
                logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipline default)
                # topk_probs here becomes (5,50), topk_indices is (5,50)
                # top 50만 고려하니까 모델이 탈선할 일도 없다
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, 1, ix) # (B, 1) - 인덱스에 해당하는 token을 가져온다 = 다음 토큰
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist() # tolist: 텐서를 리스트로 변환
            decoded = enc.decode(tokens) # tokens -> text
            print(f"rank {ddp_rank} sample {i}: {decoded}")
     
    # training loop
    model.train()
    optimizer.zero_grad() # 매번 gradient 초기화해준다 = 기울기가 누적되면 안되니까
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() # get the next batch
        x, y = x.to(device), y.to(device) # 너무 많은 메모리를 GPU에 두고 있지 않도록 이때 옮긴다
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # bfloat16: Ampere GPU 이상 에서만 지원
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps # 꼭 나눠줘야 한다!
        loss_accum += loss.detach() # 출력을 위한 loss
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # 마지막 마이크로 스텝에서만 gradient를 동기화한다
        loss.backward()
        
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # 모든 프로세스의 loss를 평균낸다
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gpt3 paper
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    torch.cuda.synchronize() # GPU에서 연산이 끝날 때까지 기다린다
    t1 = time.time() # 끝난 시간
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt # B * T / time
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}") # loss는 텐서이기 때문에 .item()으로 값을 가져온다
                                                                                                                                                         # .item()을 호출하면 GPU에 있던 텐서를 CPU로 옮겨서 숫자로 변환한다
if ddp:
    destroy_process_group() # DDP를 종료한다





