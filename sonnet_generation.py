'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

# Testing JC's edits!!

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model
from modules.attention import LoraLayer

from optimizer import AdamW
import random
from collections import defaultdict

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True



def corrupt_sonnet(lines, strategy):
    """Generate a negative example from a list of sonnet lines."""
    lines = [l for l in lines if l.strip()]  # remove empty lines
    
    if strategy == 'shuffle':
        # shuffle lines but keep couplet somewhat intact
        body = lines[:12]
        couplet = lines[12:14]
        random.shuffle(body)
        return body + couplet
    
    elif strategy == 'truncate':
        # cut to 8-10 lines
        cut = random.randint(8, 10)
        return lines[:cut]
    
    elif strategy == 'repeat':
        # duplicate 2-3 random lines
        result = lines.copy()
        for _ in range(random.randint(2, 3)):
            idx = random.randint(0, len(lines)-1)
            result.insert(idx, lines[idx])
        return result[:14]  # keep same length
    
    elif strategy == 'extra_lines':
        # add corrupted extra lines beyond 14
        extra = random.sample(lines, min(3, len(lines)))
        return lines + extra
    
def build_preference_pairs(sonnets_path):
    pairs = []
    
    with open(sonnets_path, 'r') as f:
        content = f.read()
    
    # parse into individual sonnets
    raw_sonnets = []
    current = []
    for line in content.split('\n'):
        if line.strip().isdigit() and current:
            if len(current) >= 14:
                raw_sonnets.append(current[:14])
            current = []
        elif line.strip() and not line.strip().isdigit():
            current.append(line.strip())
    if len(current) >= 14:
        raw_sonnets.append(current[:14])
    
    print(f"Loaded {len(raw_sonnets)} sonnets for corruption")
    
    strategies = ['shuffle', 'truncate', 'repeat', 'extra_lines']
    for sonnet_lines in raw_sonnets:
        winner = '\n'.join(sonnet_lines)
        strategy = random.choice(strategies)
        corrupted = corrupt_sonnet(sonnet_lines, strategy)
        loser = '\n'.join(corrupted)
        pairs.append({
            'winner': winner,
            'loser': loser,
            'strategy': strategy
        })
    
    print(f"Built {len(pairs)} preference pairs")
    return pairs

class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    if hasattr(args, 'use_lora') and args.use_lora:
        # freeze all base weights
        for param in self.gpt.parameters():
            param.requires_grad = False
        # apply LoRA to all layers
        for layer in self.gpt.gpt_layers:
            attn = layer.self_attention
            attn.query = LoraLayer(attn.query, rank=args.lora_rank, alpha=args.lora_alpha)
            attn.value = LoraLayer(attn.value, rank=args.lora_rank, alpha=args.lora_alpha)
        # print trainable params
        total = sum(p.numel() for p in self.gpt.parameters())
        trainable = sum(p.numel() for p in self.gpt.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    else:
        for param in self.gpt.parameters():
            param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    ### YOUR CODE HERE
    outputs = self.gpt(input_ids, attention_mask)
    return self.gpt.hidden_state_to_token(outputs['last_hidden_state']) # everything


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.85, top_p=0.9, max_length=300):
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    for _ in range(max_length):
        logits_sequence = self.forward(token_ids, attention_mask)
        logits_last_token = logits_sequence[:, -1, :] / temperature

        probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= top_p
        top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
        top_p_mask[..., 0] = True
        filtered_probs = sorted_probs * top_p_mask
        filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

        sampled_index = torch.multinomial(filtered_probs, 1)
        sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

        if sampled_token.item() == self.tokenizer.eos_token_id:
            break

        token_ids = torch.cat([token_ids, sampled_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
        )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

def compute_log_prob(model, input_ids, attention_mask):
    """Compute sum of log probs for the sequence."""
    logits = model(input_ids, attention_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[:, :-1].gather(
        dim=-1,
        index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * attention_mask[:, 1:].float()).sum(dim=-1)


def simpo_loss(model, winner_ids, winner_mask, loser_ids, loser_mask, beta=2.0, gamma=0.5):
    """SimPO loss - no reference model needed."""
    logp_w = compute_log_prob(model, winner_ids, winner_mask)
    logp_l = compute_log_prob(model, loser_ids, loser_mask)
    
    len_w = winner_mask.sum(dim=-1).float()
    len_l = loser_mask.sum(dim=-1).float()
    
    # length normalized rewards
    reward_w = (beta / len_w) * logp_w
    reward_l = (beta / len_l) * logp_l
    
    loss = -F.logsigmoid(reward_w - reward_l - gamma).mean()
    return loss

def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  # NEW: load checkpoint if provided
  start_epoch = 0
  if args.checkpoint is not None:
      saved = torch.load(args.checkpoint, weights_only=False)
      model.load_state_dict(saved['model'])
      optimizer.load_state_dict(saved['optim'])
      start_epoch = int(args.checkpoint.split('_')[0]) + 1
      del saved  # free memory immediately
      torch.cuda.empty_cache()

  # change range to start from start_epoch
  for epoch in range(start_epoch, args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask = batch['token_ids'], batch['attention_mask']
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(logits, labels, reduction='mean')
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    if epoch >= 24 and epoch % 5 == 0 or epoch == args.epochs - 1:
        save_model(model, optimizer, args, f'{epoch}_{args.filepath}')

    #print('Generating several output sonnets...')
    #model.eval()
    #for batch in held_out_sonnet_dataset:
        #encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
        #output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
        #print(f'{batch[1]}{output[1]}\n\n')
def train_simpo(args, pairs):
    """Fine-tune with SimPO on top of best checkpoint using LoRA."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # load best checkpoint
    saved = torch.load('25_gpt2-large-50-1e-05-sonnet.pt', weights_only=False)
    model = SonnetGPT(saved['args'])
    model.load_state_dict(saved['model'])
    
    # freeze base, add LoRA for SimPO fine-tuning
    for param in model.gpt.parameters():
        param.requires_grad = False
    for layer in model.gpt.gpt_layers:
        attn = layer.self_attention
        attn.query = LoraLayer(attn.query, rank=16, alpha=32)
        attn.value = LoraLayer(attn.value, rank=16, alpha=32)
    
    model = model.to(device)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.simpo_lr
    )
    
    print(f"Training SimPO on {len(pairs)} preference pairs")
    
    for epoch in range(args.simpo_epochs):
        model.train()
        random.shuffle(pairs)
        total_loss = 0
        num_batches = 0
        
        for pair in pairs:
            # tokenize winner and loser
            winner_enc = model.tokenizer(
                pair['winner'], 
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)
            
            loser_enc = model.tokenizer(
                pair['loser'],
                return_tensors='pt', 
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)
            
            optimizer.zero_grad()
            loss = simpo_loss(
                model,
                winner_enc['input_ids'], winner_enc['attention_mask'],
                loser_enc['input_ids'], loser_enc['attention_mask']
            )
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"SimPO Epoch {epoch}: loss {avg_loss:.3f}")
        # replace the save line
        if (epoch + 1) % 5 == 0 and epoch > 0:
            save_model(model, optimizer, args, f'{epoch}_simpo_{args.filepath}')

@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
    parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_size", type=str,
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
    parser.add_argument("--checkpoint", type=str, default=None)

    # LoRA parameters
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # SimPO

    parser.add_argument("--simpo_epochs", type=int, default=5)
    parser.add_argument("--run_simpo", action='store_true')
    parser.add_argument("--simpo_lr", type=float, default=1e-5)

    args = parser.parse_args()
    return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args

if __name__ == "__main__":
    args = get_args()
    if args.use_lora:
        args.filepath = f'{args.model_size}-lora{args.lora_rank}-{args.epochs}-{args.lr}-sonnet.pt'
    elif args.run_simpo:
        args.filepath = f'{args.model_size}-simpo-{args.simpo_epochs}-{args.simpo_lr}-sonnet.pt'
    else:
        args.filepath = f'{args.model_size}-{args.epochs}-{args.lr}-sonnet.pt'
    
    seed_everything(args.seed)
    
    if args.run_simpo:
        pairs = build_preference_pairs(args.sonnet_path)
        train_simpo(args, pairs)
    else:
        train(args)
        generate_submission_sonnets(args)