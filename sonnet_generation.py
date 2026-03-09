'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

# Testing JC's edits!!

import argparse
import os
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from modules.attention import LoraLayer

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

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


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # Freeze base weights, apply LoRA to all layers
    for param in self.gpt.parameters():
      param.requires_grad = True

    # for layer in self.gpt.gpt_layers:
    #   attn = layer.self_attention
    #   attn.query = LoraLayer(attn.query, alpha=16, rank=16)
    #   attn.value = LoraLayer(attn.value, alpha=16, rank=16)


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
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output



def get_completion_logprobs(model, input_ids, attention_mask, completion_mask):
  """Sum of log probs over completion tokens only.

  completion_mask: (B, T-1) float tensor, 1 for completion token positions.
  """
  logits = model(input_ids, attention_mask)
  log_probs = F.log_softmax(logits, dim=-1)

  # token_log_probs[t] = log p(input_ids[t+1] | input_ids[:t+1])
  token_log_probs = log_probs[:, :-1, :].gather(
      dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
  ).squeeze(-1)

  return (token_log_probs * completion_mask * attention_mask[:, 1:].float()).sum(dim=-1)


def make_completion_mask(input_ids, prompt_lens, device):
  """Build a (B, T-1) mask that is 1 only for completion token positions."""
  B, T = input_ids.shape
  # token_log_probs has shape (B, T-1); position t predicts token t+1
  # completion starts at prompt_len in input_ids -> index prompt_len-1 in token_log_probs
  mask = torch.zeros(B, T - 1, device=device)
  for i, pl in enumerate(prompt_lens):
    mask[i, pl - 1:] = 1.0
  return mask


def train_dpo(args):
  """DPO fine-tuning using training sonnets as chosen and model generations as rejected."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # Load the best SFT model as both policy and frozen reference
  saved = torch.load('train_best.pt', weights_only=False)

  policy_model = SonnetGPT(saved['args'])
  policy_model.load_state_dict(saved['model'])
  policy_model = policy_model.to(device)

  ref_model = SonnetGPT(saved['args'])
  ref_model.load_state_dict(saved['model'])
  ref_model = ref_model.to(device)
  ref_model.eval()
  for param in ref_model.parameters():
    param.requires_grad = False

  tokenizer = policy_model.tokenizer
  sonnet_dataset = SonnetsDataset(args.sonnet_path)

  # Pre-generate all rejected samples offline before DPO training
  print("Generating rejected samples from training sonnets...")
  preference_pairs = []
  policy_model.eval()
  with torch.no_grad():
    for _, sonnet_text in tqdm(sonnet_dataset, desc='generating rejected'):
      lines = [l for l in sonnet_text.strip().split('\n') if l.strip()]
      if len(lines) < 4:
        continue
      prompt_text = '\n'.join(lines[:3]) + '\n'
      win_text = '\n'.join(lines) + '\n'

      enc = tokenizer(prompt_text, return_tensors='pt').to(device)
      rejected_ids, _ = policy_model.generate(enc['input_ids'], temperature=args.temperature, top_p=args.top_p)
      rejected_text = tokenizer.decode(rejected_ids[0].cpu())

      preference_pairs.append((prompt_text, win_text, rejected_text))

  print(f"Generated {len(preference_pairs)} preference pairs.")
  torch.cuda.empty_cache()

  # DPO training loop
  optimizer = AdamW(policy_model.parameters(), lr=args.lr * 0.1)

  # Resume DPO from checkpoint if it exists
  best_dpo_loss = float('inf')
  if os.path.exists('dpo_best.pt'):
    print("Resuming DPO from dpo_best.pt")
    dpo_saved = torch.load('dpo_best.pt', weights_only=False)
    policy_model.load_state_dict(dpo_saved['model'])
    optimizer.load_state_dict(dpo_saved['optim'])
    best_dpo_loss = dpo_saved.get('best_loss', float('inf'))
    print(f"  -> Best DPO loss so far: {best_dpo_loss:.3f}")

  policy_model.train()

  for epoch in range(args.dpo_epochs):
    total_loss = 0
    num_batches = 0
    random.shuffle(preference_pairs)

    # Iterate over mini-batches
    for i in tqdm(range(0, len(preference_pairs), args.dpo_batch_size), desc=f'dpo-{epoch}', disable=TQDM_DISABLE):
      batch = preference_pairs[i: i + args.dpo_batch_size]
      prompt_texts  = [p[0] for p in batch]
      chosen_texts  = [p[1] for p in batch]
      rejected_texts = [p[2] for p in batch]

      # Per-sample prompt lengths (in tokens) — same for chosen and rejected
      prompt_lens = [tokenizer(p, return_tensors='pt')['input_ids'].shape[1] for p in prompt_texts]

      # Tokenize and pad chosen / rejected separately (they may differ in length)
      chosen_enc  = tokenizer(chosen_texts,  return_tensors='pt', padding=True, truncation=True).to(device)
      rejected_enc = tokenizer(rejected_texts, return_tensors='pt', padding=True, truncation=True).to(device)

      chosen_cmask  = make_completion_mask(chosen_enc['input_ids'],  prompt_lens, device)
      rejected_cmask = make_completion_mask(rejected_enc['input_ids'], prompt_lens, device)

      optimizer.zero_grad()

      with torch.amp.autocast('cuda'):
        logp_chosen  = get_completion_logprobs(policy_model, chosen_enc['input_ids'],  chosen_enc['attention_mask'],  chosen_cmask)
        logp_rejected = get_completion_logprobs(policy_model, rejected_enc['input_ids'], rejected_enc['attention_mask'], rejected_cmask)

      with torch.no_grad(), torch.amp.autocast('cuda'):
        logp_chosen_ref  = get_completion_logprobs(ref_model, chosen_enc['input_ids'],  chosen_enc['attention_mask'],  chosen_cmask)
        logp_rejected_ref = get_completion_logprobs(ref_model, rejected_enc['input_ids'], rejected_enc['attention_mask'], rejected_cmask)

      rewards_chosen  = args.dpo_beta * (logp_chosen  - logp_chosen_ref)
      rewards_rejected = args.dpo_beta * (logp_rejected - logp_rejected_ref)
      loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      num_batches += 1

    dpo_loss = total_loss / num_batches
    print(f"DPO Epoch {epoch}: loss :: {dpo_loss:.3f}")
    if dpo_loss < best_dpo_loss:
      best_dpo_loss = dpo_loss
      save_model(policy_model, optimizer, saved['args'], 'dpo_best.pt', best_loss=best_dpo_loss)
      print(f"  -> New best DPO model saved (loss {best_dpo_loss:.3f})")


def save_model(model, optimizer, args, filepath, best_loss=None):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'best_loss': best_loss,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


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


  # Auto-resume from train_best.pt if it exists
  start_epoch = 0
  best_loss = float('inf')
  if os.path.exists('train_best.pt'):
      print("Resuming from train_best.pt")
      saved = torch.load('train_best.pt', weights_only=False)
      model.load_state_dict(saved['model'])
      optimizer.load_state_dict(saved['optim'])
      best_loss = saved.get('best_loss', float('inf'))
      print(f"Resuming from checkpoint (best loss so far: {best_loss:.3f})")

  scaler = torch.amp.GradScaler('cuda')

  for epoch in range(start_epoch, args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask = batch['token_ids'], batch['attention_mask']
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
          logits = model(b_ids, b_mask)
          logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
          labels = b_ids[:, 1:].contiguous().flatten()
          loss = F.cross_entropy(logits, labels, reduction='mean')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    if train_loss < best_loss:
      best_loss = train_loss
      save_model(model, optimizer, args, 'train_best.pt', best_loss=best_loss)
      print(f"  -> New best SFT model saved (loss {best_loss:.3f})")

    # print('Generating several output sonnets...')
    # model.eval()
    # for batch in held_out_sonnet_dataset:
    #     encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
    #     output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
    #     print(f'{batch[1]}{output[1]}\n\n')

@torch.no_grad()
def generate_submission_sonnets(args, model_path=None):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  if model_path is None:
    model_path = 'train_best.pt'
  saved = torch.load(model_path, weights_only=False)

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
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dev.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=0.7)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  parser.add_argument("--dpo", action='store_true', help="Run DPO fine-tuning after SFT")
  parser.add_argument("--dpo_epochs", type=int, default=5, help="Number of DPO training epochs")
  parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta (KL penalty strength)")
  parser.add_argument("--dpo_batch_size", type=int, default=5, help="Batch size for DPO training")

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
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  if args.dpo:
    torch.cuda.empty_cache()
    train_dpo(args)
    generate_submission_sonnets(args, model_path='dpo_best.pt')
  else:
    generate_submission_sonnets(args)