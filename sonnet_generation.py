'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import copy
import random
import re
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

    # # By default, fine-tune the full model. TODO: this is maybe not idea.
    # for param in self.gpt.parameters():
    #   param.requires_grad = True

    # By default, fine-tune the full model.
    for param in self.gpt.parameters():
      param.requires_grad = False

    for layer in self.gpt.gpt_layers[-8:]:
      attn = layer.self_attention
      attn.query = LoraLayer(attn.query, alpha = 8, rank = 4)
      attn.value = LoraLayer(attn.value, alpha = 8, rank = 4)

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """

    outputs = self.gpt(input_ids, attention_mask)
    return self.gpt.hidden_state_to_token(outputs['last_hidden_state']) # everything


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
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

  @torch.no_grad()
  def generate_beam(self, encoding, temperature=0.7, beam_size=5,
                    max_lines=14, min_line_tokens=6, max_line_tokens=16):
    """
    Constrained beam search for real sonnet generation.

    Enforces at each step:
      1. Line length: suppress newline before min_line_tokens, force after max_line_tokens.
      2. No consecutive newlines.
      3. Shakespeare rhyme scheme ABAB CDCD EFEF GG: when a beam tries to end a line
         that belongs to a rhyme group already seen, apply a penalty if the last word
         does not suffix-rhyme with the reference word for that group.
    Beams are ranked by length-normalised cumulative log prob.
    """
    # Shakespeare ABAB CDCD EFEF GG rhyme groups (0-indexed over 14 lines)
    RHYME_GROUPS = [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 6]
    RHYME_PENALTY = 8.0   # subtracted from log prob when rhyme constraint violated

    def last_word(text):
      words = re.findall(r"[a-zA-Z]+", text)
      return words[-1].lower() if words else ''

    def rhymes(w1, w2):
      if not w1 or not w2 or w1 == w2:
        return False
      return w1[-3:] == w2[-3:] or w1[-2:] == w2[-2:]

    device = self.get_device()
    newline_id = self.tokenizer.encode('\n')[0]
    prompt_len = encoding.shape[1]

    # Seed rhyme_words from the prompt lines themselves
    prompt_text = self.tokenizer.decode(encoding[0].cpu().tolist())
    prompt_lines = [l for l in prompt_text.split('\n') if l.strip()]
    initial_lines = len(prompt_lines)
    seed_rhyme_words = {}
    for i, line in enumerate(prompt_lines):
      if i < len(RHYME_GROUPS):
        w = last_word(line)
        g = RHYME_GROUPS[i]
        if w and g not in seed_rhyme_words:
          seed_rhyme_words[g] = w

    # beam state: ids [1,T], score, lines completed, tokens on current line,
    #             token index where current line started, rhyme word dict
    beams = [{
      'ids': encoding.clone(),
      'score': 0.0,
      'lines': initial_lines,
      'line_tokens': 0,
      'line_start': encoding.shape[1],
      'rhyme_words': dict(seed_rhyme_words),
    }]
    completed = []

    for _ in range(max_lines * max_line_tokens):
      if not beams:
        break

      candidates = []

      for beam in beams:
        ids = beam['ids']
        attn = torch.ones(ids.shape, dtype=torch.int64, device=device)
        logits = self.forward(ids, attn)
        log_probs = F.log_softmax(logits[0, -1, :] / temperature, dim=-1)  # [vocab]

        force_newline  = beam['line_tokens'] >= max_line_tokens
        suppress_newline = beam['line_tokens'] < min_line_tokens

        if force_newline:
          next_tokens = [newline_id]
          next_lps    = [log_probs[newline_id].item()]
        else:
          # Fetch extra candidates so filtering doesn't leave us with too few
          top_lps, top_toks = torch.topk(log_probs, beam_size * 3)
          next_tokens = top_toks.tolist()
          next_lps    = top_lps.tolist()

        for tok, lp in zip(next_tokens, next_lps):
          if tok == self.tokenizer.eos_token_id:
            continue
          if tok == newline_id and ids[0, -1].item() == newline_id:
            continue
          if tok == newline_id and suppress_newline:
            continue

          adjusted_lp = lp

          # --- Rhyme constraint: penalise ending a line that violates the scheme ---
          if tok == newline_id:
            line_idx = beam['lines']
            if line_idx < len(RHYME_GROUPS):
              group = RHYME_GROUPS[line_idx]
              ref_word = beam['rhyme_words'].get(group)
              if ref_word is not None:
                line_toks = ids[0, beam['line_start']:].cpu().tolist()
                line_text = self.tokenizer.decode(line_toks)
                w = last_word(line_text)
                if w and not rhymes(w, ref_word):
                  adjusted_lp -= RHYME_PENALTY

          new_ids        = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=1)
          new_lines      = beam['lines'] + (1 if tok == newline_id else 0)
          new_line_toks  = 0 if tok == newline_id else beam['line_tokens'] + 1
          new_score      = beam['score'] + adjusted_lp

          # Update rhyme registry when a line ends
          new_rhyme_words = beam['rhyme_words']
          new_line_start  = beam['line_start']
          if tok == newline_id:
            line_idx = beam['lines']
            if line_idx < len(RHYME_GROUPS):
              group = RHYME_GROUPS[line_idx]
              if group not in new_rhyme_words:
                line_toks = ids[0, beam['line_start']:].cpu().tolist()
                w = last_word(self.tokenizer.decode(line_toks))
                if w:
                  new_rhyme_words = dict(new_rhyme_words)
                  new_rhyme_words[group] = w
            new_line_start = new_ids.shape[1]

          candidates.append({
            'ids': new_ids,
            'score': new_score,
            'lines': new_lines,
            'line_tokens': new_line_toks,
            'line_start': new_line_start,
            'rhyme_words': new_rhyme_words,
          })

      if not candidates:
        break

      def norm_score(b):
        return b['score'] / max(b['ids'].shape[1] - prompt_len, 1)

      candidates.sort(key=norm_score, reverse=True)

      beams = []
      for c in candidates:
        if c['lines'] >= max_lines:
          completed.append(c)
        elif len(beams) < beam_size:
          beams.append(c)

    pool = completed if completed else beams
    best = max(pool, key=lambda b: b['score'] / max(b['ids'].shape[1] - prompt_len, 1))
    text = self.tokenizer.decode(best['ids'][0].cpu().tolist())[3:]
    return best['ids'], text


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


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  # for batch in sonnet_dataset:
  #   print(batch)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  best_loss = float('inf')
  patience = 3
  patience_counter = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device) #[batch_size, sen_leng]
      b_mask = b_mask.to(device)


      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      # print(logits.shape) # ([batch_size, 166, 50257])
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      # print(labels) # ([(batch_size * 166 - 1)])
      loss = F.cross_entropy(logits, labels, reduction='mean') #finding the correct token_id and its corresponding
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    model.eval()

    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')

    if train_loss < best_loss:
      best_loss = train_loss
      patience_counter = 0
      # save_model(model, optimizer, args, f'{epoch}_{args.filepath}')
      save_model(model, optimizer, args, f'best_{args.filepath}')
    else:
      patience_counter += 1
      if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}: loss has not improved for {patience} epochs.")
        break


def get_sequence_logprob(model, input_ids, attention_mask, prompt_lens):
  """
  Sum of log probs for completion tokens only (tokens after prompt_len).
  Handles batched inputs with per-sample prompt_lens and padding masks.
  input_ids:      [B, T]
  attention_mask: [B, T]  (0 for padding)
  prompt_lens:    list of ints, one per sample
  Returns:        [B]
  """
  logits = model(input_ids, attention_mask)              # [B, T, vocab]
  log_probs = F.log_softmax(logits[:, :-1], dim=-1)     # [B, T-1, vocab]
  target_ids = input_ids[:, 1:]                          # [B, T-1]
  token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

  # Mask: only score completion tokens that are not padding
  completion_mask = torch.zeros_like(token_log_probs)
  for i, prompt_len in enumerate(prompt_lens):
    # logprob at index j predicts token j+1; completion starts at token prompt_len
    completion_mask[i, prompt_len - 1:] = attention_mask[i, prompt_len:]

  return (token_log_probs * completion_mask).sum(-1)     # [B]


def train_dpo(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # Load best SFT model as policy
  saved = torch.load(f'best_{args.filepath}', weights_only=False)
  sft_args = saved['args']  # has d, l, num_heads from add_arguments()
  policy = SonnetGPT(sft_args).to(device)
  policy.load_state_dict(saved['model'])

  # Frozen reference model (same weights, no grad)
  ref_model = copy.deepcopy(policy).to(device)
  for param in ref_model.parameters():
    param.requires_grad = False
  ref_model.eval()

  tokenizer = policy.tokenizer

  # Load dev prompts (first 3 lines) and ground truth full sonnets
  dev_dataset  = SonnetsDataset(args.held_out_sonnet_dev_path)
  true_dataset = SonnetsDataset(args.true_sonnet_dev_path)

  # Build preference pairs offline using SFT policy
  print('Generating preference pairs for DPO...')
  preference_pairs = []
  policy.eval()
  with torch.no_grad():
    for dev_batch, true_batch in zip(dev_dataset, true_dataset):
      prompt_text = dev_batch[1]   # first 3 lines
      win_text    = true_batch[1]  # full ground truth sonnet

      prompt_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(device)
      prompt_len = prompt_ids.shape[1]

      _, lose_text = policy.generate(prompt_ids, temperature=args.temperature, top_p=args.top_p)

      win_ids  = tokenizer(win_text,  return_tensors='pt', truncation=True).input_ids.to(device)
      lose_ids = tokenizer(lose_text, return_tensors='pt', truncation=True).input_ids.to(device)

      preference_pairs.append((prompt_len, win_ids, lose_ids))

  # Collate all preference pairs into padded batches
  pad_id = tokenizer.pad_token_id
  prompt_lens = [p[0] for p in preference_pairs]

  def pad_batch(id_list):
    max_len = max(x.shape[1] for x in id_list)
    padded = torch.full((len(id_list), max_len), pad_id, dtype=torch.long, device=device)
    mask   = torch.zeros(len(id_list), max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(id_list):
      L = ids.shape[1]
      padded[i, :L] = ids[0]
      mask[i, :L]   = 1
    return padded, mask

  win_ids_batch,  win_mask_batch  = pad_batch([p[1] for p in preference_pairs])
  lose_ids_batch, lose_mask_batch = pad_batch([p[2] for p in preference_pairs])

  optimizer = AdamW(filter(lambda p: p.requires_grad, policy.parameters()), lr=args.lr * 0.1)
  beta = 0.1
  best_dpo_loss = float('inf')

  for epoch in range(args.dpo_epochs):
    policy.train()

    pi_win  = get_sequence_logprob(policy, win_ids_batch,  win_mask_batch,  prompt_lens)
    pi_lose = get_sequence_logprob(policy, lose_ids_batch, lose_mask_batch, prompt_lens)

    with torch.no_grad():
      ref_win  = get_sequence_logprob(ref_model, win_ids_batch,  win_mask_batch,  prompt_lens)
      ref_lose = get_sequence_logprob(ref_model, lose_ids_batch, lose_mask_batch, prompt_lens)

    loss = -F.logsigmoid(beta * ((pi_win - ref_win) - (pi_lose - ref_lose))).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"DPO Epoch {epoch}: loss :: {loss.item():.3f}.")

    if loss.item() < best_dpo_loss:
      best_dpo_loss = loss.item()
      save_model(policy, optimizer, sft_args, f'best_dpo_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  #saved = torch.load(f'best_dpo-{args.filepath}', weights_only=False)
  saved = torch.load('best_dpo_10-1e-05-sonnet.pt', weights_only=False)

  model_args = saved['args']
  if not hasattr(model_args, 'd'):
    model_args = add_arguments(model_args)
  model = SonnetGPT(model_args)
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    _, decoded_output = model.generate_beam(encoding['input_ids'], temperature=args.temperature)
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

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets_single.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_single.txt")
  parser.add_argument("--held_out_sonnet_dev_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--true_sonnet_dev_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  parser.add_argument("--dpo_epochs", type=int, default=3)

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=0.8)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

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
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  # train_dpo(args)
  generate_submission_sonnets(args)