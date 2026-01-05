import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

class WikiTextDataset:
    def __init__(self, tokenizer, seq_length=1024, split="train"):
        """
        Loads WikiText-103 and tokenizes it on the fly.
        Ref: 'WikiText-103 provides classical LM dynamics'
        """
        print(f"Loading WikiText-103 ({split})... this might take a minute.")
        # Using 'wikitext-103-v1' as standard
        self.dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Filter out empty lines to speed up training
        self.dataset = self.dataset.filter(lambda x: len(x['text']) > 0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. Get raw text
        text = self.dataset[idx]['text']
        
        # 2. Tokenize
        # We truncate to seq_length or pad if too short
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 3. Format for PyTorch (Input = Target for causal LM)
        input_ids = encodings['input_ids'].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone() # GPT learns to predict the next token
        }

def get_dataloader(tokenizer, batch_size=4, split="train"):
    ds = WikiTextDataset(tokenizer, split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for Windows compatibility