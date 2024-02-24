import torch
from transformers import AutoTokenizer
import datasets
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('maywell/kiqu-70b')
dataset = datasets.load_dataset('maywell/kiqu_samples')
dataset = dataset.shuffle(seed=42)

def mistral_format(instruction, output):
    data = f'[INST] {instruction}\n[/INST] {output}\n\n'
    return data

n_sample = []
sample = ''

for i,d in enumerate(tqdm(dataset['train'])):
    sample += mistral_format(d['instruction'], d['output'])
    
    if len(tokenizer(sample[:-1], add_special_tokens=False)['input_ids']) > 4096:
        n_sample.append(tokenizer(sample[:-1], return_tensors="pt", max_length=4096, truncation=True)['input_ids'])
        sample = ''
        
torch.save(n_sample, './kiqu-instruction_4096_context_length_kiqu.pth')
