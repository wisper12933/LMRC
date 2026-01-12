import sys
import json
import argparse

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from .args import read_specify_task_eval_args
from .data_loader import get_template_and_fix_tokenizer
from .model_loader import load_tokenizer, load_model


class InferenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        message = [{"role": "user", "content": item['instruction'] + '\n\n' + item['input']}]
        return {"message": message, "title": item.get('title', 'UNKNOWN')}


def main(args):
    data_args, model_args, generation_args = read_specify_task_eval_args(args)
    
    tokenizer = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args)
    
    tokenizer.padding_side = "left"
    gen_kwargs = generation_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    
    # load test data
    device = model.device
    # "/mnt/home/user28/LMRC/RC/data/all_true/docred_dev.json"
    with open(args.test_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    eval_dataset = InferenceDataset(raw_data)
    
    def inference_collator(batch_items):
        input_ids_list = []
        titles = []
        for item in batch_items:
            message = item["message"]
            titles.append(item["title"])
            
            ids = template.encode_inputs(tokenizer, message)
            input_ids_list.append(ids)
        
        batch_inputs = tokenizer.pad(
            {"input_ids": input_ids_list},
            padding=True,
            return_tensors="pt"
        )
        batch_inputs["titles"] = titles
        return batch_inputs
    
    data_loader = DataLoader(
        eval_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=inference_collator,
        num_workers=2, 
    )
    # inference
    results = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            titles = batch["titles"]
            
            # generate
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

            input_len = input_ids.shape[1]
            generated_tokens = generated_tokens[:, input_len:] 
            
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for title, pred in zip(titles, decoded_preds):
                results.append({
                    "title": title,
                    "output": pred
                })
    
    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2 Evaluation Main Function")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file containing DataArgs, ModelArgs, and GenerationArgs."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to the test data file for inference."
    )
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="Path to the output result file to save inference results."
    )
    args = parser.parse_args()
    
    main(args)
    