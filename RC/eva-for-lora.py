import os
import sys

import argparse
import torch
import json
import transformers
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, pipeline

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    test_path: str = "",
    out_path: str = "",
    temperature: float = 0.1,
    prompt_template: str = ""
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if lora_weights is not None:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if lora_weights is not None:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )


    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto"
    )

    def evaluate(
        instruction,
        input=None,
        temperature=temperature,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=768,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        
        sequences = generation_pipe(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            num_beams=num_beams,
        )   

        s = sequences[0]["generated_text"]
        print(s)
        sys.stdout.flush()
        return s


    # add test set
    test_set = json.loads(open(test_path).read())
    outputs = []

    for item in tqdm(test_set):
        instruction = item['instruction']
        _input = item['input']
        outputs.append({'text': evaluate(instruction, _input)})
    
    with open(out_path, 'w') as fo:
        janswer = json.dumps(outputs)
        fo.write(janswer) 



if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_weights', default=None, type=str,
                        help="If None, perform inference on the base model")
    parser.add_argument('--test_path', default=None, type=str, required=True)
    parser.add_argument('--out_path', default=None, type=str, required=True, help="the path to output triplets")
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--load_8bit', action='store_true',
                        help='only use CPU for inference')
    args = parser.parse_args()
    print(f'Debug:base_model={args.base_model}')
    main(args.load_8bit, args.base_model, args.lora_weights, args.test_path, args.out_path, args.temperature, "")
