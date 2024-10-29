import time

import torch
import intel_extension_for_pytorch as ipex

from transformers import AutoModelForCausalLM, AutoTokenizer

# Accelerate Library
from accelerate import Accelerator
from accelerate.utils import gather_object

# Initialize Accelerate
accelerator = Accelerator()

prompts_all = [
    "What is Python",
    "What is Java",
    "What is C++",
    "What is C",
    "What is HTML",
    "What is Javascript",
    "What is CSS",
    "What is Rust",
] * 8

# load a base model and tokenizer
model_path = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map = {"": accelerator.process_index},
    torch_dtype = torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)  

# Sync GPUs and start the timer
accelerator.wait_for_everyone()
start = time.time()

# Divide the prompt list onto the available GPUs 
with accelerator.split_between_processes(prompts_all) as prompts:

    results=dict(outputs=[], num_tokens=0)

    for prompt in prompts:
        prompt_tokenized = tokenizer(prompt, return_tensors="pt").to("xpu")
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100)[0]

        output_tokenized=output_tokenized[len(prompt_tokenized["input_ids"][0]):]

        results["outputs"].append( tokenizer.decode(output_tokenized) )
        results["num_tokens"] += len(output_tokenized)

    results=[ results ]

results_gathered=gather_object(results)

if accelerator.is_main_process:
    timediff=time.time()-start
    num_tokens=sum([r["num_tokens"] for r in results_gathered ])

    print(f"tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}")