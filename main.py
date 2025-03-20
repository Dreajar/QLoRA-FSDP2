# Objective: Finetune Llama 3.1 8B on 2+ GPUs with FSDP2
# WTS this working in a free Kaggle notebook with 2 x Tesla T4 GPUs
# WT utilize 0 bubble scheduling somehow :pray:
#   TODO: look into TorchPipe vs integrate w accel's pipeline parallelism
#   TODO: prolly need to wrap model into pipeline stages + use pipeline scheduler?
#   TODO: Look into "torch.distributed.pipeline" module
#   TODO: accelerate (FSDP2/related) https://github.com/huggingface/accelerate/pull/3394, Torch Titan, other repos etc.
# TODO: I want it to be fully transformers compatible -> TODO: use TrainingArguments, Trainer, or TRL related classes
# Loss should = single GPU training (otherwise there's no point lmao)
# Use nf4 from bitsandbytes
# TODO: Put working sample in Kaggle 2x Tesla T4 notebook

# TODO: Fix it all
import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# 1. Remove Unsloth Patches (IMPORTANT YO!)
def remove_patched_module(package_name):
    modules_to_delete = [
        name for name in sys.modules
        if name == package_name or name.startswith(package_name + ".")
    ]
    for name in modules_to_delete: del sys.modules[name]

remove_patched_module("trl")
remove_patched_module("transformers")
remove_patched_module("peft")
remove_patched_module("bitsandbytes")

# 2. Environment Setup
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True," \
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"

# 3. Model and Tokenizer Loading
max_seq_length = 2048
model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit"  # or meta-llama/Meta-Llama-3-8B
dtype = torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="sdpa",
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token # very important, add pad token.

# 4. LoRA Configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() #verify lora parameters are trainable.

# 5. Dataset Preparation
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files={"train": url}, split="train[:10%]").shuffle(seed=42)

def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_seq_length, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy() #labels are input ids for causal LM
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, num_proc=4)

# 6. Training Arguments and Trainer with FSDP2
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1, # adjust epochs
    per_device_train_batch_size=2, # adjust batch size
    gradient_accumulation_steps=4, # adjust gradient accumulation
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True, # enable mixed precision
    fsdp="full", # enable FSDP
    fsdp_transformer_layer_cls_to_wrap="LlamaDecoderLayer", # specify the transformer layer to wrap
    fsdp_state_dict_type="FULL_STATE_DICT", # or SHARDED_STATE_DICT
    fsdp_offload_params=True, # enable CPU offloading
    fsdp_auto_wrap_policy={torch.nn.Linear}, # auto wrap linear layers
    report_to="none", # remove when debugging is done
    logging_steps=10,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 7. Training
trainer.train()

# 8. Clean up
del model
import gc
gc.collect()
torch.cuda.empty_cache()

# MISSING: Pipeline Parallelism

# MISSING: Loss Equivalence Check (Guidance)
# Prolly import matplotlib and overlap 2 loss curves