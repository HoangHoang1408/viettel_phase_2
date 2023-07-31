import sys
from datetime import datetime
from glob import glob

import torch
import wandb
from datasets import Dataset, concatenate_datasets
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

date_time_start_run = datetime.now().strftime("%Y%m%d-%H%M").replace("-", "_")
args = {
    "huggingface_api_key": "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",
    "wandb_api_key": "59759d7f774b09319a3e0e3aebefc7fcf7ccf4f1",
    "dataset_folder_path": "/kaggle/input/training-test/",
    "dataset_size": None,
    "base_model_name": "bigscience/bloomz-3b",
    "tokenizer_args": {
        "model_max_length": 2048,
    },
    "lora_args": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
    },
    "training_args": {
        "output_dir": "checkpoints_" + date_time_start_run,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "optim": "paged_adamw_8bit",
        "logging_steps": 500,
        "save_steps": 500,
        "learning_rate": 2e-4,
        "fp16": True,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "report_to": "wandb",
        "push_to_hub": True,
        "group_by_length": True,
        # "max_steps": 360, # to test
    },
    "hub_adapter_repo_name": "chatbot_qlora_" + date_time_start_run,
    "push_adapter_to_hub": True,
}

# hubs login
login(args["huggingface_api_key"])
wandb.login(key=args["wandb_api_key"])


# load dataset
def load_dataset(folder_path, dataset_size=None):
    data = []
    for path in glob(folder_path + "/*.jsonl"):
        ds = Dataset.from_json(path)
        if len(ds.column_names) != 1:
            raise ValueError("Dataset must have only one text column")
        ds = ds.rename_column(ds.column_names[0], "text")
        data.append(ds)
    if dataset_size is None:
        return concatenate_datasets(data, axis=0).shuffle()
    return concatenate_datasets(data, axis=0).shuffle().select(range(dataset_size))


ds = load_dataset(args["dataset_paths"], args["dataset_size"])
print("\n-----------Dataset Loaded-----------\n")

# load base model
model = AutoModelForCausalLM.from_pretrained(
    args["base_model_name"],
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        device_map="auto",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    trust_remote_code=True,
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args["base_model_name"], trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = args["tokenizer_args"]["model_max_length"]

# load peft model
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(
    model,
    LoraConfig(
        lora_alpha=args["lora_args"]["alpha"],
        lora_dropout=args["lora_args"]["dropout"],
        r=args["lora_args"]["rank"],
        bias="none",
        task_type="CAUSAL_LM",
    ),
)
model.config.use_cache = False
print("\n-----------Model Loaded-----------\n")
model.print_trainable_parameters()
print()


# test inference
prompt = """Write a poem
"""
encoding = tokenizer(prompt, return_tensors="pt")
with torch.inference_mode():
    outputs = model.generate(**encoding)
tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n-----------Finish Inference-----------\n")

# model training
training_ds = ds.map(
    lambda x: tokenizer(
        x["text"],
        truncation=True,
        max_length=args["tokenizer_args"]["model_max_length"],
    ),
    remove_columns=["text"],
)

trainer = Trainer(
    model=model,
    train_dataset=training_ds,
    args=TrainingArguments(**args["training_args"]),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

if torch.__version__ >= "2" and sys.platform != "win32":
    print("Compiling model for faster runtime...")
    model = torch.compile(model)

print("\n-----------Start Training-----------\n")
trainer.train()

# push to hub
model.save_pretrained("./finals")
if args["push_adapter_to_hub"]:
    model.push_to_hub(args["hub_adapter_repo_name"])
print("\n-----------Completed-----------\n")
