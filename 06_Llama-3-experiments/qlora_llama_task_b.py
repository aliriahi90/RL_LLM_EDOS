import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig
from trl import SFTTrainer

df = pd.read_csv("edos_labelled_aggregated.csv")
df = df[df['label_category'] != 'none']

train_df, dev_df, test_df = df[df['split'] == 'train'], df[df['split'] == 'dev'], df[df['split'] == 'test']

print(f"train: {train_df.shape[0]}, dev:{dev_df.shape[0]}, test:{test_df.shape[0]}")

# convert to Dataset format
train = Dataset.from_pandas(train_df[['text', 'label_category']])
dev = Dataset.from_pandas(dev_df[['text', 'label_category']])
test = Dataset.from_pandas(test_df[['text', 'label_category']])

print(train)
print(dev)
print(test)

llm_path = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(llm_path, 
                                          padding_side="right",
                                          token="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")

tokenizer.pad_token = tokenizer.eos_token

# 4-bit Quantization Configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model with 4-bit precision
model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=quant_config, device_map={"": 0}, token="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")

# Set PEFT Parameters
peft_params = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Define training parameters
training_params = TrainingArguments(
    output_dir="task_b_llm",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

prompt_template = """Category of Sexism: for posts which are sexist, a four-class classification where systems have to predict one of four categories: (1) threats, (2)  derogation, (3) animosity, (4) prejudiced discussion. 

Given a post determine the post is belong to which class:
1. threats, plans to harm and incitement
2. derogation 
3. animosity
4. prejudiced discussions

### Post: 
{POST}
### Class: """

def preprocess_function(examples):
    inputs = [prompt_template.replace("{POST}", text) + label  for text, label in zip(examples['text'], examples['label_category'])]
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    return model_inputs

tokenized_train = train.map(preprocess_function, batched=True)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_train,
    peft_config=peft_params,
    dataset_text_field="input_ids",  # Specify the correct field for text
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_params,
    packing=False
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("task_b_llm")