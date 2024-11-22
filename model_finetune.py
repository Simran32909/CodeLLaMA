from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

MODEL_NAME = "meta-llama/Llama-3.2-1B"
DATASET_NAME = "codeparrot/codeparrot-clean"
OUTPUT_DIR = "./llama_finetuned"
LOGGING_DIR = "./logs"
MAX_LENGTH = 2048
MAX_STEPS = 15
LEARNING_RATE = 5e-5
BATCH_SIZE = 1
EPOCHS = 3

def load_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def load_and_tokenize_dataset(dataset_name, tokenizer):
    dataset = load_dataset(dataset_name, streaming=True, split="train")

    def tokenize_function(examples):
        encodings = tokenizer(examples["content"],
                              padding="max_length",
                              truncation=True,
                              max_length=MAX_LENGTH,
                              return_attention_mask=True)  # Ensure attention mask is returned
        encodings['labels'] = encodings['input_ids'].copy()
        return encodings

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["content"])
    return tokenized_dataset

def apply_lora_configuration(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def train_model(model, tokenized_dataset, tokenizer):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_steps=100,
        save_total_limit=2,
        logging_dir=LOGGING_DIR,
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        optim="adamw_torch",
        max_steps=MAX_STEPS
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    trainer.train()

def save_model_and_tokenizer(model, tokenizer):
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

def main():
    tokenizer, model = load_tokenizer_and_model(MODEL_NAME)
    tokenized_dataset = load_and_tokenize_dataset(DATASET_NAME, tokenizer)
    model = apply_lora_configuration(model)
    train_model(model, tokenized_dataset, tokenizer)
    save_model_and_tokenizer(model, tokenizer)

if __name__ == "__main__":
    main()
