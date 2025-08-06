from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import os
from tqdm import tqdm

def prepare_model():
    print("🔄 Loading base model...")
    model_name = "deepseek-ai/deepseek-coder-7b-base"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def prepare_dataset(tokenizer, data_path="legal_dataset.json"):
    print("📚 Loading and processing dataset...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    # Load dataset
    dataset = load_dataset('json', data_files=data_path)['train']
    
    def tokenize_function(examples):
        # Format: instruction + input + output
        prompts = [
            f"Instruction: {inst}\nInput: {inp}\nOutput: {out}\n"
            for inst, inp, out in zip(
                examples['instruction'],
                examples['input'],
                examples['output']
            )
        ]
        return tokenizer(
            prompts,
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def train_model(model, tokenizer, dataset):
    print("🚀 Starting training...")
    training_args = TrainingArguments(
        output_dir="./lora-deepseek-legal",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([f['input_ids'] for f in data]),
            'attention_mask': torch.stack([f['attention_mask'] for f in data])
        }
    )
    
    trainer.train()
    
    # Save the final model
    print("💾 Saving model...")
    trainer.save_model("./final-lora-deepseek-legal")

def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not available. Training will be very slow on CPU.")
        user_input = input("Do you want to continue? (y/n): ")
        if user_input.lower() != 'y':
            return
    
    try:
        # Prepare model and tokenizer
        model, tokenizer = prepare_model()
        
        # Prepare dataset
        dataset = prepare_dataset(tokenizer)
        
        # Train model
        train_model(model, tokenizer, dataset)
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")

if __name__ == "__main__":
    main()