from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import os
from tqdm import tqdm
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for different models"""
    name: str
    path: str
    target_modules: list
    lora_alpha: int = 32
    lora_r: int = 16
    lora_dropout: float = 0.05

class ModelRegistry:
    """Registry of supported models"""
    MODELS = {
        "deepseek": ModelConfig(
            name="deepseek",
            path="deepseek-ai/deepseek-coder-7b-base",
            target_modules=["q_proj", "v_proj"]
        ),
        "llama2": ModelConfig(
            name="llama2",
            path="meta-llama/Llama-2-7b-hf",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        "llama3": ModelConfig(
            name="llama3",
            path="meta-llama/Llama-3-7b",  # Update with actual path when available
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
    }

def prepare_model(model_name: str = "deepseek", device: str = "auto"):
    """Prepare model with configurable model choice"""
    if model_name not in ModelRegistry.MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {list(ModelRegistry.MODELS.keys())}")
    
    model_config = ModelRegistry.MODELS[model_name]
    print(f"🔄 Loading {model_name} model...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_config.path)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.path,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=model_config.target_modules,
        lora_dropout=model_config.lora_dropout,
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

def train_model(model, tokenizer, dataset, output_dir: str):
    """Train model with configurable output directory"""
    print("🚀 Starting training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
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
    trainer.save_model(os.path.join(output_dir, "final"))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train a language model for legal tasks')
    parser.add_argument('--model', type=str, default='deepseek', 
                      choices=list(ModelRegistry.MODELS.keys()),
                      help='Model to use for training')
    parser.add_argument('--output_dir', type=str, default='./models',
                      help='Directory to save the trained model')
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not available. Training will be very slow on CPU.")
        user_input = input("Do you want to continue? (y/n): ")
        if user_input.lower() != 'y':
            return
    
    try:
        # Create output directory for this specific model
        model_output_dir = os.path.join(args.output_dir, args.model)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Prepare model and tokenizer
        model, tokenizer = prepare_model(args.model)
        
        # Prepare dataset
        dataset = prepare_dataset(tokenizer)
        
        # Train model
        train_model(model, tokenizer, dataset, model_output_dir)
        
        print(f"✅ Training completed successfully! Model saved to {model_output_dir}")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")

if __name__ == "__main__":
    main()