import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

def main(args):
    model_name = "facebook/bart-base" # or "t5-base" depending on config
    
    # 1. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 2. Load Silver Dataset
    dataset = load_dataset("csv", data_files={"train": "data/silver/train.csv", "validation": "data/silver/val.csv"})
    
    # 3. Preprocessing function
    def preprocess_function(examples):
        inputs = examples["factual"]
        targets = examples["satirical"]
        
        # If using T5, you might need to prepend a prefix:
        # inputs = ["translate factual to satire: " + inp for inp in inputs]
        
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        labels = tokenizer(targets, max_length=128, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # 4. Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # 5. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpoints/satire_model",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True # Enable mixed precision if using a GPU
    )
    
    # 6. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 7. Train and Save
    trainer.train()
    trainer.save_model("checkpoints/satire_model_final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bart_training.yaml")
    args = parser.parse_args()
    main(args)