import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)
print("Started")
# Load and prepare dataset
df = pd.read_csv("emails_dataset.csv")
df = df.rename(columns={"email_thread": "input", "summary": "output"})
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.1)

# Tokenize
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess(example):
    inputs = tokenizer(example["input"], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(example["output"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=["input", "output"])

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Training setup
training_args = Seq2SeqTrainingArguments(
    output_dir="./longt5-email-summarizer",
    eval_steps=500,  # Replace evaluation_strategy with eval_steps for periodic evaluation
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    predict_with_generate=True,
    #logging_dir="./logs",
    #save_total_limit=1,
    fp16=False  # Set to True if you're using a GPU
)

print("Trainer started")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer
)
print("Train samples:", len(tokenized["train"]))
print("Test samples:", len(tokenized["test"]))


# Start training
trainer.train()
trainer.save_model("longt5-email-summarizer")
print("Model saved")
