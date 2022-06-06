from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification,TrainingArguments, Trainer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def prepare_dataset(batch):
    batch_ = tokenizer(batch["free_text"], padding=True, truncation=True)
    batch_['labels'] = batch['label_id']
    return batch_

dataset = load_dataset('csv',data_files={'train' : '../../data/processed/train.csv',
                                         'val' : '../../data/processed/val.csv'})

dataset_prepare = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], batch_size=16, num_proc=8, batched=True) 

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)

training_args = TrainingArguments(
    output_dir="./phoBert-sa",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_prepare['train'],
    eval_dataset=dataset_prepare['val'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()