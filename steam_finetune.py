import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from config import data_path, ensure_data_dir

ensure_data_dir()

# CSV 로드 후 데이터셋 변환 (data/steam.csv)
df = pd.read_csv(data_path("steam.csv"))
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1)

# 토크나이저 로드
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 토큰화 함수
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize_fn, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./steam_kcelectra",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()
trainer.evaluate()

# requirements
# !pip install transformers==4.41.2
# !pip install peft==0.7.1
# !pip install numpy==1.25.2

# 사용법
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("./stream-kcelectra-finetuned")
# tokenizer = AutoTokenizer.from_pretrained("./stream-kcelctra-finetuned")