import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, default_data_collator
from optimum.intel.openvino import OVConfig, OVModelForSequenceClassification, OVTrainer
from optimum.intel.openvino.configuration import DEFAULT_QUANTIZATION_CONFIG
from utils import enable_overflow_fix

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_id)    
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset("glue", "sst2")
dataset = dataset.map(
    lambda examples: tokenizer(examples["sentence"], padding=True, truncation=True, max_length=128), batched=True
)
metric = load_metric("accuracy")
compute_metrics = lambda p: metric.compute(
    predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
)

# The directory where the quantized model will be saved
save_dir = "nncf_qat_results"

# Load the default quantization configuration detailing the quantization we wish to apply

ov_config_dict = DEFAULT_QUANTIZATION_CONFIG
ov_config_dict["overflow_fix"] = enable_overflow_fix()
ov_config = OVConfig(compression=ov_config_dict)

#trainer = Trainer(
trainer = OVTrainer(
    model=model,
    args=TrainingArguments(save_dir, num_train_epochs=1.0, do_train=True, do_eval=True),
    train_dataset=dataset["train"].select(range(300)),
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    ov_config=ov_config,
    feature="sequence-classification",
)
train_result = trainer.train()
metrics = trainer.evaluate()
trainer.save_model()

optimized_model = OVModelForSequenceClassification.from_pretrained(save_dir)
