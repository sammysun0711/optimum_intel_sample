#from transformers import AutoModelForSequenceClassification
from optimum.intel.openvino import OVModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
#model = AutoModelForSequenceClassification.from_pretrained(model_id)
hf_model = OVModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_pipe_cls = pipeline("text-classification", model=hf_model, tokenizer=tokenizer)
text = "He's a dreadful magician."
hf_outputs = hf_pipe_cls(text)
print("FP32 model outputs: ", hf_outputs)

ov_ptq_model = OVModelForSequenceClassification.from_pretrained("nncf_ptq_results", from_transformers=False)
ov_ptq_pipe_cls = pipeline("text-classification", model=ov_ptq_model, tokenizer=tokenizer)
ov_ptq_outputs = ov_ptq_pipe_cls(text)
print("PTQ quantized INT8 model outputs: ", ov_ptq_outputs)

ov_qat_model = OVModelForSequenceClassification.from_pretrained("nncf_qat_results", from_transformers=False)
ov_qat_pipe_cls = pipeline("text-classification", model=ov_qat_model, tokenizer=tokenizer)
ov_qat_outputs = ov_qat_pipe_cls(text)
print("QAT quantized INT8 model outputs: ", ov_qat_outputs)

