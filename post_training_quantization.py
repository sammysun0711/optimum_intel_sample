from functools import partial
from optimum.intel.openvino import OVQuantizer, OVModelForSequenceClassification, OVConfig
from optimum.intel.openvino.configuration import DEFAULT_QUANTIZATION_CONFIG
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import enable_overflow_fix

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_id)    
tokenizer = AutoTokenizer.from_pretrained(model_id)
def preprocess_fn(examples, tokenizer):
    return tokenizer(
        examples["sentence"], padding=True, truncation=True, max_length=128
    )

quantizer = OVQuantizer.from_pretrained(model)
calibration_dataset = quantizer.get_calibration_dataset(
    "glue",
    dataset_config_name="sst2",
    preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
    num_samples=100,
    dataset_split="train",
    preprocess_batch=True,
)
ov_config_dict = DEFAULT_QUANTIZATION_CONFIG
ov_config_dict["overflow_fix"] = enable_overflow_fix()
ov_config = OVConfig(compression=ov_config_dict)

# The directory where the quantized model will be saved
save_dir = "nncf_ptq_results"
# Apply static quantization and save the resulting model in the OpenVINO IR format
quantizer.quantize(calibration_dataset=calibration_dataset, save_directory=save_dir, quantization_config=ov_config)
# Load the quantized model
optimized_model = OVModelForSequenceClassification.from_pretrained(save_dir)
