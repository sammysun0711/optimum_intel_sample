# optimum_intel_sample
This repo provides a simple example of how to use Optimum Intel to optimize and accelerate inference of Hugging Face Model DistilBERT with OpenVINO and NNCF on Intel CPU. 

The code was validated with following package: 
- pytorch==1.9.1
- onnx==1.13.0
- optimum-intel==1.5.2

### Setup Environment
```
conda create -n optimum-intel python=3.8
conda activate optimum-intel
python -m pip install -r requirements.txt
```

### Model Quantization with NNCF Post-Training Quantization (PTQ)
```
python post_training_quantization.py
```

### Model Quantization with NNCF Quantization-Aware training (QAT)
```
python quantization_aware_training.py
```

### FP32 & INT8 Model Inference with OpenVINO Runtime for Sequence Classfication Task
```
python openvino_inference.py
```
