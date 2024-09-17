# AgeGoogleNet

Inference code for AgeGoogleNet model which available on [onnx/models](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/age_gender) repository. 

# Installation

```shell
pip install -r requirements.txt
```

# Pretrained models

| Name         | Model Size (MB) | Link                                                                                        | SHA-256                                                          |
|--------------|-----------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| AgeGoogleNet | 22.9            | [ONNX](https://github.com/clibdev/agegooglenet/releases/latest/download/age-googlenet.onnx) | fa2a3228e425056aa2b080b3afd3cf607327c86616e952602ed67b5fc16ab356 |

# Inference

```shell
python test.py --model-path age-googlenet.onnx --image-path data/88_megaage_asian_32_age.jpg
```
