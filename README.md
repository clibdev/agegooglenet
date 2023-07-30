# AgeGoogleNet

Inference code for AgeGoogleNet model which available on [onnx/models](https://github.com/onnx/models/tree/main/vision/body_analysis/age_gender) repository. 

# Installation

```shell
pip install -r requirements.txt
```

# Pretrained models

| Name         | Link                                                                                        |
|--------------|---------------------------------------------------------------------------------------------|
| AgeGoogleNet | [ONNX](https://github.com/clibdev/agegooglenet/releases/latest/download/age_googlenet.onnx) |

# Inference

```shell
python test.py --model-path age_googlenet.onnx --image-path data/88_megaage_asian_32_age.jpg
```
