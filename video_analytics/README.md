## Video Analytics:

- **Description:** This usecase enables an interactive chat experience with video input. Using advanced multimodal models like [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA), it allows users to ask questions or interact with the content of a video.
- It aims to integrate multimodal learning, processing both video and language inputs, to perform tasks like video captioning, question answering, and more. The model builds on LLaMA architecture and adapts it for video-specific challenges.

## Verified Environment:
[Intel® Tiber™ AI Cloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/services.html)

[Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html)

### Environment Setup:
The following information outlines the specifications used for this project:

| Name      | Details                   |
|-----------|---------------------------|
| Platform  | GPU                       |
| Version   | 2.1.40+xpu                |
| OS        | Linux                     |
| Package   | pip                       |


### Conda Environment Setup

```bash
python3 -m venv vllama
source vllama/bin/activate
```


Install the packages with help of requirements.txt file:

```
cd video-analytics
pip install -r requirements.txt
```
```
pip install --upgrade --upgrade-strategy eager "optimum[neural-compressor]"
```
Install [IPEX](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.30%2bxpu&os=linux%2fwsl2&package=pip) with the below commands:
```
python -m pip install torch==2.1.0.post3 torchvision==0.16.0.post3 torchaudio==2.1.0.post3 intel-extension-for-pytorch==2.1.40+xpu oneccl_bind_pt==2.1.400+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
```
python -m pip install numpy==1.26.4
```

## Run the application:
Before running the [code](https://github.com/rskasturi/usecases/blob/master/video_analytics/Running_on_XPU.ipynb) we need to download the models with help of git lfs

```
git lfs install
```
```
git clone https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Pretrained
```

Now go to the below path "**./eval_configs/video_llama_eval_withaudio.yaml**" and update the respective paths for llama_model, imagebind_ckpt_path, ckpt and ckpt2.

which looks like below

![image](https://github.com/user-attachments/assets/63234ae8-4e95-435e-8b8f-5302d7642b11)

**NOTE**

If you get the below error:

- ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor' please follow the below work around.
  
  Go to the below path inside your environment <your_env>/lib/python3.x/site-packages/pytorchvideo/transforms/augmentations.py and comment the below import i.e
  
  import torchvision.transforms.functional.to_tensor as F_t
  







