# Small Language Model Finetuning on Synthetic Data

This Notebook is created by Rajashekar Kasturi and Thasneem Vazim, Inspired from the Orginal work of Finetuning Notebook by [Rahul Nair](https://github.com/rahulunair/genAI/blob/main/gemma_xpu_finetuning.ipynb)

## LoRA Finetuning with Synthetic Data on Intel XPUs

### Conda Environment and Jupyter kernel Setup on XPUs

```bash
# Creating and Activating Conda environment

conda create -n xpu-finetune python=3.10 -y
conda activate xpu-finetune
```

### Setup Instructions

- Intel Extension for PyTorch: [Link](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.3.110%2bxpu&os=linux%2fwsl2&package=pip)

```bash
# Install PyTorch, Transformers, Torch

python -m pip install torch==2.3.1+cxx11.abi torchvision==0.18.1+cxx11.abi torchaudio==2.3.1+cxx11.abi intel-extension-for-pytorch==2.3.110+xpu oneccl_bind_pt==2.3.100+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
python -m pip install ipykernel peft pandas
python -m pip install datasets==3.0.1
```

#### Sanity Test

```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```

### Launch Jupyter Kernel

```bash
python -m ipykernel install --user --name=xpu-finetune
```

### PyTorch 2.5 now natively supports Intel GPUs [Recommended]

- Official Release Doc: [here](https://pytorch.org/docs/stable/notes/get_start_xpu.html)
- Pre-requisites for setup on Intel GPUs: [here](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-5.html)
- The Inter Tiber AI Cloud has preconfigured oneAPI Environment where we can get started easily [here](https://console.cloud.intel.com/learning).