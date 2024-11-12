# Synthetic Data generation

## Setup Jupyter Kernel

* Setup Instructions for XPU with vLLM are [here](../vllm-setup/README.md/#conda-environment-setup).

    ```bash  
    python -m pip install ipykernel
    python -m ipykernel install --user --name=vllm-xpu    
    ```

## Environment & Hardware

* [**Intel Tiber AI Cloud**](https://console.cloud.intel.com)

```bash
[opencl:cpu] Intel(R) OpenCL, Intel(R) Xeon(R) Platinum 8468V OpenCL 3.0 (Build 0) [2024.18.7.0.11_160000]
[opencl:gpu] Intel(R) OpenCL Graphics, Intel(R) Data Center GPU Max 1100 OpenCL 3.0 NEO  [23.35.27191.42]
[level_zero:gpu] Intel(R) Level-Zero, Intel(R) Data Center GPU Max 1100 1.3 [1.3.27191]
```
