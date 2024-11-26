# Synthetic Data generation

* **Pre-requisites:** Requires you to setup vLLM on Intel XPUs, You can follow them from [here](../vllm-setup/README.md/#conda-environment-setup).

## Setup Jupyter Kernel

* Below instructions help you to create Jupyter kernel to run data generation notebook.

    ```bash
    #make sure you activate `vllm-xpu` environment in your terminal
    
    python -m pip install ipykernel
    python -m ipykernel install --user --name=vllm-xpu
    ```

* Make sure you select the ```vllm-xpu``` kernel while running the notebook.

## 💯Now you are good to run the 📒[Notebook](./synthetic_datagen_xpu.ipynb)
