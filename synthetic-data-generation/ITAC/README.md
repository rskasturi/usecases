# Synthetic data generation Notebooks

Here is quick README, for comprehensive setup visit [home](../).

* These notebooks are completely standalone to run on the Jupyter environment.

* These Notebooks currently require creating **conda environment** to run them directly on training nodes of [ITAC Jupyter Notebooks](https://console.cloud.intel.com/training), because ```installation of vLLM is from source```, due to this it is causing writable permission issue on default kernels.

## Conda Environment and Jupyter Kernel Creation

  ```bash
  conda create -n sdg python=3.10 -y
  conda activate sdg
  python -m pip install ipykernel tqdm ipywidgets
  python -m ipykernel install --user --name=sdg
  ```

* Choose the ```sdg``` kernel in the notebook

## There are two notebooks

* Installation of packages using ```subprocess``` generation (has more finegrained control on exception handling):
  * [vLLM_Synthetic_Data_with_subprocess_installation](./vLLM_Synthetic_Data_with_subprocess_installation.ipynb) 
* Installation of packages using ```os.system```. The default way like how other notebooks are on ITAC.
  * [vLLM_Synthetic_Data_os_system_installation](./vLLM_Synthetic_Data.ipynb)
  