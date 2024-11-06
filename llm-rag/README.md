# Hybrid RAG

## Usecase: Cache-Enabled Hybrid Retrievals
Enhancing Contextual Understanding in Chat Histories

## Verified Environment
- **Platform:** [Intel® Tiber™ AI Cloud](#https://www.intel.com/content/www/us/en/developer/tools/devcloud/services.html)
- **Hardware:** [Intel® Data Center GPU Max Series](#https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html)

## Environment Setup

### 1. Unset and List SYCL devices:
Reset the oneAPI device selector variable to display all available OpenCL and Level Zero devices found in the ITAC.
```bash
    unset ONEAPI_DEVICE_SELECTOR
    sycl-ls
```

### 2. Create and Activate Python Virtual Environment: 
To create and activate a Python virtual environment to manage dependencies and isolate project settings.
```bash
python3 -m venv llm-rag_env
source llm-rag_env/bin/activate
```

### 3. Install IPEX(Intel® Extension for PyTorch):
This command installs the latest versions of PyTorch, TorchVision, TorchAudio, and oneCCL bindings for PyTorch.

**Note:** This command may change with new releases. Please check for the most up-to-date installation instructions.

The following information outlines the specifications used for this project:

| Name      | Details                   |
|-----------|---------------------------|
| Platform  | GPU                       |
| Version   | v2.3.110+xpu              |
| OS        | Linux                     |
| Package   | pip                       |


```bash
python -m pip install torch==2.3.1+cxx11.abi torchvision==0.18.1+cxx11.abi torchaudio==2.3.1+cxx11.abi intel-extension-for-pytorch==2.3.110+xpu oneccl_bind_pt==2.3.100+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### Sanity check:

Run a simple sanity test to confirm the correct versions of PyTorch* and Intel® Extension for PyTorch* are installed, along with detected GPU card(s).

```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```

### 4. Install the required packages:
To ensure the project functions correctly, download and set up the necessary software dependencies.
    
```bash
    python -m pip install -r requirements.txt
    python -m pip install sentence-transformers==2.2.2
```
**Note:** To run this project, it is essential to use a compatible version of the sentence-transformers library. Please install version 2.2.2

## Overview

### Description
This project implements a conversational AI system using LangChain, PyTorch, and Hugging Face Transformers. It enables users to ask questions based on the content of PDF documents and provides answers by retrieving relevant information.

### Key Highlights
1. **Hybrid Retrievers**: Combines Chroma and BM25 retrieval methods to enhance the accuracy of information retrieval.
   
2. **Cache Techniques**: Utilizes cache-backed embeddings for efficient retrieval and to reduce processing time.

3. **Chat Histories**: Maintains session-based chat histories, allowing for contextual awareness in conversations.

4. **Contextual Retrieval**: Reformulates user questions based on chat history to provide more relevant responses.

5. **Local Storage Management**: Manages persistent storage of documents and embeddings using local file storage.


