# Synthetic Data Generation

- Synthetic data is, as the name suggests, artificial data generated to mimic real data.
- Typically, synthetic data is generated using sophisticated Generative AI techniques to create data similar in structure, features, and characteristics to the data found in real-world applications.
- Some key considerations when evaluating the quality of synthetic data include the randomness of the sample, how well it captures the statistical distribution of real data, and whether it includes missing or erroneous values.

## Our Strategy for Synthetic Data Generation

- This work incorporates insights from [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094) .
- Previous research tends to diversify the data synthesis prompt through the following two paradigms, which are instance-driven and key-point-driven, but unfortunately, neither can practically achieve scalable synthetic data creation.

### Persona-driven Synthetic Data Creation Approach

- We follow a novel persona-driven data synthesis methodology.
- The personas can be regarded as distributed carriers of
world knowledge, and each individual can be associated with their unique knowledge, experience,
interest, personality and profession.
- Thus, they can tap into almost every perspective encapsulated
within the LLM to create diverse synthetic data at scale.
- This approach involves integrating a persona into the appropriate position in a data synthesis prompt.
- Driven by the 1 billion personas in Persona Hub, this approach can easily create
diverse synthetic data at a billion scale.

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/rskasturi/usecases.git && cd usecases
cd synthetic-data-generation
```

### Pre-requistes

#### [**Setting up vLLM**](./vllm-setup/)

- Detailed instructions for vLLM setup on Intel® GPUs are [here](./vllm-setup/).
- **Why Use vLLM?**: vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.

## Notebooks

- Synthetic Data generation using vLLM on Intel® XPUs: [Notebook](./data-generation/synthetic_datagen_xpu.ipynb)
- Small Language Model finetuning with Synthetic Data on Intel® XPUs: [Notebook](./finetuning-synthetic-data/)

## Verfied Environment & Hardware

- [**Intel® Tiber™ AI Cloud**](https://console.cloud.intel.com)
  
  ```bash
    [opencl:cpu] Intel(R) OpenCL, Intel(R) Xeon(R) Platinum 8468V OpenCL 3.0 (Build 0) [2024.18.7.0.11_160000]
    [opencl:gpu] Intel(R) OpenCL Graphics, Intel(R) Data Center GPU Max 1100 OpenCL 3.0 NEO  [23.35.27191.42]
    [level_zero:gpu] Intel(R) Level-Zero, Intel(R) Data Center GPU Max 1100 1.3 [1.3.27191]
    ```
