# Automatic Speech Recognition Quantization using Intel Neural Compressor
This usecase describes a detailed step-by-step code walkthrough on How to use [Intel Neural Compressor](https://github.com/intel/neural-compressor) for Quantizing [Whisper](https://huggingface.co/openai/whisper-tiny) Model.
- Load the Model and inference before Quantization
- Quantize the Model
- Inference of the Quanized model

## Environment

- Platform:Intel Tiber AI Cloud
- OS version : Ubuntu 22.04
- CPU : Intel(R) Xeon(R) Platinum 8468V

## Environment setup

Step 1. Create and activative the virtual environment
```bash
  python3 -m venv asr-quant
  source ~/env/asr-quant/bin/activate
```

Step 2. Install the required dependencies

```bash
python pip install -r requirements.txt
```

Step 3. Below mentioned changes need to be done inorder to source files of the packages to run the quantization script.

- cd ```/home/<user-name>/env/asr-quant/lib/python3.10/site-packages/transformers/models/whisper```
- In ```modeling_whisper.py```
     
  Line number may vary with respect to transfomer versions

    ```bash
    # Remove Below Line
    expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]

    # Add below code
    try:
     expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
    except AttributeError:
       try:
           expected_seq_length = self.config.max_source_positions * self.conv1.module.module.stride[0] * self.conv2.module.module.stride[0]
       except AttributeError:
           try:
               expected_seq_length = self.config.max_source_positions * self.conv1.module.module.module.stride[0] * self.conv2.module.module.module.stride[0]
           except AttributeError:
               expected_seq_length = None 
     ```
- In ```generation_whisper.py``` replace 
    ```bash
    # Remove Below line
    
    input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]

    # Add below code
    
    try:
        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        except AttributeError:
            try:
                input_stride = self.model.encoder.conv1.module.module.stride[0] * self.model.encoder.conv2.module.module.stride[0]
            except AttributeError:
                try:
                    input_stride = self.model.encoder.conv1.module.module.module.stride[0] * self.model.encoder.conv2.module.module.module.stride[0]
                except AttributeError:
                    input_stride = None 
    ```

## CLI Run
  ```bash
  python Whisper_Inference_without_quantization.py 
  ```
  ![alt text](image.png)
  
  ```bash
  python Whisper_quantization.py
  ```
  ![alt text](image-1.png)
  
  ```bash
  python quantized_inference.py
  ```
  ![alt text](image-2.png)

  ## To run the detailed [whisper_notebook](https://github.com/rskasturi/usecases/blob/master/asr-quantization/Whisper_quantization.ipynb)
  
