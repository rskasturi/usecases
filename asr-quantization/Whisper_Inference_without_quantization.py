import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration,pipeline
from datasets import load_dataset
import time
import torchaudio


# Inference before quantization

model_id='openai/whisper-small'
model = WhisperForConditionalGeneration.from_pretrained(
    model_id,use_safetensors=True
)


processor = WhisperProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]

start_time = time.time()

result = pipe(sample)

end_time=time.time()
print(result["text"])
print(f"Time spent :{end_time-start_time} seconds")

