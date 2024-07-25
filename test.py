from vllm import LLM, SamplingParams
import torch

prompts = [
    "Hello, my name is",
]
sampling_params = SamplingParams(top_p=1, top_k=1, temperature=1)

llm = LLM(model="facebook/opt-125m", dtype=torch.float32)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")    
    