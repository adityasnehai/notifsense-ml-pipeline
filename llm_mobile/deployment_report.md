# TinyLlama Mobile Prep (llama.cpp, CPU-only)

## Artifacts
- Converted GGUF (f16): `models/tinyllama-f16.gguf`
- Final quantized model: `models/tinyllama-q4_k_m.gguf`

## Final Model Size
- `models/tinyllama-q4_k_m.gguf`: `667,815,232` bytes (`636.18 MiB`)

## Pipeline Summary
1. Built `llama.cpp` locally with CMake + Ninja.
2. Used locally cached official TinyLlama v1.0 safetensors from Hugging Face cache.
3. Converted HF safetensors -> GGUF f16:
   - `python convert_hf_to_gguf.py <hf_snapshot> --outfile models/tinyllama-f16.gguf --outtype f16`
4. Quantized GGUF f16 -> Q4_K_M:
   - `./llama.cpp/build/bin/llama-quantize models/tinyllama-f16.gguf models/tinyllama-q4_k_m.gguf Q4_K_M`
5. Ran CPU-only benchmark and structured prompt test.

## CPU Benchmark (llama-bench)
- Command:
  - `./llama.cpp/build/bin/llama-bench -m ./models/tinyllama-q4_k_m.gguf -ngl 0 -t 24 -p 128 -n 64`
- Results:
  - Prompt processing (`pp128`): `297.06 ± 52.50 tok/s`
  - Token generation (`tg64`): `44.08 ± 1.93 tok/s`

## Structured Prompt Inference (64 max tokens)
- Grammar constrained to:
  - `Decision: <ALLOW/DELAY/SUPPRESS>`
  - `Reason: <short explanation>`
- Observed run metrics:
  - Prompt throughput: `297.0 tok/s`
  - Generation throughput: `6.3 tok/s`

## Memory Usage
- Peak RSS during structured run: `1,209,292 kB` (`~1180.95 MiB`)
- Peak RSS during 1-token cold-start run: `1,205,300 kB` (`~1177.05 MiB`)

## Cold Start Latency
- Measured via one-token run (`-n 1`, `--no-warmup`):
  - Wall time: `1.57 s`

## Example Structured Output
```text
Decision: DELAY
Reason: The notification is not relevant to the current activity and may distract you from your intended task. Consider skipping it or setting a reminder for later.
```

## Logs
- Benchmark log: `llm_mobile/bench.txt`
- Structured response: `llm_mobile/response_structured.txt`
- Structured run metrics: `llm_mobile/run_metrics_structured.txt`
- Cold-start run metrics: `llm_mobile/run_metrics_n1.txt`
