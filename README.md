# deepseek-cheat-sheet




# Resource Requirements

<br><br>

## DeepSeek-Coder-V2-Lite

| **Model**                     | **GPU Requirements**                 | **Additional Notes**                                                                                                                                 |
|-------------------------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **DeepSeek-Coder-V2**         | 80GB * 8 GPUs (BF16 format)          | Standard version requires a significant amount of GPU memory for inference.                                                                          |
| **DeepSeek-Coder-V2-Lite**    | 40GB * 1 GPU (BF16 format)           | Inference possible on a single 40GB GPU.                                                                                                             |
| **DeepSeek-Coder-V2-Lite**    | 3 * 24GB GPUs (e.g., 4090 GPUs)      | Requires **TP 2** (Tensor Parallelism Level 2) or **PP 2** (Pipeline Parallelism Level 2) to function correctly. Alternatively, use a quantized model. |
| **Quantized Version**         | Less than 24GB GPU memory required   | Available for use on platforms like Ollama. Offers reduced memory requirements while maintaining reasonable performance.                              |

### Notes:
- For users with **A100 40GB GPUs**, additional throughput information may be needed for optimization.
- The quantized version is recommended for setups with limited GPU memory.














<br><br>
___
___
<br><br>

## llama.cpp 
- https://huggingface.co/bartowski/DeepSeek-Coder-V2-Instruct-GGUF
