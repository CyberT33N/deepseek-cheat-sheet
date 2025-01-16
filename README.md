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
<br><br>
___
___
<br><br>
<br><br>





# DeepSeek-Coder-V2-Instruct


<br><br>

## DeepSeek-Coder-V2-Instruct-GGUF (llama.cpp)

<details><summary>Click to expand..</summary>

# Details
-  Q4_K_M not working on 4090 way too slow..


### Download
- https://huggingface.co/bartowski/DeepSeek-Coder-V2-Instruct-GGUF
```shell
huggingface-cli download bartowski/DeepSeek-Coder-V2-Instruct-GGUF --include "DeepSeek-Coder-V2-Instruct-Q4_K_M.gguf/*" --local-dir "/home/userName/Projects/ai/resources/models/llm/deepseek"
```


| Modell                                             | Version     | VRAM (geschätzt) | Beschreibung                                                                                        | Eignung für RTX 4090                 |
|----------------------------------------------------|-------------|------------------|----------------------------------------------------------------------------------------------------|--------------------------------------|
| **DeepSeek-Coder-V2-Instruct-Q4_K_M.gguf**         | Q4_K_M      | 142.45 GB        | Gute Qualität, nutzt etwa 4.83 Bits pro Gewicht, empfohlen.                                          | Zu groß für RTX 4090                |
| **DeepSeek-Coder-V2-Instruct-Q3_K_XL.gguf**        | Q3_K_XL     | 123.8 GB         | Experimentell, verwendet f16 für Einbettungs- und Ausgabewichtungen. Niedrigere Qualität, aber nutzbar. | Zu groß für RTX 4090                |
| **DeepSeek-Coder-V2-Instruct-Q3_K_M.gguf**         | Q3_K_M      | 112.7 GB         | Relativ niedrige Qualität, aber nutzbar.                                                              | Zu groß für RTX 4090                |
| **DeepSeek-Coder-V2-Instruct-Q2_K_L.gguf**         | Q2_K_L      | 87.5 GB          | Experimentell, verwendet f16 für Einbettungs- und Ausgabewichtungen. Niedrige Qualität, aber nutzbar.  | Eventuell zu groß für RTX 4090      |
| **DeepSeek-Coder-V2-Instruct-Q2_K.gguf**           | Q2_K        | 86.0 GB          | Niedrige Qualität, aber nutzbar.                                                                     | Eventuell zu groß für RTX 4090      |
| **DeepSeek-Coder-V2-Instruct-IQ2_XS.gguf**         | IQ2_XS      | 68.7 GB          | Niedrigere Qualität, nutzt SOTA-Techniken zur Nutzbarkeit.                                            | Gut geeignet für RTX 4090           |
| **DeepSeek-Coder-V2-Instruct-IQ1_M.gguf**          | IQ1_M       | 52.7 GB          | Extrem niedrige Qualität, nicht empfohlen.                                                           | Gut geeignet für RTX 4090           |




</details>

