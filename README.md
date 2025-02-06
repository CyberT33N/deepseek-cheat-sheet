# deepseek-cheat-sheet



# Deepseek R1
- https://github.com/deepseek-ai/DeepSeek-R1

<details><summary>Click to expand..</summary>

# Hardware Requirements

| **Model**                             | **Parameters (B)** | **VRAM Requirement (GB)** | **Recommended GPU**                  |  
|----------------------------------------|-------------------|--------------------------|--------------------------------------|  
| DeepSeek-R1-Distill-Qwen-1.5B         | 1.5               | ~0.7                     | NVIDIA RTX 3060 12GB or higher      |  
| DeepSeek-R1-Distill-Qwen-7B           | 7                 | ~3.3                     | NVIDIA RTX 3070 8GB or higher       |  
| DeepSeek-R1-Distill-Llama-8B          | 8                 | ~3.7                     | NVIDIA RTX 3070 8GB or higher       |  
| DeepSeek-R1-Distill-Qwen-14B          | 14                | ~6.5                     | NVIDIA RTX 3080 10GB or higher      |  
| DeepSeek-R1-Distill-Qwen-32B          | 32                | ~14.9                    | NVIDIA RTX 4090 24GB                |  
| DeepSeek-R1-Distill-Llama-70B         | 70                | ~32.7                    | NVIDIA RTX 4090 24GB (x2)           |  


# Ollama
- https://ollama.com/library/deepseek-r1:32b
```shell
# ollama run deepseek-r1:14b
ollama run deepseek-r1:32b
```
- 32b Works with 4090 but not fast. But it is okay :)  If you want faster resonse then try 14b


  
</details>









<br><br>
<br><br>




# DeepSeek Coder

<details><summary>Click to expand..</summary>










# DeepSeek-Coder-V2-Lite-Instruct






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





## DeepSeek-Coder-V2-Lite-Instruct-GGUF (llama.cpp)
- https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF
- Perfect for RTX 4090

<details><summary>Click to expand..</summary>


| Modell                                               | Version    | VRAM (geschätzt) | Beschreibung                                                                                        | Eignung für RTX 4090                 |
|------------------------------------------------------|------------|------------------|----------------------------------------------------------------------------------------------------|--------------------------------------|
| **DeepSeek-Coder-V2-Lite-Instruct-Q8_0_L.gguf**       | Q8_0_L     | 17.09 GB         | Experimentell, verwendet f16 für Einbettungs- und Ausgabewichtungen. Extrem hohe Qualität, selten nötig. | Gut geeignet, aber overkill          |
| **DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf**         | Q8_0       | 16.70 GB         | Extrem hohe Qualität, selten nötig.                                                                | Gut geeignet, aber overkill          |
| **DeepSeek-Coder-V2-Lite-Instruct-Q6_K_L.gguf**       | Q6_K_L     | 14.56 GB         | Experimentell, verwendet f16 für Einbettungs- und Ausgabewichtungen. Sehr hohe Qualität, fast perfekt. | Gut geeignet, leicht übertrieben     |
| **DeepSeek-Coder-V2-Lite-Instruct-Q6_K.gguf**         | Q6_K       | 14.06 GB         | Sehr hohe Qualität, fast perfekt.                                                                  | Gut geeignet, leicht übertrieben     |
| **DeepSeek-Coder-V2-Lite-Instruct-Q5_K_L.gguf**       | Q5_K_L     | 12.37 GB         | Experimentell, verwendet f16 für Einbettungs- und Ausgabewichtungen. Hohe Qualität, empfohlen.       | Sehr gut geeignet                    |
| **DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf**       | Q5_K_M     | 11.85 GB         | Hohe Qualität, empfohlen.                                                                           | Sehr gut geeignet                    |
| **DeepSeek-Coder-V2-Lite-Instruct-Q5_K_S.gguf**       | Q5_K_S     | 11.14 GB         | Hohe Qualität, empfohlen.                                                                           | Sehr gut geeignet                    |
| **DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L.gguf**       | Q4_K_L     | 10.91 GB         | Experimentell, verwendet f16 für Einbettungs- und Ausgabewichtungen. Gute Qualität, empfohlen.       | Sehr gut geeignet                    |
| **DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf**       | Q4_K_M     | 10.36 GB         | Gute Qualität, empfohlen.                                                                           | Sehr gut geeignet                    |
| **DeepSeek-Coder-V2-Lite-Instruct-Q4_K_S.gguf**       | Q4_K_S     | 9.53 GB          | Etwas niedrigere Qualität, aber mehr Speicherersparnis. Empfohlen.                                  | Sehr gut geeignet                    |
| **DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf**       | IQ4_XS     | 8.57 GB          | Anständige Qualität, kleiner als Q4_K_S mit ähnlicher Leistung. Empfohlen.                           | Sehr gut geeignet                    |
| **DeepSeek-Coder-V2-Lite-Instruct-Q3_K_L.gguf**       | Q3_K_L     | 8.45 GB          | Niedrigere Qualität, aber nutzbar, gut für geringe RAM-Verfügbarkeit.                                | Sehr gut geeignet                    |
| **DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf**       | Q3_K_M     | 8.12 GB          | Noch niedrigere Qualität.                                                                          | Sehr gut geeignet                    |
| **DeepSeek-Coder-V2-Lite-Instruct-IQ3_M.gguf**        | IQ3_M      | 7.55 GB          | Mittel-niedrige Qualität, neue Methode mit anständiger Leistung, vergleichbar mit Q3_K_M.           | Sehr gut geeignet                    |
| **DeepSeek-Coder-V2-Lite-Instruct-Q3_K_S.gguf**       | Q3_K_S     | 7.48 GB          | Niedrige Qualität, nicht empfohlen.                                                                 | Gut geeignet                        |
| **DeepSeek-Coder-V2-Lite-Instruct-IQ3_XS.gguf**       | IQ3_XS     | 7.12 GB          | Niedrigere Qualität, neue Methode mit anständiger Leistung, leicht besser als Q3_K_S.               | Gut geeignet                        |
| **DeepSeek-Coder-V2-Lite-Instruct-IQ3_XXS.gguf**      | IQ3_XXS    | 6.96 GB          | Niedrigere Qualität, neue Methode mit anständiger Leistung, vergleichbar mit Q3-Quantisierungen.    | Gut geeignet                        |
| **DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf**         | Q2_K       | 6.43 GB          | Sehr niedrige Qualität, aber überraschend nutzbar.                                                   | Gut geeignet                        |
| **DeepSeek-Coder-V2-Lite-Instruct-IQ2_M.gguf**        | IQ2_M      | 6.32 GB          | Sehr niedrige Qualität, nutzt SOTA-Techniken, ebenfalls überraschend nutzbar.                        | Gut geeignet                        |
| **DeepSeek-Coder-V2-Lite-Instruct-IQ2_S.gguf**        | IQ2_S      | 6.00 GB          | Sehr niedrige Qualität, nutzt SOTA-Techniken, nutzbar.                                               | Gut geeignet                        |
| **DeepSeek-Coder-V2-Lite-Instruct-IQ2_XS.gguf**       | IQ2_XS     | 5.96 GB          | Sehr niedrige Qualität, nutzt SOTA-Techniken, nutzbar.                                               | Gut geeignet                        |

### Fazit:
- **Modell-Empfehlungen:** Die Modelle mit weniger als 10 GB VRAM (wie IQ4_XS, IQ3_M, IQ2_XS) sind hervorragend für die RTX 4090 geeignet, ohne den VRAM zu überlasten. Modelle wie Q8_0 oder Q6_K_L liefern extreme Qualität, aber könnten die GPU unnötig stark auslasten, wenn du eine leichtere Nutzung anstrebst.



<br><br>

### Download
- https://huggingface.co/bartowski/DeepSeek-Coder-V2-Instruct-GGUF
```shell
# q8
huggingface-cli download bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF --include "DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf" --local-dir "/home/userName/Projects/ai/resources/models/llm/deepseek/Coder V2 Lite"

# q6
huggingface-cli download bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF --include "DeepSeek-Coder-V2-Lite-Instruct-Q6_K.gguf" --local-dir "/home/userName/Projects/ai/resources/models/llm/deepseek/Coder V2 Lite"
```


# Details
```
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.name str              = DeepSeek-Coder-V2-Lite-Instruct
llama_model_loader: - kv   2:                      deepseek2.block_count u32              = 27
llama_model_loader: - kv   3:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   4:                 deepseek2.embedding_length u32              = 2048
llama_model_loader: - kv   5:              deepseek2.feed_forward_length u32              = 10944
llama_model_loader: - kv   6:             deepseek2.attention.head_count u32              = 16
llama_model_loader: - kv   7:          deepseek2.attention.head_count_kv u32              = 16
llama_model_loader: - kv   8:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv   9: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  10:                deepseek2.expert_used_count u32              = 6
llama_model_loader: - kv  11:                          general.file_type u32              = 18
llama_model_loader: - kv  12:        deepseek2.leading_dense_block_count u32              = 1
llama_model_loader: - kv  13:                       deepseek2.vocab_size u32              = 102400
llama_model_loader: - kv  14:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  15:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  16:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  17:       deepseek2.expert_feed_forward_length u32              = 1408
llama_model_loader: - kv  18:                     deepseek2.expert_count u32              = 64
llama_model_loader: - kv  19:              deepseek2.expert_shared_count u32              = 2
llama_model_loader: - kv  20:             deepseek2.expert_weights_scale f32              = 1.000000
llama_model_loader: - kv  21:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  22:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  23:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  24: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  25: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.070700
llama_model_loader: - kv  26:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  27:                         tokenizer.ggml.pre str              = deepseek-llm
llama_model_loader: - kv  28:                      tokenizer.ggml.tokens arr[str,102400]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  29:                  tokenizer.ggml.token_type arr[i32,102400]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  30:                      tokenizer.ggml.merges arr[str,99757]   = ["Ġ Ġ", "Ġ t", "Ġ a", "i n", "h e...
llama_model_loader: - kv  31:                tokenizer.ggml.bos_token_id u32              = 100000
llama_model_loader: - kv  32:                tokenizer.ggml.eos_token_id u32              = 100001
llama_model_loader: - kv  33:            tokenizer.ggml.padding_token_id u32              = 100001
llama_model_loader: - kv  34:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  35:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  36:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  37:               general.quantization_version u32              = 2
llama_model_loader: - kv  38:                      quantize.imatrix.file str              = /models/DeepSeek-Coder-V2-Lite-Instru...
llama_model_loader: - kv  39:                   quantize.imatrix.dataset str              = /training_data/calibration_datav3.txt
llama_model_loader: - kv  40:             quantize.imatrix.entries_count i32              = 293
llama_model_loader: - kv  41:              quantize.imatrix.chunks_count i32              = 139
```
- **27 layer (deepseek2.block_count)**

<br><br>

### Run
```shell
cd /home/t33n/Projects/ai/LLM/RUNTIME/llama.cpp/build/bin/

# g6
./llama-cli -m '/home/t33n/Projects/ai/resources/models/llm/deepseek/Coder V2 Lite/DeepSeek-Coder-V2-Lite-Instruct-Q6_K.gguf' -p "Create express.js hello world project" -no-cnv -ngl 27

# g8
./llama-cli -m '/home/t33n/Projects/ai/resources/models/llm/deepseek/Coder V2 Lite/DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf' -p "Create express.js hello world project" -no-cnv -ngl 27
```
- With RTX 4090 we can use alle 27 layer






  
</details>





















<br><br>
<br><br>
___
___
<br><br>
<br><br>





# DeepSeek-Coder-V2-Instruct



<br><br>

## DeepSeek-Coder-V2-Instruct-GGUF (llama.cpp)
- Too big for RTX 4090

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



</details>
