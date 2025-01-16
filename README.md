# deepseek-cheat-sheet





















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
huggingface-cli download bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF --include "DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf" --local-dir "/home/userName/Projects/ai/resources/models/llm/deepseek"
```





  
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

