# Stage 3 All Quant Benchmark Summary

| quant_method | rows | avg_ttft_ms | avg_total_latency_ms | avg_decode_tokens_per_s | avg_request_tokens_per_s | avg_peak_gpu_mem_mb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 9 | 33.557 | 1841.770 | 35.406 | 34.760 | 6009.961 |
| bnb_int8 | 9 | 244.580 | 12975.846 | 4.948 | 4.855 | 3398.132 |
| bnb_int4 | 9 | 59.987 | 2428.496 | 27.026 | 26.358 | 2140.986 |
| awq | 9 | 55.881 | 3022.898 | 19.577 | 19.208 | 1984.557 |
| gptq | 9 | 95.042 | 4868.792 | 13.408 | 13.147 | 2095.575 |
