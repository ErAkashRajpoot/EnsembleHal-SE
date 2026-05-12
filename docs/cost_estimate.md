# Cost Estimate Template

Assumptions:
- k = 5 samples per task per model
- One greedy sample (temperature = 0.0) stored separately
- Average output length: TODO tokens
- Average input length: TODO tokens

Formula:
Total output tokens = tasks * k * avg_output_tokens
Total input tokens = tasks * k * avg_input_tokens
Estimated cost = (total_input_tokens / 1,000,000) * input_cost_per_1m + (total_output_tokens / 1,000,000) * output_cost_per_1m

## Cost Table (Fill in)

| Model | Provider | Input $/1M | Output $/1M | Avg input tokens | Avg output tokens | Tasks | k | Est. cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-4o-mini | openai | TODO | TODO | TODO | TODO | TODO | 5 | TODO |
| gemini-1.5-flash | google | TODO | TODO | TODO | TODO | TODO | 5 | TODO |
| llama-3-70b | groq/together | TODO | TODO | TODO | TODO | TODO | 5 | TODO |
| deepseek-coder-v2 | together | TODO | TODO | TODO | TODO | TODO | 5 | TODO |
| qwen2.5-coder-7b | together | TODO | TODO | TODO | TODO | TODO | 5 | TODO |

Notes:
- Update the pricing columns once provider pricing is finalized.
- If any model is free-tier, set cost to 0 and record usage limits.
