# LLM_Token_Limitations
Testing LLM accuracy when adding token limitations

## Main Process
1. Download data with an initial schema
2. Transform into a unified format
3. Extract features (ex. number of tokens)
4. Explore datasets in terms of data distribution 
5. Refine schema based on eda
6. Get balanced samples
7. Run experiments
8. Analyze findings



### SAMPLING:
We’ll do stratified sampling per dataset using:

GSM8K
total_tokens_estimate (short / medium / long)
reasoning_steps (low / medium / high)

StrategyQA
total_tokens_estimate
reasoning_steps

TruthfulQA
total_tokens_estimate
num_incorrect_answers

# Experiment
Store "token_limits": [0.25, 0.5, 0.75, 1.0]
Then apply truncation dynamically during inference.

# Strategy/ Thesis Alternatives:
Role isolation experiments
1. Generator only: (no verification)
2. Generator + verifier (same model)
3. Generator + verifier (different model)

