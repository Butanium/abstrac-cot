## Installation

```bash
conda create -n abs_cot
conda activate abs_cot
pip install -r requirements.txt
```

## Datasets
- HotpotQA
- StrategyQA
- OpenBookQA
- TruthfulQA

## Reproducing the results

### Prompt generation:
```bash
git submodule init
git submodule update
python prompt_generation.py
```