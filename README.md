# difficulty_controllable_MWP_gen
Repository for AIED2023 paper Automatic Educational Question Generation with Diﬃculty Level Controls

## Processed datasets
You can find preprocessed datasets in data. data/AsDiv_template_mod.jsonl and data/GSM8K_template_mod.jsonl contain basic-level seed problems; data/MathQA_template.jsonl contains advanced-level seed problems. 

## Run generation

All the expert-model checkpoints for energy computation are available [here](https://polybox.ethz.ch/index.php/s/fMnmfQyrTBxmrlG)

- Basic topic transfer
```
bash bash/topic transfer/b_dm.sh
bash bash/topic transfer/b_dp.sh
```

- Basic text rewriting
```
bash bash/text rewriting/b_d_base.sh
bash bash/text rewriting/b_m_base.sh
bash bash/text rewriting/b_p_base.sh
```

-Advanced text rewriting
```
bash bash/text rewriting/a_d.sh
bash bash/text rewriting/a_m.sh
bash bash/text rewriting/a_p.sh
```

## Auto-evaluation

Download eval_bart from [here](https://polybox.ethz.ch/index.php/s/fMnmfQyrTBxmrlG)

```
python evaluation.py
```
