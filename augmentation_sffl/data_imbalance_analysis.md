## Data Imbalance Analysis - Feature Envy Detection

### Observations:
- Project: **activemq**
- Total Methods: 15,482
- Smelly Methods (label=1): 585
- Normal Methods (label=0): 14,897

### Problem:
Only **3.9%** of methods are labeled as smelly. This high imbalance causes the model to be biased toward the majority (normal) class, leading to **low recall** and difficulty detecting minority class examples.



### Goal:
To address this imbalance, we propose and test **data augmentation strategies** such as:
- **Naive oversampling** of minority class (as baseline)
- **GraphSMOTE** (as used in SCG)
- Possibly **GraphMix** or **edge perturbation** as additional ideas

This step documents the core limitation and gives us a baseline to improve on.
