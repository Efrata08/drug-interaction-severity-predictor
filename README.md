# Drug Interaction Severity Predictor

A machine learning model that predicts whether a combination of two medications will produce a Minor, Moderate, or Major adverse interaction — built to assist pharmacists and prescribing physicians in flagging dangerous drug combinations before a patient ever takes them.

## Research Question

Can a machine learning model accurately predict the severity of a drug-drug interaction — classified as Minor, Moderate, or Major — using pharmacological features including drug category, metabolizing enzymes, and molecular targets, and does model performance vary significantly across different drug categories?

## Background

Adverse drug reactions caused by drug-drug interactions result in over 1.25 million serious adverse events and nearly 175,000 deaths in the United States annually (FDA FAERS, 2022). Existing pharmacy systems rely on static, label-based alert checkers that generate so many warnings that healthcare providers begin ignoring them — a phenomenon known as alert fatigue. This project proposes a data-driven alternative trained on real clinical interaction data.

## Datasets

| Dataset | Source | Description |
|--|--|--|
| DDInter | [ddinter.scbdd.com](http://ddinter.scbdd.com) | ~240,000 curated drug-drug interaction records across 8 pharmacological categories, labeled Minor / Moderate / Major |
| DrugBank | [Kaggle](https://www.kaggle.com/datasets/devildev89/drug-bank-5110) | Pharmacological features for 15,000+ drugs including drug category, mechanism of action, metabolizing enzymes, and molecular targets |

## Methodology

- Merged DDInter interaction severity labels with DrugBank pharmacological features for both drugs in each pair
- Applied balanced sampling — 200 rows per severity class per drug category — to correct class imbalance
- Combined structured features (ATC codes, approval status, shared CYP enzymes) with TF-IDF vectorized text features from mechanism, metabolism, and target descriptions
- Trained and compared three models: Random Forest (primary), XGBoost (comparison), and KNN (baseline)
- Evaluated bias separately across all 8 DDInter drug categories

## Results

| Model | Accuracy | Major Recall | Minor Recall | Moderate Recall |
|--|--|--|--|--|
| Random Forest | 79% | 82% | 87% | 67% |
| XGBoost | 77% | 80% | 85% | 66% |
| KNN Baseline | 59% | 69% | 66% | 41% |

Random Forest was selected as the primary model. Only 5 out of 240 Major interactions in the test set were misclassified as Minor — a 2% catastrophic miss rate.

## Bias Evaluation

| Category | Accuracy | Major Recall | Moderate Recall |
|--|--|--|--|
| A — Digestive/Diabetes | 77.5% | 69.2% | 75.0% |
| B — Blood drugs | 82.9% | 92.0% | 72.0% |
| D — Dermatology | 72.7% | 84.6% | 51.5% |
| H — Hormonal | 79.3% | 89.2% | 70.0% |
| L — Cancer/Immune | 75.0% | 75.0% | 58.1% |
| P — Antiparasitic | 83.0% | 81.2% | 70.8% |
| R — Respiratory | 78.4% | 75.0% | 71.4% |
| V — Various | 80.0% | 85.3% | 70.6% |

Blood drugs (B) performed strongest at 92% Major recall. Diabetes/digestive drugs (A) had the lowest Major recall at 69.2% — the most clinically significant gap.

## Key Findings

- Drug category (ATC codes) was the single strongest predictor of interaction severity
- CYP3A4 enzyme involvement, receptor binding, and metabolite pathways appeared in the top 20 most important features — confirming the model learned genuine pharmacological patterns
- Moderate interactions were consistently the hardest class to predict across all categories, reflecting the biological ambiguity of the Minor-Moderate boundary

## Limitations

- Mechanism, metabolism, and target text data was unavailable for a subset of drug pairs due to incomplete DrugBank coverage
- The model generalizes at the drug category level — predictions for entirely novel drug classes with no comparable ATC category would have lower confidence
- DDInter and DrugBank primarily reflect Western pharmaceutical data, which may limit generalizability to drug combinations more common in other regions

## How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/drug-interaction-severity-predictor.git
cd drug-interaction-severity-predictor

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebooks/drug_interaction_model.ipynb
```

Or open directly in Google Colab by clicking the notebook file in the repo and selecting **Open in Colab**.

## Team

| Name | Institution |
|--|--|
| Kiran Thamida |
| Efrata Getachew Bogale | 
| Saron Nigussie | 
| Abdul Ahad | 
| Fariha Tasnim Amir | 

## Citations

1. Xiong, G., Yang, Z., Yi, J., et al. (2022). DDInter: an online drug–drug interaction database towards improving clinical decision-making and patient safety. *Nucleic Acids Research*, 50(D1), D1200–D1207.
2. Yan, Z., Fan, K., Yu, T., et al. (2025). Polypharmacy, drug–drug interactions and adverse drug reactions in older Chinese cancer patients. *Frontiers in Pharmacology*, 16, 1579023.
3. StatPearls. Adverse Drug Reactions. National Library of Medicine. https://www.ncbi.nlm.nih.gov/books/NBK599521/
4. U.S. Food and Drug Administration. Preventable Adverse Drug Reactions: A Focus on Drug Interactions. https://www.fda.gov/drugs/drug-interactions-labeling/preventable-adverse-drug-reactions-focus-drug-interactions
5. Lu, Y., Chen, J., Fan, N., et al. (2026). Machine learning models for drug-drug interaction prediction. *npj Digital Medicine*, 9, 198.

## Acknowledgements

Built as part of the AI4ALL Ignite Program, Summer 2026.
