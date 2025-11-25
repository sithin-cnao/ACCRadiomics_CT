# Investigating the Radiomic Performance Gap Driven by Delineation Strategy: Radiotherapy Gross Tumor Volume vs. Dedicated Lesion Segmentation in Proton-Treated Adenoid Cystic Carcinoma

Repository supporting the article submitted to MDPI

If you use this codebase for your research, please cite our paper if available; otherwise, please cite this repository:
```bibtex
TBA
```
### **Repository structure**
#### **Overview:**

* Signature Development - Feature Filtering + Feature Selection using Backward Elimination (with LR, L-SVM, and RF)
* Statistical Analysis - Permutation Test, 95% AUC Confidence Interval using Percentile/BCa Method, DeLong's Test
* Correlation of Selected Signatures
* Visualizations
* Jupyter notebooks with usage examples

#### **Contents:**
```
ACCRadiomics_CT
├── radiomicsFeatures                  # radiomics features extracted from cleaned GTV and TRAD using radiomicsSettingsCT.yaml
│    └── features_GTV.xlsx  
│    └── features_TRAD.xlsx
│    └── radiomicsSettingsCT.yaml
├── results                            # directory containing some important results of the analysis
│    └── boot_dist.npy                 #---- bootstrap distribution  
│    └── perm_dist.npy                 #---- Permutation test null distribution
│    └── selected_signature.npy        #---- selected radiomics signatures
├── main.ipynb                         # Jupyter notebook containing all components involved in the analysis
├── LICENCE                            # MIT License
├── README.md

```

