# TGEA 2.0

Datasets and codes for the paper "TGEA 2.0: A Large-Scale Diagnostically Annotated
Dataset with Benchmark Tasks for Text Generation of
Pretrained Language Models".
 
## Data License

Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license. 
(License URL: https://creativecommons.org/licenses/by-sa/4.0/)

## Quick Start


### Data Preprocessing
Converting raw data to the format of each task
```
unzip data.zip
python data/convert_raw_data_to_benchmarks.py 
python data/convert_gec_format.py
```
### Benchmarks

1. Erroneous Text Detection
```
sh Diagnosis_tasks/train_b1.sh
```
2. MiSEW Extraction
```
sh Diagnosis_tasks/train_b2.sh
```
3. Erroneous Span Location
```
sh Diagnosis_tasks/train_b3.sh
```
4. Error Type Classification
```
sh Diagnosis_tasks/train_b4.sh
```
5. Error Correction
```
sh Diagnosis_tasks/train_b5.sh
```

[m2scorer](https://github.com/nusnlp/m2scorer/) is used to evaluate results of error correction.

6. Generation Pathology Mitigation
```
sh Generation_Pathology_Mitigation/train_b6.sh
python Generation_Pathology_Mitigation/evaluate.py
```
