# TQIP-TRAUMA-NOTE-Extraction
This repository creates a baseline RAG pipeline to test the ability of an LLM to summarize medical notes in the [NTDS-18](https://www.facs.org/quality-programs/trauma/quality/national-trauma-data-bank/national-trauma-data-standard/) Benchmark. The NTDS-18 benchmark is used by TQIP to evaluate hospital performance. The notes generated in data/syntheic_ntds_trauma_notes_gemini.csv are synthetic notes produced by generator (part 1 of our pipeline). These were generated with the help of an LLM. data/ntds_18_complications.json provides an overview of all the complications tested for within the NTDS-18 dataset.

## Data
Because real clinical data, especially trauma notes and medication orders, are protected and sensitive, all of our data, the patient features, were stored in monitored and protected environment. The Prompt with CoT were created based on the [National Trauma Data Standard Data Dictionary 2025 Admission](https://health.wyo.gov/wp-content/uploads/2025/01/2025-Data-Dictionary.pdf) as the instruction.


## File Structure
<pre>
  <code>
    📁 Project Root
      ├── README.md
      ├── main.py
      ├── preprocess_export.py
      ├── refine.py
      ├── eval_test.py
      ├── test.py
      ├── environment.yml
      ├── archive
      ├── complication_results_wt_test
      │   └── [csn1].json
      │   └── [csn2].json
      │   └── [...]
      └── notebooks.ipynb
    </code>
</pre>

## Introduction

<p align="center">
  <table>
    <tr>
      <td align="center"><img src="image/Intro_Pipe.jpg" alt="Overview" width="1000"/></td>
    </tr>
  </table>
</p>

## Hot to Use
1. Environment set up
We strongly recommand using Linux for the experinment. To set up the conda environment for Linux, type
```
python3 -m pip install -U pip
```
```
pip install -r requirements_local_llm_py39.txt
```
in terminal.Then, activate it.

We also recommand using tmux to assit your long-time running:
```
sudo yum install -y tmux
```

2. Run and Evaluate



## Experiment Method


## Result



## Contribution
- Kaijie Zhang: 

- Viv Somani: 