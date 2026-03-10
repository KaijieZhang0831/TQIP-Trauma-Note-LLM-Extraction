# Large Language Models for Trauma Care Quality: NTDS Complication Abstraction

## Background & Research Question

• Manual abstraction is slow and inconsistent when applied to long, noisy trauma notes, and the professional staff required for this task are costly.

Our Solution: LLMs offer a potential solution to streamline this process. We hypothesized that a LLM could be applied to review patient charts and identify complications as defined by the Trauma Quality Improvement Program, offering an effective adjunct to manual chart reviews. This is to a complication screening pipeline driven by LLMs that can support clinical staff to construct TQIP Trauma Note Complication Reports.

• Key constraint: secure setting, no external online APIs beyond AWS Bedrock, and no local GPU deployment.

• We compare inference-time strategies under the same retrieval pipeline.

• We ask whether an LLM can detect NTDS complications with auditable note evidence.

<p align="center">
  <table>
    <tr>
      <td align="center"><img src="image/DSC_Final_Pitch_TQIP_B08_3.png" alt="Overview" width="1000"/></td>
    </tr>
  </table>
</p>

Website Link: https://kaijiezhang0831.github.io/TQIP-Trauma-Note-LLM-Extraction/
## Data

## Data Collection and Access
Because real clinical data, especially trauma notes and medication orders, are protected and sensitive, all of our data, the patient features, were stored in monitored and protected environment. The Prompt with CoT were created based on the [National Trauma Data Standard Data Dictionary 2025 Admission](https://health.wyo.gov/wp-content/uploads/2025/01/2025-Data-Dictionary.pdf) as the instruction. Note: Other clinical notes dataset such as MIMIC-III won't be able to apply to this case because they serve different purposes. We cannot provide any way to access the raw data. The raw data is belong to UCSD Health and is under protection according to local policies. 

However, we can provide a platform of required data files as a reference, and processed log examples during experiments. These references can help you better understand the data & research.

The LLM we used is "us.deepseek.r1-v1:0" and the embedding we used is "amazon.titan-embed-text-v2:0" via Amazon Bedrock.

### Required file 1: patient_features
`patient_features.json` is required for the experinment, where Notes were retrieved using FHIR R4. A sample format is below. Make sure your format align for successful testing:
<pre>
  <code>
[
  {
    "csn": "CSN_000001",
    "patient_id": "P_000001",
    "admit_datetime": "2026-02-01T10:15:00",
    "discharge_datetime": "2026-02-07T14:30:00",

    "binary": [
      {
        "note_id": "N1",
        "note_type": "ED_PROVIDER_NOTE",
        "timestamp": "2026-02-01T10:30:00",
        "note": "ED TRAUMA H&P: ... full text ..."
      },
      {
        "note_id": "N2",
        "note_type": "ICU_PROGRESS_NOTE",
        "timestamp": "2026-02-02T08:00:00",
        "note": "ICU DAY 1 PROGRESS NOTE: ... full text ..."
      },
      {
        "note_id": "N3",
        "note_type": "DISCHARGE_SUMMARY",
        "timestamp": "2026-02-07T13:00:00",
        "note": "DISCHARGE SUMMARY: ... full text ..."
      }
    ],

    "medication_orders": [
      {
        "order_id": "M1",
        "timestamp": "2026-02-02T09:10:00",
        "dosage_text": "Heparin 5000 units SQ q8h"
      },
      {
        "order_id": "M2",
        "timestamp": "2026-02-03T12:20:00",
        "dosage_text": "Cefepime 2 g IV q8h"
      }
    ],

    "labels": {
      "aki": 0,
      "aws": 1,
      "ards": 0,
      "cardiac_arrest_cpr": 0,
      "cauti": 0,
      "delirium": 1,
      "dvt": 0,
      "mi": 0,
      "osteomyelitis": 0,
      "pressure_ulcer": 0,
      "pe": 0,
      "severe_sepsis": 0,
      "stroke_cva": 0,
      "superficial_ssi": 0,
      "unplanned_icu_admission": 0,
      "unplanned_intubation": 0,
      "unplanned_or_visit": 0,
      "vap": 0
    }
  }
]
    </code>
</pre>

### Required file 2: ground_truth.csv
`ground_truth.csv` is required for the experinment. This is constructed by professional clinical staff. A sample format is below. Make sure your format align for successful testing:
<pre>
  <code>
csn,aki,aws,ards,cardiac_arrest_cpr,cauti,delirium,dvt,mi,osteomyelitis,pressure_ulcer,pe,severe_sepsis,stroke_cva,superficial_ssi,unplanned_icu_admission,unplanned_intubation,unplanned_or_visit,vap
CSN_000001,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
CSN_000002,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0
    </code>
</pre>

## Directory structure Explaination
<pre>
  <code>
    📁 Project Root
      ├── README.md
      ├── main_baseline.py
      ├── main_CoT.py
      ├── main_Beam.py
      ├── main_DVTS.py
      ├── eval_test.py
      ├── environment.yml
      ├── archive
      ├── data
      │   ├── ground_truth.csv
      │   ├── features_folder
      │   │   ├── patient_features.json
      │   │   └── [...]
      │   └── prompt
      │       ├── CoT_prompt.json
      │       └── atmoic_prompt.json
      ├── other_tools
      │   ├── data_manipulation.ipynb
      │   ├── preprocess_export.py
      │   ├── refine.py
      │   ├── test.py
      │   └── visualization.ipynb
      ├── complication_results_folder
      │   └── complication_results_wt_test1
      │   │   ├── vote_log.csv
      │   │   ├── eval_per_complication.json
      │   │   ├── eval_individual.csv
      │   │   ├── timing_summery.json
      │   │   ├── [patient1_mrn].json
      │   │   └── [...]
      │   ├── complication_results_wt_test2
      │   └── [...]
      ├── complication_results_wt_test
      │   ├── [csn1].json
      │   ├── [csn2].json
      │   └── [...]
      └── question_dict.py
    </code>
</pre>

## Introduction to the Experinment
Our project explores whether large language models can support trauma registry abstraction by identifying the 18 NTDS-defined complications from clinical documentation. Because protected trauma notes cannot be used directly for development, we built an end-to-end pipeline that generates realistic synthetic notes with ground-truth labels and evaluates an LLM-based extractor. We report initial benchmarking results and outline improvements to retrieval, chunking, and decision voting to close the gap between controlled tests and real-world performance.

<p align="center">
  <table>
    <tr>
      <td align="center"><img src="image/Intro_Pipe.jpg" alt="Overview" width="1000"/></td>
    </tr>
  </table>
</p>

## How to Use?
### 1. Environment set up
We strongly recommand using Linux for the experinment. To set up the conda environment for Linux, type
```
python3 -m pip install -U pip
```
```
pip install -r requirements_local_llm_py39.txt
```
in terminal. Then, activate it.

We also recommand using tmux to assit your long-time running:
```
sudo yum install -y tmux
```

### 2. Run and Evaluate
Make sure `patient_features.json` is in the root folder with correct format. Simply run,
```
python3 main_baseline.py
```
in the terminal. It will take some time and you can review the progress by the log. More optional are avaliable, such as:
```
python3 main_baseline | tee run_test.log
```

After that, you will get a new folder in the root named `complication_results_wt_xxx/`. They are the result/prediction your model made. For the evaluation step, make sure `ground_truth.csv` is in the root folder with correct format and the selected `complication_results_wt_xxx/` path is correct in `eval_test.py`. Then run,
```
python3 eval_test.py
```
or optionally,
```
python3 eval_test.py | tee eval_test.log
```
#### Run experinment with different structures
To test Best-of-N, Beam Search, and DVTS, you are welcome to run:
```
python3 main_CoT.py | tee run_test.log
```
```
python3 main_Beam.py | tee run_test.log
```
```
python3 main_DVTS.py | tee run_test.log
```
, which have file names quite straightforward. 

#### File path management
To make results from multiple experiments managable, you should edit file path for different experinments. Here are some significant file paths:
main - features input:
```
fhir_data = pd.read_json("./features_folder/patient_features.json")
```
main - result output:
```
output_dir = 'complication_results_folder/complication_results_holistic'
```
eval - read output:
```
output_dir = 'complication_results_folder/complication_results_holistic'
```


## Experiment Method
We evaluate an LLM-based extractor that predicts 18 NTDS-defined trauma complications from each encounter’s clinical documentation. For each case, we aggregate all available note text into a single input and split it into overlapping chunks to support retrieval. We embed chunks, build a per-case vectorstore, and retrieve the most relevant evidence for each complication before prompting the LLM to output a binary decision. To improve robustness, we run multiple independent LLM calls per complication and apply threshold voting to produce the final label. We compute TP/FP/FN/TN across all complications and report sensitivity, PPV, and NPV at both the per-complication and overall levels. We also profile runtime by module to identify the dominant latency contributors and prioritize optimization.


## Expected outputs: Evaluation and Visualization

After evaluation test, you might see detailed the performance matrix inside of log files in the root and selected output path, isupporting calculation of Sensitivity, PPV, and NPV if you are interested. For details, please check sample files provided. Also, these tables are something you can derived from them:

Note: Sensitive evaluation sections (e.g. those containing patients' mrn/csn) are excluded.
### Table 1: Complication-level performance summary

| Complication | Sensitivity | PPV | NPV |
| --- | ---: | ---: | ---: |
| Alcohol Withdrawal Syndrome | 0.429 | 0.750 | 0.840 |
| Delirium | 0.895 | 0.370 | 0.951 |
| DVT/Thrombophlebitis | 0.625 | 0.625 | 0.940 |
| Stroke/CVA | 0.733 | 0.314 | 0.971 |
| Unplanned Intubation | 0.889 | 0.267 | 0.982 |
| Unplanned Admission to ICU | 0.750 | 0.340 | 0.950 |
| Severe Sepsis | 1.000 | 0.262 | 1.000 |
| Pressure Ulcer | 0.714 | 0.128 | 0.985 |
| Cardiac Arrest with CPR | 1.000 | 0.500 | 1.000 |
| Acute Kidney Injury | 0.778 | 0.206 | 0.986 |
| Unplanned Visit to OR | 1.000 | 0.224 | 1.000 |
| Pulmonary Embolism | 0.500 | 0.040 | 0.993 |
| Myocardial Infarction | 0.667 | 0.111 | 0.994 |
| VAP | 0.333 | 0.167 | 0.947 |
| ARDS | 0.800 | 0.471 | 0.987 |
| CAUTI | 0.500 | 0.154 | 0.988 |
| Osteomyelitis | 1.000 | 0.400 | 1.000 |
| Superficial Incisional SSI | 0.667 | 0.111 | 0.994 |

### Table 2: Results by task, including Alcohol Withdrawal Syndrome

| Experiment | Sens. (%) | N (%) | NPV (%) | PPV (%) | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original (No CoT, 5-vote majority) | 85.71 | 250.00 | 98.97 | 26.47 | 1425.93 |
| Best-of-N using CoT (3 candidates) | 61.90 | 140.00 | 97.49 | 31.71 | 1476.67 |
| Beam Search using atomic decision tree (3 candidates, beam width = 3) | 38.10 | 65.00 | 96.17 | 38.10 | 1292.40 |

### Table 3: Results by task, not including Alcohol Withdrawal Syndrome

| Experiment | Sens. (%) | N (%) | NPV (%) | PPV (%) | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original (No CoT, 5-vote majority) | 85.71 | 314.29 | 99.30 | 21.43 | 1425.93 |
| Best-of-N using CoT (3 candidates) | 78.57 | 200.00 | 99.00 | 28.21 | 1476.67 |
| Beam Search using atomic decision tree (3 candidates, beam width = 3) | 57.14 | 85.71 | 98.13 | 40.00 | 1292.40 |

In addition, in tools folder, you are welcome to use some sections written in `visualization.ipynb` to produce visualizations presented in our poster.


## Contribution
- Kaijie Zhang: I was responsible for the model inference pipeline, implementation, and experimental evaluation of the different test-time methods, excluding the filter optimization component. I also led all visualization work and created the poster, including the figures, result presentation, and overall visual layout.

- Viv Somani:  I created an improved scheme to perform initial regular expressions chunk filtering. In addition, I wrote the introduction and methods section -- and assisted in writing several other sections of this report.
