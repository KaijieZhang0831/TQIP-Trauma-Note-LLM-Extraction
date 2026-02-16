# TQIP-TRAUMA-NOTE-Extraction
LLMs offer a potential solution to streamline this process. We hypothesized that a LLM could be applied to review patient charts and identify complications as defined by the Trauma Quality Improvement Program, offering an effective adjunct to manual chart reviews. This is to a complication screening pipeline driven by LLMs that can support clinical staff to construct TQIP Trauma Note Complication Reports.

## Data
Because real clinical data, especially trauma notes and medication orders, are protected and sensitive, all of our data, the patient features, were stored in monitored and protected environment. The Prompt with CoT were created based on the [National Trauma Data Standard Data Dictionary 2025 Admission](https://health.wyo.gov/wp-content/uploads/2025/01/2025-Data-Dictionary.pdf) as the instruction.

The LLM we used is "us.deepseek.r1-v1:0" and the embedding we used is "amazon.titan-embed-text-v2:0" via Amazon Bedrock.


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
```
python3 main.py
```
```
python3 eval_test.py
```
