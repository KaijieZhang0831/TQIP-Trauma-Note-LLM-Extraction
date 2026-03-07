# TQIP Trauma Note LLM Extraction
## Project Overview

This repository contains code for automatically detecting medical complications in trauma patient medical notes using Large Language Models (LLMs). The project is part of the Trauma Quality Improvement Program (TQIP) initiative to automate the abstraction of complications from clinical documentation.

## Important Context

### Data Sensitivity
- **Patient data is NOT included in this repository** - all patient medical notes and identifiers are considered sensitive PHI (Protected Health Information)
- The actual patient data resides in a secure AWS environment
- Only the code for processing the data is stored here

### Development Workflow
1. Code is developed and edited locally (in this repository)
2. Code changes are copied/deployed to the AWS environment where patient data resides
3. All data processing and model inference happens on AWS with access to the secure data

### Infrastructure
- **Current**: Uses AWS Bedrock for LLM inference via boto3
- **Previous**: Ran locally using vLLM on AWS EC2 instances (migrated)

## Repository Structure

```
.
├── main.py                  # Main processing pipeline for complication detection
├── preprocess_export.py     # Preprocesses TQIP case data from CSV exports
├── test.py                  # Evaluation script - compares predictions with ground truth
├── eval_test.py             # Enhanced evaluation with per-complication metrics
├── refine.py                # Post-processing validation of LLM predictions
├── notebook.ipynb           # Jupyter notebook for data exploration and analysis
├── README.md                # Project documentation and usage instructions
└── archive/                 # Older versions of scripts
    ├── lllm.py
    └── test.py
```

## Core Functionality
LLMs easy
### main.py - Main Processing Pipeline

The primary script that orchestrates complication detection:

**Key Components:**
- Uses **AWS Bedrock** with DeepSeek R1 (`us.deepseek.r1-v1:0`) for LLM inference via boto3
- **Amazon Titan Embeddings** (`amazon.titan-embed-text-v2:0`) for semantic search
- **LangChain** for prompt templating and orchestration
- **Chroma vector store** for RAG (Retrieval Augmented Generation)
- **Outlines library** for structured JSON output generation

**Process Flow:**
1. Loads patient medical notes from JSON files (organized by CSN - Contact Serial Number)
2. For each complication category:
   - Filters note chunks using regex patterns to identify potentially relevant sections
   - Uses text splitter to chunk notes into manageable sizes (~650 chars)
   - Creates vector embeddings and stores in Chroma for semantic search
   - Retrieves most relevant note chunks using MMR (Maximal Marginal Relevance)
   - Generates LLM prompts with abstraction instructions and medical documentation
   - Uses structured JSON output to get rationale and yes/no determination
   - Employs ensemble voting (generates 5 responses, requires 2+ "yes" for positive)
3. Saves results as JSON per patient (CSN)

**Complications Detected:**
- Alcohol Withdrawal Syndrome
- Delirium
- DVT/Thrombophlebitis
- Stroke/CVA
- Unplanned Intubation
- Unplanned Admission to ICU
- Severe Sepsis
- Pressure Ulcer
- Cardiac Arrest with CPR
- Acute Kidney Injury
- Unplanned Visit to OR
- Pulmonary Embolism
- Myocardial Infarction
- VAP (Ventilator-Associated Pneumonia)
- ARDS (Acute Respiratory Distress Syndrome)
- CAUTI (Catheter-Associated UTI)
- Osteomyelitis
- Superficial Surgical Site Infection

### preprocess_export.py

Preprocesses raw TQIP case export data:
- Groups cases by REGISTRYNUM
- Aggregates multiple MRNs, CSNs, and complications per patient
- Filters out cases with no valid complications
- Outputs structured ground truth data for evaluation

### test.py - Evaluation & Metrics

Calculates performance metrics by comparing model predictions against ground truth:

**Metrics Calculated:**
- **Sensitivity** (True Positive Rate) per complication and overall
- **Specificity** (True Negative Rate)
- **PPV** (Positive Predictive Value)
- **NPV** (Negative Predictive Value)
- False positives and false negatives per complication
- Processing time statistics

### eval_test.py - Enhanced Evaluation

Advanced evaluation script with detailed per-complication analysis:

**Features:**
- Aggregates all prediction JSON files from output directory
- Computes individual case-level metrics (sensitivity, additional complications %)
- Generates comprehensive per-complication statistics (TP/TN/FP/FN counts)
- Calculates sensitivity, PPV, and NPV for each of the 18 complications
- Identifies top 3 most predicted complications
- Tracks total processing time across all cases
- Outputs three evaluation files:
  - `eval_individual.csv` - Per-case metrics
  - `eval_aggregated.json` - Overall performance summary
  - `eval_per_complication.json` - Detailed per-complication statistics

### refine.py - Post-Processing Validation

Secondary validation step that reviews LLM predictions:
- Takes the rationales generated by main.py
- Has the LLM self-critique whether the rationale aligns with medical documentation
- Filters out predictions based on vague reasoning ("I infer", "I think")
- Outputs refined/validated predictions

### notebook.ipynb - Data Exploration

Jupyter notebook for exploratory data analysis:
- Loads and inspects `patient_features.json` structure
- Analyzes data format and field sizes
- Provides statistical summaries of cases:
  - Number of notes and medications per case
  - Character counts (total, max, median) for notes
  - Note type distributions
  - CSN uniqueness validation
- Implements streaming JSON parser for handling large datasets efficiently
- Useful for understanding data quality and structure before processing

## Data Files (NOT in Repository)

These files exist on AWS but are NOT tracked in git:

- `patient_features.json` - Patient data with structured format:
  - `csn` - Contact Serial Number (case identifier)
  - `patient_id` - Patient identifier
  - `admit_datetime` / `discharge_datetime` - Timestamps
  - `binary` - Array of clinical notes (note_id, note_type, timestamp, note text)
  - `medication_orders` - Array of medications (order_id, timestamp, dosage_text)
  - `labels` - Ground truth binary labels for 18 complications (aki, aws, ards, etc.)
- `ground_truth.csv` - Binary complication labels per CSN (columns: csn, aki, aws, ards, cardiac_arrest_cpr, cauti, delirium, dvt, mi, osteomyelitis, pressure_ulcer, pe, severe_sepsis, stroke_cva, superficial_ssi, unplanned_icu_admission, unplanned_intubation, unplanned_or_visit, vap)
- `cases.csv` - Raw TQIP case export data (legacy format)
- `complication_results_wt_test/*.json` - Per-patient prediction outputs from main.py
- `eval_individual.csv`, `eval_aggregated.json`, `eval_per_complication.json` - Evaluation outputs from eval_test.py

## Technical Details

### Model Configuration

**LLM:**
- **Model**: `us.deepseek.r1-v1:0` via AWS Bedrock
- **Access**: boto3 Bedrock runtime client
- **Temperature**: 0.3
- **Max tokens**: 512
- **Ensemble voting**: 5 independent calls per complication (requires 2+ "yes" for positive)

**Embedding Model:**
- **Model**: `amazon.titan-embed-text-v2:0` via AWS Bedrock
- Used for semantic similarity search in RAG pipeline
- Replaces previous HuggingFace instructor-large embeddings

### Text Chunking
- **Chunk size**: 650 characters
- **Chunk overlap**: 100 characters
- **Separator**: Sentence boundaries (periods)

### Retrieval Strategy
- **Method**: MMR (Maximal Marginal Relevance)
- **Top-k**: 8 chunks retrieved per query
- **Similarity threshold**: 0.8

## Prompt Engineering

The system uses a structured prompt template:
1. **System message**: Defines role as trauma quality abstractor
2. **Abstraction instructions**: Medical documentation for the specific complication
3. **Question**: Specific yes/no question about complication presence
4. **Medical notes**: Retrieved relevant note chunks
5. **Output format**: Structured JSON with rationale and option fields

Key prompt guidance:
- "DO NOT INFER or take a likely guess"
- "Use explicit evidence from the medical notes"
- "Generate clear rationale by thinking step-by-step"

## Performance Benchmarking

**Sample Results** (20 samples using DeepSeek R1 + Amazon Titan Embeddings):
- **Overall Sensitivity**: 85.7% (18/21 true complications detected)
- **Total TP/FP/FN/TN**: 18 / 40 / 3 / 299
- **Average Additional Complications**: 200% (high false positive rate)

**Per-Complication Highlights:**
- Best performance: Cardiac Arrest with CPR (100% Sensitivity, 100% PPV)
- High sensitivity: Delirium (100%), Unplanned Intubation (100%), Unplanned ICU Admission (100%)
- Challenges: DVT/Thrombophlebitis (33.3% sensitivity), several complications with no positive samples in test set

**Time Analysis** (5 samples):
- **Total Runtime**: 865.25 seconds (~173 seconds per case)
- **LLM Engine**: 79.7% of runtime (12.32 sec per call)
- **Vectorstore Build**: 19.3% of runtime (3.09 sec per build)
- **Retriever Invoke**: 0.9% of runtime (0.14 sec per query)
- **Other overhead**: <1%

**Key Observations:**
- LLM inference is the primary bottleneck
- High false positive rate suggests need for improved prompt engineering or post-processing
- Rare complications difficult to evaluate due to limited positive samples

## Known Issues & Considerations

1. **AWS Bedrock Dependency**: Requires AWS credentials and Bedrock model access
2. **File Path Hardcoding**: Paths in eval_test.py assume specific directory structure (e.g., `/home/kaz034/tqip/tqip/ground_truth.csv`)
3. **Regex Filtering**: Case-sensitive regex patterns require careful review
4. **No Caching**: Vector stores are deleted after each query (could be optimized)
5. **Output Directory Naming**: Results stored in timestamped directories like `complication_results_wt_test/`

## Future Improvements

1. **Parameterize Paths**: Make file paths configurable via environment variables (especially in eval_test.py)
2. **Add Error Handling**: More robust exception handling in ask_llm() and data loading
3. **Implement Caching**: Cache embeddings and vector stores for repeated queries
4. **Add Logging**: Enhance logging for better debugging and monitoring
5. **Batch Processing**: Optimize batch sizes based on available compute resources
6. **Model Evaluation**: Test alternative Bedrock models (e.g., Claude models) for comparison
7. **Improve Chunking**: Optimize chunk size and overlap based on performance metrics
8. **Add Configuration Files**: Create config files for model parameters, paths, and experiment settings

## Dependencies

Main Python packages (see `requirements_local_llm_py39.txt` for full list):
- boto3 (for AWS Bedrock API access)
- langchain
- langchain-experimental
- outlines
- pandas
- numpy
- chromadb
- sentence-transformers

**Environment Setup:**
- Use .venv as the environment, with python 3.9+
- Linux environment strongly recommended
- AWS credentials configured for Bedrock access
- Optional: tmux for managing long-running processes

## Running the Code

**Setup:**
```bash
# Install dependencies
python3 -m pip install -U pip
pip install -r requirements_local_llm_py39.txt

# Optional: Install tmux for long-running processes
sudo yum install -y tmux
```

**Execution:**
```bash
# Run main processing pipeline
python3 main.py

# Or with logging
python3 main.py | tee run_test.log

# Run evaluation
python3 eval_test.py

# Or with logging
python3 eval_test.py | tee eval_test.log
```

**Requirements:**
- AWS credentials configured for Bedrock access
- `patient_features.json` in root directory with correct format
- `ground_truth.csv` for evaluation
- Linux environment (strongly recommended)
- Patient data files (NOT included in repository - stored securely on AWS)

## Project Documentation

This CLAUDE.md file provides technical implementation details. For:
- **User-facing documentation and instructions**: See `README.md`
- **TQIP abstraction guidelines**: Refer to [National Trauma Data Standard Data Dictionary 2025](https://health.wyo.gov/wp-content/uploads/2025/01/2025-Data-Dictionary.pdf)
- **Medical complication definitions**: TQIP Data Dictionary and abstraction manuals from the American College of Surgeons

## Contact & Support

For questions about TQIP abstraction guidelines or medical complication definitions, refer to the TQIP Data Dictionary and abstraction manuals provided by the American College of Surgeons.
