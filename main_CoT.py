import vllm
from langchain_experimental.pydantic_v1 import BaseModel
import pandas as pd
from outlines import models, generate
from langchain.prompts import PromptTemplate
import numpy as np
import pdb
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings
import ast
import logging
import time
import json
import os 
import re
import csv

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from bedrock_llm_router import BedrockLLMRouter, extract_json


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import sys

LOG_PATH = os.environ.get("LOG_PATH", "run_test.log")

# Vote log (append-only). We set the actual path after `output_dir` is created.
VOTE_LOG_PATH = None
VOTE_LOG_FIELDS = [
    "ts", "csn", "question_key", "prompt_kind", "subprompt_idx",
    "has_splits", "n_splits", "k_retrieved",
    "samples", "parsed", "yes", "no", "invalid", "rule",
    "json_structured_samples", "json_fallback_samples",
    "decision", "rationale_preview"
]

def append_vote_log(row: dict):
    global VOTE_LOG_PATH
    if not VOTE_LOG_PATH:
        return
    is_new = not os.path.exists(VOTE_LOG_PATH)
    with open(VOTE_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VOTE_LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in VOTE_LOG_FIELDS})
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH)],
    force=True
)


logger = logging.getLogger(__name__)

class RunningStat:
    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.min = float('inf')
        self.max = 0.0

    def add(self, x):
        self.n += 1
        self.total += x
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

    def to_dict(self):
        mean = self.total / self.n if self.n else 0.0
        min_v = 0.0 if self.min == float('inf') else self.min
        return {"count": self.n, "total_sec": self.total, "mean_sec": mean, "min_sec": min_v, "max_sec": self.max}

TIMING = {}
COUNTS = {}
SLOW_STEP_SEC = float(os.environ.get("SLOW_STEP_SEC", "5"))
CURRENT_CSN = None
CURRENT_QKEY = None

def _stat(stage):
    if stage not in TIMING:
        TIMING[stage] = RunningStat()
    return TIMING[stage]

def _count(key, val):
    COUNTS[key] = COUNTS.get(key, 0) + val

def record_timing(stage, dt):
    _stat(stage).add(dt)
    if dt >= SLOW_STEP_SEC:
        prefix = ""
        if CURRENT_CSN is not None:
            prefix += f"csn={CURRENT_CSN} "
        if CURRENT_QKEY is not None:
            prefix += f"q={CURRENT_QKEY} "
        logger.debug(f"SLOW {prefix}{stage} {dt:.3f}s")

def write_timing_summary(path):
    summary = {k: v.to_dict() for k, v in TIMING.items()}
    out = {"timing": summary, "counts": COUNTS}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

BEDROCK_REGION = os.environ.get("AWS_REGION", "us-west-2")
BEDROCK_CFG = Config(connect_timeout=60, read_timeout=840, retries={"max_attempts": 10, "mode": "standard"})
bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION, config=BEDROCK_CFG)

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "meta.llama3-1-8b-instruct-v1:0")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "3"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))


text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=650, chunk_overlap=100, separators=["."]
        )

prompt_question = PromptTemplate(
    template=
"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Trauma quality abstractor. Your task is to review patient's medical notes and answer the given compliance question following the abstraction instructions. You will be given multiple notes. If there is a "yes" answer for any of these notes, return "yes".
DO NOT INFER or take a likely guess. 
Generate clear rationale to your answer by thinking step-by-step.<|eot_id|><|start_header_id|>user<|end_header_id|>

QUESTION: {question}
ABSTRACTION INSTRUCTIONS: {corpus}
MEDICAL NOTES: {context}. 

OUTPUT FORMAT: Return the answer as a JSON object following the format,
{{"rationale": str, "option": "yes" or "no"}}. 

Remember DO NOT INFER or take a likely guess.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    input_variables=["context", "question", "corpus"]
)


class BedrockTitanTextEmbeddingsV2:
    def __init__(self, client, model_id):
        self.client = client
        self.model_id = model_id

    def _embed_one(self, text: str):
        body = json.dumps({"inputText": text})
        resp = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        data = json.loads(resp["body"].read())
        emb = data.get("embedding")
        if emb is None:
            raise ValueError(f"Titan embed response missing 'embedding': keys={list(data.keys())}")
        return emb

    def embed_documents(self, texts):
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text):
        return self._embed_one(text)



embedding = BedrockTitanTextEmbeddingsV2(bedrock, EMBED_MODEL_ID)


STRUCTURED_MODE = os.environ.get("STRUCTURED_MODE", "auto")

_llm_router = BedrockLLMRouter(
    bedrock_client=bedrock,
    model_id=LLM_MODEL_ID,
    temperature=0.3,
    top_p=0.9,
    max_tokens=LLM_MAX_TOKENS,
    max_gen_len=LLM_MAX_TOKENS,
    structured_mode=STRUCTURED_MODE,
)

COT_PROMPT_PATH = os.environ.get("COT_PROMPT_PATH", "CoT_prompt.json")

try:
    with open(COT_PROMPT_PATH, "r") as f:
        COT_PROMPTS = json.load(f)
except Exception:
    COT_PROMPTS = {}

if not isinstance(COT_PROMPTS, dict) or len(COT_PROMPTS) == 0:
    raise ValueError(f"CoT prompts not loaded or empty: {COT_PROMPT_PATH}")

QKEY_TO_COTKEY = {
    "ACUTE KIDNEY INJURY": "Acute Kidney Injury (AKI)",
    "ARDS": "Acute Respiratory Distress Syndrome (ARDS)",
    "ALCOHOL WITHDRAWAL SYNDROME": "Alcohol Withdrawal Syndrome",
    "CARDIAC ARREST WITH CPR": "Cardiac Arrest with CPR",
    "CAUTI": "Catheter-Associated Urinary Tract Infection (CAUTI)",
    "DVT/THROMBOPHLEBITIS": "Deep Vein Thrombosis (DVT)",
    "DELIRIUM": "Delirium",
    "MYOCARDIAL INFARCTION": "Myocardial Infarction (MI)",
    "OSTEOMYELITIS": "Osteomyelitis",
    "PRESSURE ULCER": "Pressure Ulcer",
    "PULMONARY EMBOLISM": "Pulmonary Embolism (PE)",
    "SEVERE SEPSIS": "Severe Sepsis",
    "STROKE/CVA": "Stroke/CVA",
    "SUPERFICIAL INCIS SURG SITE INF": "Superficial Incisional Surgical Site Infection (S/I SSI)",
    "UNPLANNED ADMISSION TO ICU": "Unplanned Admission to the ICU",
    "UNPLANNED INTUBATION": "Unplanned Intubation",
    "UNPLANNED VISIT TO THE OR": "Unplanned Visit to the Operating Room",
    "VAP": "Ventilator-Associated Pneumonia (VAP)",
}

def get_cot_prompt(question_key: str) -> str:
    k = QKEY_TO_COTKEY.get(question_key, question_key)
    p = COT_PROMPTS.get(k, "")
    if isinstance(p, str):
        return p
    return ""

def fill_medical_notes(prompt_template: str, medical_notes: str) -> str:
    out = prompt_template
    out = out.replace("<<MEDICAL_NOTES>>", medical_notes)
    out = out.replace("[MEDICAL_NOTES]", medical_notes)
    out = out.replace("{MEDICAL_NOTES}", medical_notes)
    out = out.replace("{context}", medical_notes)
    return out

LLAMA_SYSTEM_PROPOSER = (
    "You are a Trauma quality abstractor. Your task is to review patient's medical notes and answer the given "
    "compliance question following the abstraction instructions. You will be given multiple notes. If there is a "
    "\"yes\" answer for any of these notes, return \"yes\". DO NOT INFER or take a likely guess."
)

LLAMA_SYSTEM_VERIFIER = (
    "You are a strict verifier. Follow the user instructions exactly and return ONLY valid JSON with no extra text."
)

def wrap_llama_chat(user_body: str, system_msg: str) -> str:
    # Llama chat template wrapper for more stable JSON-only outputs
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        + system_msg
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        + user_body
        + "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


_verifier_router = BedrockLLMRouter(
    bedrock_client=bedrock,
    model_id=LLM_MODEL_ID,
    temperature=0.0,
    top_p=1.0,
    max_tokens=LLM_MAX_TOKENS,
    max_gen_len=LLM_MAX_TOKENS,
    structured_mode=STRUCTURED_MODE,
)

VERIFIER_PROMPT = """You are a strict verifier for medical abstraction outputs.

You will be given:
1) The exact TASK PROMPT used to generate a candidate answer.
2) Several candidate JSON answers.

Your job:
- Score each candidate from 0 to 100.
- Select the single best candidate.

Scoring rules:
- 0 if the candidate is not valid JSON with exactly keys "rationale" and "option", or option is not "yes"/"no".
- Do NOT require any special rationale format. Do NOT penalize missing evidence quotes.
- If a candidate includes quoted text inside its rationale, prefer candidates where each quoted substring appears verbatim in TASK PROMPT MEDICAL NOTES.
- Prefer candidates whose final option is best supported by MEDICAL NOTES and the decision-tree logic in TASK PROMPT.

Return ONLY this JSON:
{{"best_index": <int>, "scores": [<int>], "reason": "<short>"}}

Notes:
- best_index is 0-based.
- scores length must equal number of candidates.

TASK PROMPT:
{task_prompt}

CANDIDATES:
{candidates}
"""

def ask_llm(prompt_text):
    n_tries = 0
    parsed = []

    while True:
        try:
            t0 = time.perf_counter()
            parsed = []
            raw_failures = 0
            structured_ct = 0
            fallback_ct = 0

            for _ in range(N_SAMPLES):
                obj, used_structured, raw_fail = _llm_router.generate_json(prompt_text)
                if used_structured:
                    structured_ct += 1
                else:
                    fallback_ct += 1

                if isinstance(obj, dict) and "option" in obj and "rationale" in obj:
                    parsed.append(obj)
                else:
                    raw_failures += 1

            record_timing("llm_engine", time.perf_counter() - t0)
            break

        except Exception:
            n_tries += 1
            if n_tries > 5:
                return {
                    "rationale": "No valid response",
                    "option": "no",
                    "votes": {
                        "samples": N_SAMPLES,
                        "parsed": 0,
                        "yes": 0,
                        "no": 0,
                        "invalid": N_SAMPLES,
                        "rule": "best_of_n",
                        "tries": n_tries,
                    },
                }

    yes_list = [r for r in parsed if str(r.get("option", "")).lower() == "yes"]
    no_list = [r for r in parsed if str(r.get("option", "")).lower() == "no"]

    votes = {
        "samples": N_SAMPLES,
        "parsed": len(parsed),
        "yes": len(yes_list),
        "no": len(no_list),
        "invalid": N_SAMPLES - len(parsed),
        "rule": "best_of_n",
        "json_structured_samples": structured_ct,
        "json_fallback_samples": fallback_ct,
    }

    if not parsed:
        return {"rationale": "No valid response", "option": "no", "votes": votes}

    best_idx = 0
    if len(parsed) > 1:
        candidates_lines = []
        for i, c in enumerate(parsed):
            candidates_lines.append(f"{i}: {json.dumps(c, ensure_ascii=False)}")
        verifier_prompt = VERIFIER_PROMPT.format(
            task_prompt=prompt_text,
            candidates="\n".join(candidates_lines),
        )

        t1 = time.perf_counter()
        raw, _ = _verifier_router.generate_raw(wrap_llama_chat(verifier_prompt, LLAMA_SYSTEM_VERIFIER))
        record_timing("llm_verifier", time.perf_counter() - t1)

        vobj = extract_json(raw)
        if isinstance(vobj, dict):
            bi = vobj.get("best_index")
            if isinstance(bi, int) and 0 <= bi < len(parsed):
                best_idx = bi

    out = dict(parsed[best_idx])
    out["votes"] = votes
    return out

def filter_chunks(chunks, filters, excluders=[],filter_case=None):
    filtered_notes = []
    for chunk in chunks:
        matches = []
        for i, fil in enumerate(filters):
            flags = re.IGNORECASE
            if filter_case:
                if filter_case[i]:
                    flags = 0

            match = re.search(
                fil, 
                chunk, 
                flags=flags
            )
            if match:
                for exc in excluders:
                    exc_match = re.search(
                        exc, 
                        chunk, 
                        flags=re.IGNORECASE
                    )
                    if exc_match:
                        match = False

                if match:                        
                    matches.append(fil)
        
        if matches:
            filtered_notes.append(chunk)

    return filtered_notes

question_dictionary = {
    'ALCOHOL WITHDRAWAL SYNDROME': {
        'corpus': '''Consider any withdrawal symptoms resulting from the cessation or reduction of alcohol use following a period of heavy, prolonged use.''',
        'question': '''Is there documentation of potential alcohol withdrawal, alcohol abuse, or CIWA in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale. \n\n EXAMPLES: \n "Alcohol abuse- CIWA" = yes \n "It is possible patient is withrawing" = yes \n "Recovering from intoxication" = yes \n "Alcohol withdrawal" = yest''',
        'filters': ['ciwa', 'withdrawal', 'etoh', 'alcohol'] 
    },
    'DELIRIUM': {
        'corpus': '''Delirium is characterized by acute onset of behaviors characterized by restlessness, illusions, and incoherence of thought and speech. Delirium can often be traced to one or more contributing factors, such as a severe or chronic medical illness, changes in your metabolic balance (such as low sodium), medication, infection, surgery, or alcohol or drug withdrawal. Do not consider if delirium is due to alcohol withdrawal.''',
        'question': '''Is there explicit documentation of Delirium in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale. \n\n EXAMPLES:\n  "BID given sedation/delirium in the ICU" = Yes''',
        'filters': ['delir.*']
    },
    'DVT/THROMBOPHLEBITIS': {
        'corpus': '''Diagnosis of Deep Vein Thrombosis (DVT) must be documented in the patient's medical record, which may be confirmed by venogram, ultrasound, or CT. The patient must also have been treated with anticoagulation therapy and/or placement of a vena cava filter or clipping of the vena cava. The use of DVT prophylaxis doesn't mean DVT is diagnosed.''',
        'question': '''Is there explicit documentation of a confirmed diagnosis of Deep Vein Thrombosis (DVT) in the patient notes? Do not consider prophylaxis (ppx) or "concern for" (c/f) or "could be" (c/b). Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': ['dvt', 'deep.*vein'],
        'excluders': [' ppx', 'prophyla'] 
    },
    'STROKE/CVA': {
        'corpus': '''Stroke is characterized by a focal or global neurological deficit of rapid onset and NOT present on admission caused by a clot obstructing the flow of blood flow to the brain (ischemic stroke). Or by a blood vessel rupturing and preventing blood flow to the brain (hemorrhagic stroke). Or a transient ischemic attack which is temporary caused by a temporary clot. 
        1. The patient must have at least one of the following symptoms: • Change in level of consciousness • Hemiplegia • Hemiparesis • Numbness or sensory loss affecting on side of the body • Dysphasia or aphasia • Hemianopia • Amaurosis fugax • Other neurological signs or symptoms consistent with stroke AND Duration of neurological deficit ≥ 24 h (OR)
        2. Duration of deficit < 24 h, if neuroimaging (MR, CT, or cerebral angiography) documents a new hemorrhage or infarct consistent with stroke, or therapeutic intervention(s) were performed for stroke, or the neurological deficit results in death AND • No other readily identifiable non-stroke cause, e.g., progression of existing traumatic brain injury, seizure, tumor, metabolic or pharmacologic etiologies, is identified AND • Diagnosis is confirmed by neurology or neurosurgical specialist or neuroimaging procedure (MR, CT, angiography) or lumbar puncture (CSF demonstrating intracranial hemorrhage that was not present on admission)
    Although the neurologic deficit must not present on admission, risk factors predisposing to stroke (e.g., blunt cerebrovascular injury, dysrhythmia) may be present on admission. Remember stroke sign off doesn't mean stroke is present.''',
        'question': '''Is there explicit documentation of Stroke captured in the patient notes?. Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'subprompts': [
            'Does the patient have explicit documentation of presence of atleast one of the follow symptoms and duration of neurological deficit > 24h? Symptoms: Change in level of consciousness • Hemiplegia • Hemiparesis • Numbness or sensory loss affecting on side of the body • Dysphasia or aphasia • Hemianopia • Amaurosis fugax • Other neurological signs or symptoms consistent with stroke. Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.',
            'Answer with Yes (ONLY if all the following 3 conditions are all true) or No for option and with appropriate evidence supporting the option from the medical note for rationale. 1. Does the patient have documentation of neuroimaging (MR, CT, or cerebral angiography) that present new new hemorrhage or infarct consistent with stroke, or therapeutic intervention(s) were performed for stroke, or the neurological deficit results in death? AND 2. Are there no other readily identifiable non-stroke cause, e.g., progression of existing traumatic brain injury, seizure, tumor, metabolic or pharmacologic etiologies, is identified? AND 3. Is the Diagnosis confirmed by neurology or neurosurgical specialist or neuroimaging procedure (MR, CT, angiography) or lumbar puncture (CSF demonstrating intracranial hemorrhage that was not present on admission)?',       
            '''Is there documentation of Stroke in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.'''
        ],
        'filters': ['stroke']
    },
    'UNPLANNED INTUBATION': {
        'corpus': '''Unplanned Intubation is characterized by patients who require placement of an endotracheal tube and mechanical or assisted ventilation manifested by severe respiratory distress, hypoxia, hypercarbia, or respiratory acidosis. It is also characterized by patients who were intubated in the field or emergency department, or those intubated for surgery, an unplanned intubation occurs if they require reintubation > 24 hours after they were extubated. Make sure to use explicit evidence when abstracting.''',
        'question': '''Is there explicit documentation of Unplanned Invasive Intubation captured in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.\n\n EXAMPLES: bipap = No \n reintubation = Yes''',
        'filters': ['intub', 'vent'] 
    },
    'UNPLANNED ADMISSION TO ICU': {
        'corpus': '''Unplanned Admission to ICU is characterized by patients admitted to the ICU after initial transfer to the floor, and/or patients with an unplanned return to the ICU after initial ICU discharge.''',
        'question': '''Is there documentation of any Unplanned Admission to ICU, escalation of care to the ICU, or upgraded level of care to ICU in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale. \n\n POSITIVE EXAMPLES: Answer "Yes" if any of the following are present \n "Patient worsening, upgraded to ICU" \n "Rapid called, transferred to ICU"  \n "Admit to ICU because of aspiration event" \n "Transferred to ICU for concern of" \n "upgraded to ICU level of care" \n''',
        'filters': ['(?-i:ICU).*(?i:wors)', '(?i:wors).*(?-i:ICU)', '(?-i:ICU).*(?i:escalat)', '(?i:escalat).*(?-i:ICU)', '(?i:deterio).*(?-i:ICU)', '(?-i:ICU).*(?i:deterio)', '(?i:upgrad).*(?-i:ICU)', '(?-i:ICU).*(?i:upgrad)', '(?-i:ICU).*(?i:transfer)', '(?i:transfer).*(?-i:ICU)'] ,
        'filter_case': [True, True, True, True, True, True, True, True, True, True] 
    },
    'SEVERE SEPSIS': {
        'corpus': '''Severe sepsis is characterized by sepsis plus organ dysfunction, hypotension (low blood pressure), or hypoperfusion (insufficient blood flow) to 1 or more organs. It is also characterized by sepsis with persisting arterial hypotension or hypoperfusion despite adequate fluid resuscitation. Account severe sepsis when patients are treated or suspected for sepsis.''',
        'question': '''Is there documentation of Severe Sepsis in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': ['sepsis']
    },
    'PRESSURE ULCER': {
        'corpus': '''Pressure ulcer is characterized by a localized injury to the skin and/or underlying tissue usually over a bony prominence, as a result of pressure, or pressure in combination with shear. A number of contributing or confounding factors are also associated with pressure ulcers; the significance of these factors is yet to be elucidated. Equivalent to NPUAP Stages II-IV, Unstageable/Unclassified, and Suspected Deep Tissue Injury. Do not infer random information as potentital indicators for Pressure Ulcer. Use only concrete evidence.''',
        'question': '''Is there documentation of Pressure Ulcer in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': ['ulcer', 'sore', 'pressure.*injury']
    }, 
    'CARDIAC ARREST WITH CPR': {
        'corpus': '''Cardiac arrest is the sudden cessation of cardiac activity after hospital arrival. The patient becomes unresponsive with no normal breathing and no signs of circulation. If corrective measures are not taken rapidly, this condition progresses to sudden death. Only consider patients who, after arrival at your hospital, have had an episode of cardiac arrest evaluated by hospital personnel, and received compressions or defibrillation or cardioversion or cardiac pacing to restore circulation. Do not consider patients whose ONLY episode of cardiac arrest with CPR was on arrival to your hospital.''',
        'question': '''Is there explicit documentation of Cardiac Arrest with CPR in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': ['cpr', 'defib', 'cardioversion', 'compress']
    },
    'ACUTE KIDNEY INJURY': {
        'corpus': '''Acute Kidney Injury, AKI (stage 3), is characterized by abrupt decrease in kidney function. It does not include Patients with renal failure that were requiring chronic renal replacement therapy such as
periodic peritoneal dialysis, hemodialysis, hemofiltration, or hemodiafiltration prior to injury.''',
        'question': '''Is there explicit documentation of Acute Kidney Injury (AKI) in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [' aki ', 'kidney.*injury']
    },
    'UNPLANNED VISIT TO THE OR': {
        'corpus': '''An unplanned visit to the operating room includes an unplanned operative procedure or patients returned to the operating room after initial operative management of a related previous procedure. EXCLUDE: • Non-urgent tracheostomy and percutaneous endoscopic gastrostomy. • Pre-planned, staged and/or procedures for incidental findings. • Operative management related to a procedure that was initially performed prior to  arrival at your center. INCLUDE: "To OR Emergently" \n "Decision made to take patient to OR"''',
        'question': '''Is there explicit documentation of an unplanned visit to the operating room (OR) in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [' OR '],
        'filter_case': [True, False, False]
    },
    'PULMONARY EMBOLISM': {
        'corpus': '''A pulmonary embolism (PE) is a lodging of a blood clot in a pulmonary artery with subsequent obstruction of blood supply to the lung parenchyma. The blood clots usually originate from the deep leg veins or the pelvic venous system. Only consider PEs where the onset of symptoms began after arrival to your ED/hospital. Consider the condition present if the patient has a VQ scan interpreted as high probability of pulmonary embolism or a positive pulmonary arteriogram or positive CT angiogram and/or a diagnosis of PE is documented in the patient's medical record.''',
        'question': '''Is there explicit documentation of a PE in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [' PE ', 'pulm.*emb']
    },
    'MYOCARDIAL INFARCTION': {
        'corpus': '''A diagnosis of myocardial infarction prior to injury must be documented in the patient's medical record.''',
        'question': '''Is there history of a myocardial infarction (MI) in the six months prior to injury? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [' MI ', 'myo.*inf']
    },
    'VAP': {
        'corpus': '''Ventilator associated pneumonia (VAP) is a pneumonia where the patient is on mechanical ventilation for > 2 calendar days on the date of event, with day of ventilator placement being Day 1, amd the ventilator was in place on the date of event or the day before.''',
        'question': '''Is there explicit documentation of pneuomonia and mechanical ventilation in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [' VAP ', 'pneu', 'vent']
    },
    'ARDS': {
        'corpus': '''A diagnosis of ARDS must be documented in the patient's medical record. Onset of symptoms must have begun after arrival to your ED/hospital.''',
        'question': '''Is there explicit diagnosis of ARDS in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [' ARDS ', 'acute.*resp']
    },
    'CAUTI': {
        'corpus': '''Only consider an explicit mention of both a UTI and a catheter (such as a Foley) in the patient notes.''',
        'question': '''Is there explicit diagnosis of catheter associated urinary tract infection (CAUTI) in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [' UTI ', 'catheter', 'foley'],
        'filter_case': [True, False, False]
    },
    'OSTEOMYELITIS': {
        'corpus': '''Osteomyelitis must meet at least one of the following criteria: (1) Patient has organism(s) identified from bone by culture or non-culture based microbiologic testing method, which is performed for purposes of clinical diagnosis and treatment, for example, not Active Surveillance Culture/Testing (ASC/AST). (2) Patient has evidence of osteomyelitis on gross anatomic or histopathologic exam.''',
        'question': '''Is there explicit diagnosis of osteomyelitis in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': ['osteomyelitis']
    },
    'SUPERFICIAL INCIS SURG SITE INF': {
        'corpus': '''The superficial SSI must meet the following conditions: (1) Infection occurs within 30 days after any NHSN operative procedure (where day 1 = the procedure date). (2) the infection involves only skin and subcutaneous tissue of the incision, (3) Diagnosis of a superficial incisional SSI by the surgeon or attending physician.''',
        'question': '''Is there explicit diagnosis of a superficial surgical site infection (SSI) in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [' SSI ', 'site.*inf']
    }
}

def process_batches(notes, question_dictionary, batch_size=50):
    global CURRENT_QKEY
    results = []
    batch = []
    answered_categories = set()
    _count('n_notes_passed', len(notes))

    ''' Original Chunking (simple)
    t0 = time.perf_counter()
    notes_str ="\n\n".join(notes)
    record_timing('join_notes_str', time.perf_counter() - t0)
    _count('notes_chars_joined', len(notes_str))
    t0 = time.perf_counter()
    all_splits = text_splitter.split_text(notes_str)
    record_timing('split_chunks', time.perf_counter() - t0)
    _count('n_all_splits', len(all_splits))
    '''

    # Mixture
    t0 = time.perf_counter()
    notes_str = "\n\n".join(notes)
    record_timing("join_notes_str", time.perf_counter() - t0)
    _count("notes_chars_joined", len(notes_str))

    t0 = time.perf_counter()
    all_splits = []
    for note in notes:
        if not note:
            continue
        # Split by note boundaries; only split oversized notes with the existing splitter config.
        if len(note) > 1200:
            all_splits.extend(text_splitter.split_text(note))
        else:
            all_splits.append(note)
    record_timing("split_chunks", time.perf_counter() - t0)
    _count("n_all_splits", len(all_splits))
    #

    # Hard-fail if any complication does not have a CoT prompt.
    missing = []
    for qk in question_dictionary.keys():
        if not get_cot_prompt(qk):
            missing.append(qk)
    if missing:
        raise ValueError(f"Missing CoT prompt for question keys: {missing}")

    for question_key, question_info in question_dictionary.items():
        CURRENT_QKEY = question_key
        q0 = time.perf_counter()
        filters = question_info.get('filters')
        filter_case = question_info.get('filter_case')
        excluders = question_info.get('excluders')
        if not excluders:
            excluders = []

        t0 = time.perf_counter()
        if filters:       
            splits = filter_chunks(all_splits, filters, excluders, filter_case=filter_case)
            _count('n_filtered_splits_total', len(splits))
        else:
            splits = all_splits
            _count('n_unfiltered_splits_total', len(splits))
        record_timing('filter_chunks', time.perf_counter() - t0)

        if splits:
            logger.debug(f"Notes found for Question '{question_key}'")
            _count('n_questions_with_splits', 1)
            t0 = time.perf_counter()
            vectorstore = Chroma.from_texts(
                    texts=splits, embedding=embedding, client_settings=Settings(anonymized_telemetry=False)
                )
            record_timing('vectorstore_build', time.perf_counter() - t0)
            retriever=vectorstore.as_retriever(k=8, search_type="mmr", similarity_score_threshold=0.8)
            t0 = time.perf_counter()
            context = retriever.invoke(question_dictionary[question_key]['corpus'])
            record_timing('retriever_invoke', time.perf_counter() - t0)
            k_retrieved = len(context) if context is not None else 0
            t0 = time.perf_counter()
            vectorstore.delete_collection()
            record_timing('vectorstore_delete', time.perf_counter() - t0)
            ###
            context_text = ""
            if context is not None:
                context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in context])

            context_text = ""
            if context is not None:
                context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in context])

            cot_template = get_cot_prompt(question_key)
            if not cot_template:
                raise ValueError(f"CoT prompt missing at runtime for question_key={question_key}")

            user_body = fill_medical_notes(cot_template, context_text)
            prompt = wrap_llama_chat(user_body, LLAMA_SYSTEM_PROPOSER)
            response = ask_llm(prompt)

            v = response.get("votes", {}) if isinstance(response, dict) else {}
            append_vote_log({
                "ts": time.time(),
                "csn": CURRENT_CSN,
                "question_key": question_key,
                "prompt_kind": "cot",
                "subprompt_idx": "",
                "has_splits": True,
                "n_splits": len(splits),
                "k_retrieved": k_retrieved,
                "samples": v.get("samples", ""),
                "parsed": v.get("parsed", ""),
                "yes": v.get("yes", ""),
                "no": v.get("no", ""),
                "invalid": v.get("invalid", ""),
                "rule": v.get("rule", ""),
                "json_structured_samples": v.get("json_structured_samples", ""),
                "json_fallback_samples": v.get("json_fallback_samples", ""),
                "decision": response.get("option", ""),
                "rationale_preview": (str(response.get("rationale", ""))[:200] if isinstance(response, dict) else "")
            })

            ###
            if response['option'].lower() == 'yes':
                logger.info(f"FOUND complication '{question_key}'")
                answered_categories.add(question_key)
                results.append({
                    "note_idx": None,
                    "question_key": question_key,
                    "rationale": response['rationale'],
                    "notes_used": "\n".join(splits)
                })
        else:
            logger.debug(f"No notes found for filter string '{filters}' in question '{question_key}'")
            append_vote_log({
                "ts": time.time(),
                "csn": CURRENT_CSN,
                "question_key": question_key,
                "prompt_kind": "no_splits",
                "subprompt_idx": "",
                "has_splits": False,
                "n_splits": 0,
                "k_retrieved": 0,
                "samples": "",
                "parsed": "",
                "yes": "",
                "no": "",
                "invalid": "",
                "rule": "",
                "decision": "no",
                "rationale_preview": ""
            })

        record_timing('question_total', time.perf_counter() - q0)

    CURRENT_QKEY = None
    return results

def process_batch_and_update_results(batch, results, answered_categories, question_dictionary):
    prompts = [b[2] for b in batch]
    responses = ask_llm(prompts)
    for i, response in enumerate(responses):
        note_idx, question_key, _, prompt_type = batch[i]
        if response['option'].lower() == 'yes':
            logger.info(f"FOUND complication '{question_key}'")
            results.append({
                "note_idx": note_idx,
                "question_key": question_key,
                "rationale": response['rationale'],
                "notes_used": batch[i][2]
            })
            answered_categories.add(question_key)

    return results

def iterate_json_files(directory_path):
    for file in os.listdir(directory_path):
        if file.endswith(".json"):
            file_path = os.path.join(directory_path, file)
            file_name = os.path.splitext(file)[0]
            with open(file_path, 'r') as f:
                try:
                    file_content = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file}")
                    continue
            yield file_name, file_content

directory_path = "/home/ec2-user/fhir/mrn_json_files/"

output_dir = 'complication_results_folder/complication_results_wt_test20_llama_CoT_new5_direct'
os.makedirs(output_dir, exist_ok=True)
VOTE_LOG_PATH = os.path.join(output_dir, os.environ.get('VOTE_LOG_FILENAME', 'vote_log.csv'))
fhir_data = pd.read_json("./features_folder/patient_features_50.json") # temporary
fhir_data = [json.loads(d) for d in fhir_data['features']]

total_features = len(fhir_data)
logger.info(f"Loaded {total_features} patient features")

for idx, bundle in enumerate(fhir_data, start=1):
    start_time = time.time()
    case_t0 = time.perf_counter()
    csn = bundle['demographics'][0]['csn']
    CURRENT_CSN = csn
    if os.path.exists(os.path.join(output_dir, f'{csn}.json')):
        print(f"[{idx}/{total_features}] Data for {csn} already exists. Skipping...")
        _count('n_cases_skipped_existing', 1)
        CURRENT_CSN = None
        continue

    logger.info(f"[{idx}/{total_features}] Processing MRN {csn} for patient with {len(bundle['binary'])} notes.")

    _count('n_cases_processed', 1)
    _count('n_binary_notes_total', len(bundle.get('binary', [])))
    _count('n_med_orders_total', len(bundle.get('medication_orders', [])))

    t0 = time.perf_counter()
    notes = []
    for note in bundle['binary']:
        if 'note_type' not in note.keys():
            note_type = ''
            txt = ''
        else:
            note_type = note['note_type']
            txt = 'Note Type: ' + note_type + '\n'
        
        txt += note['note'] 
        notes.append(txt)

    for note in bundle['medication_orders']:
        txt = 'Note Type: Medication Dosage Text \n'
        txt += note['dosage_text']
        notes.append(txt)
    record_timing('build_notes_list', time.perf_counter() - t0)

    if not notes:
        logger.warning(f"[{idx}/{total_features}] No notes found for csn {csn}, skipping")
        _count('n_cases_skipped_no_notes', 1)
        CURRENT_CSN = None
        continue

    t0 = time.perf_counter()
    results = process_batches(notes, question_dictionary, batch_size=500)
    record_timing('process_batches', time.perf_counter() - t0)

    result_dict = {
        'found_questions': {result['question_key']: {
            'rationale': result['rationale'],
            'notes_used': result['notes_used'],
            'note_idx': result['note_idx']
        } for result in results if result['rationale']},
        'processing_time': time.time() - start_time
    }
    
    json_filename = os.path.join(output_dir, f'{csn}.json')
    t0 = time.perf_counter()
    with open(json_filename, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
    record_timing('write_case_json', time.perf_counter() - t0)

    record_timing('case_total', time.perf_counter() - case_t0)
    logger.info(f"[{idx}/{total_features}] Finished processing MRN: {csn}. Time taken: {result_dict['processing_time']} seconds")
    CURRENT_CSN = None

write_timing_summary(os.path.join(output_dir, 'timing_summary.json'))

