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

from questions_dict import question_dictionary
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import sys

LOG_PATH = os.environ.get("LOG_PATH", "run_test50.log")
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
LLM_MODEL_ID = "us.deepseek.r1-v1:0"
N_SAMPLES = int(os.environ.get("N_SAMPLES", "5"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))


text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=650, chunk_overlap=100, separators=["."]
        )

prompt_question = PromptTemplate(
    template=
"""SYSTEM:
You are a Trauma quality abstractor. Review the patient's medical notes and answer the compliance question.

RULES:
- If any note supports "yes", return "yes".
- DO NOT INFER. Use only explicit evidence from the notes.
- Return ONLY valid JSON (double quotes, no code fence, no extra text).

QUESTION: {question}
ABSTRACTION INSTRUCTIONS: {corpus}
MEDICAL NOTES:
{context}

OUTPUT JSON:
{{"rationale": "...", "option": "yes" or "no"}}
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

def extract_json(txt):
    s = (txt or "").strip()
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i:j+1])
        except Exception:
            return None
    return None


def ask_llm(prompt_text):
    n_tries = 0
    while True:
        try:
            t0 = time.perf_counter()
            responses = []
            for _ in range(N_SAMPLES):
                formatted_prompt = f"<｜begin▁of▁sentence｜><｜User｜>{prompt_text}<｜Assistant｜><think>\n"
                body = json.dumps({
                    "prompt": formatted_prompt,
                    "max_tokens": LLM_MAX_TOKENS,
                    "temperature": 0.3,
                    "top_p": 0.9
                })
                resp = bedrock.invoke_model(
                    modelId=LLM_MODEL_ID,
                    body=body,
                    accept="application/json",
                    contentType="application/json"
                )
                data = json.loads(resp["body"].read())
                txt = data["choices"][0]["text"]
                responses.append(txt)
            record_timing("llm_engine", time.perf_counter() - t0)
            break
        except Exception:
            n_tries += 1
            if n_tries > 5:
                return {"rationale": "No valid response", "option": "no"}

    parsed = []
    for txt in responses:
        obj = extract_json(txt)
        if obj and "option" in obj and "rationale" in obj:
            parsed.append(obj)


    if not parsed:
        sample = (responses[0] if responses else "")[:500]
        logger.warning(f"LLM output not JSON-parsable, first 500 chars: {sample!r}")
        return {"rationale": "No valid response", "option": "no"}

    y_responses = [r for r in parsed if str(r.get("option", "")).lower() == "yes"]
    n_responses = [r for r in parsed if str(r.get("option", "")).lower() == "no"]

    if len(y_responses) >= 2:
        return y_responses[0]
    if n_responses:
        return n_responses[0]
    return {"rationale": "No valid response", "option": "no"}


def process_subprompts(note, subprompts):
    for subprompt in subprompts:
        response = ask_llm(subprompt)
        if response['option'].lower() != 'yes':
            return 'no'  
    return 'yes' 

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

def process_batches(notes, question_dictionary, batch_size=50):
    global CURRENT_QKEY
    results = []
    batch = []
    answered_categories = set()
    _count('n_notes_passed', len(notes))
    t0 = time.perf_counter()
    notes_str ="\n\n".join(notes)
    record_timing('join_notes_str', time.perf_counter() - t0)
    _count('notes_chars_joined', len(notes_str))
    t0 = time.perf_counter()
    all_splits = text_splitter.split_text(notes_str)
    record_timing('split_chunks', time.perf_counter() - t0)
    _count('n_all_splits', len(all_splits))
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
            t0 = time.perf_counter()
            vectorstore.delete_collection()
            record_timing('vectorstore_delete', time.perf_counter() - t0)
            subprompts = question_info.get('subprompts')
            if subprompts:
                for subprompt in subprompts:
                    prompt = prompt_question.format(
                        corpus=question_info['corpus'],
                        context=context,
                        question=subprompt
                    )
                    response = ask_llm(prompt)
                    if response['option'].lower() == 'yes':
                        break
            else:
                prompt = prompt_question.format(
                    corpus=question_info['corpus'],
                    context=context,
                    question=question_info['question']
                )
                response = ask_llm(prompt)
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

output_dir = 'complication_results_wt_test50'
os.makedirs(output_dir, exist_ok=True)
fhir_data = pd.read_json("./patient_features_50.json") # temporary
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

