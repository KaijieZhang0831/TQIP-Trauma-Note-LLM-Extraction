import os
import json
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
import ast
import logging
import time

os.environ["TRANSFORMERS_OFFLINE"] = "1"

question_dictionary = {
    'ALCOHOL WITHDRAWAL SYNDROME': {
        'corpus': '''Alcohol Withdrawal Syndrome is characterized by tremor, sweating, anxiety, agitation, depression, nausea, and malaise. It occurs 6-48 hours after cessation of alcohol consumption and, when uncomplicated, abates after 2-5 days. It may be complicated by grand mal seizures and may progress to delirium (known as delirium tremens).''',
        'question': '''Is there explicit documentation of Alcohol Withdrawal Syndrome in the patient note? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.'''
    },
    'DELIRIUM': {
        'corpus': '''Delirium is characterized by acute onset of behaviors characterized by restlessness, illusions, and incoherence of thought and speech. Delirium can often be traced to one or more contributing factors, such as a severe or chronic medical illness, changes in your metabolic balance (such as low sodium), medication, infection, surgery, or alcohol or drug withdrawal. Do not consider if delirium is due to alcohol withdrawal.''',
        'question': '''Is there explicit documentation of Delirium in the patient note? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.'''
    },
    'DVT': {
        'corpus': '''Diagnosis of Deep Vein Thrombosis (DVT) must be documented in the patient's medical record, which may be confirmed by venogram, ultrasound, or CT. The patient must also have been treated with anticoagulation therapy and/or placement of a vena cava filter or clipping of the vena cava. The use of DVT prophylaxis doesn't mean DVT is diagnosed.''',
        'question': '''Is there explicit documentation of Deep Vein Thrombosis (DVT) in the patient note? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.'''
    },
    'STROKE': {
        'corpus': '''Stroke is characterized by a focal or global neurological deficit of rapid onset and NOT present on admission caused by a clot obstructing the flow of blood flow to the brain (ischemic stroke). Or by a blood vessel rupturing and preventing blood flow to the brain (hemorrhagic stroke). Or a transient ischemic attack which is temporary caused by a temporary clot. 
        1. The patient must have at least one of the following symptoms: • Change in level of consciousness • Hemiplegia • Hemiparesis • Numbness or sensory loss affecting on side of the body • Dysphasia or aphasia • Hemianopia • Amaurosis fugax • Other neurological signs or symptoms consistent with stroke AND Duration of neurological deficit ≥ 24 h (OR)
        2. Duration of deficit < 24 h, if neuroimaging (MR, CT, or cerebral angiography) documents a new hemorrhage or infarct consistent with stroke, or therapeutic intervention(s) were performed for stroke, or the neurological deficit results in death AND • No other readily identifiable non-stroke cause, e.g., progression of existing traumatic brain injury, seizure, tumor, metabolic or pharmacologic etiologies, is identified AND • Diagnosis is confirmed by neurology or neurosurgical specialist or neuroimaging procedure (MR, CT, angiography) or lumbar puncture (CSF demonstrating intracranial hemorrhage that was not present on admission)
    Although the neurologic deficit must not present on admission, risk factors predisposing to stroke (e.g., blunt cerebrovascular injury, dysrhythmia) may be present on admission. Remember stroke sign off doesn't mean stroke is present.''',
        'question': '''Is there explicit documentation of Stroke captured in the patient note?. Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'subprompts': [
            'Does the patient have explicit documentation of presence of atleast one of the follow symptoms and duration of neurological deficit > 24h? Symptoms: Change in level of consciousness • Hemiplegia • Hemiparesis • Numbness or sensory loss affecting on side of the body • Dysphasia or aphasia • Hemianopia • Amaurosis fugax • Other neurological signs or symptoms consistent with stroke. Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.',
            'Answer with Yes (ONLY if all the following 3 conditions are all true) or No for option and with appropriate evidence supporting the option from the medical note for rationale. 1. Does the patient have documentation of neuroimaging (MR, CT, or cerebral angiography) that present new new hemorrhage or infarct consistent with stroke, or therapeutic intervention(s) were performed for stroke, or the neurological deficit results in death? AND 2. Are there no other readily identifiable non-stroke cause, e.g., progression of existing traumatic brain injury, seizure, tumor, metabolic or pharmacologic etiologies, is identified? AND 3. Is the Diagnosis confirmed by neurology or neurosurgical specialist or neuroimaging procedure (MR, CT, angiography) or lumbar puncture (CSF demonstrating intracranial hemorrhage that was not present on admission)?',       
            '''Is there documentation of Stroke in the patient note? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.'''
        ],
        'filter_string': 'stroke'
    },
    'UINTUB': {
        'corpus': '''Unplanned Intubation is characterized by patients who require placement of an endotracheal tube and mechanical or assisted ventilation manifested by severe respiratory distress, hypoxia, hypercarbia, or respiratory acidosis. It is also characterized by patients who were intubated in the field or emergency department, or those intubated for surgery, an unplanned intubation occurs if they require reintubation > 24 hours after they were extubated. Make sure to use explicit evidence when abstracting.''',
        'question': '''Is there explicit documentation of Unplanned Intubation captured in the patient note? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.'''
    },
    'UAICU': {
        'corpus': '''Unplanned Admission to ICU is characterized by patients admitted to the ICU after initial transfer to the floor, and/or patients with an unplanned return to the ICU after initial ICU discharge. Do not consider patients with a planned pre-operative ICU stay.''',
        'question': '''Is there documentation of Unplanned Admission to ICU in the patient note? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
    },
    'SEVERE SEPSIS': {
        'corpus': '''Severe sepsis is characterized by sepsis plus organ dysfunction, hypotension (low blood pressure), or hypoperfusion (insufficient blood flow) to 1 or more organs. It is also characterized by sepsis with persisting arterial hypotension or hypoperfusion despite adequate fluid resuscitation. Account severe sepsis when patients are treated or suspected for sepsis.''',
        'question': '''Is there documentation of Severe Sepsis in the patient note? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filter_string': 'sepsis'
    },
    'PRESSURE ULCER': {
        'corpus': '''Pressure ulcer is characterized by a localized injury to the skin and/or underlying tissue usually over a bony prominence, as a result of pressure, or pressure in combination with shear. A number of contributing or confounding factors are also associated with pressure ulcers; the significance of these factors is yet to be elucidated. Equivalent to NPUAP Stages II-IV, Unstageable/Unclassified, and Suspected Deep Tissue Injury. Do not infer random information as potentital indicators for Pressure Ulcer. Use only concrete evidence.''',
        'question': '''Is there documentation of Pressure Ulcer in the patient note? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.'''
    }, 
    'CARDIAC ARREST WITH CPR': {
        'corpus': '''Cardiac arrest is the sudden cessation of cardiac activity after hospital arrival. The patient becomes unresponsive with no normal breathing and no signs of circulation. If corrective measures are not taken rapidly, this condition progresses to sudden death. Only consider patients who, after arrival at your hospital, have had an episode of cardiac arrest evaluated by hospital personnel, and received compressions or defibrillation or cardioversion or cardiac pacing to restore circulation. Do not consider patients whose ONLY episode of cardiac arrest with CPR was on arrival to your hospital.''',
        'question': '''Is there explicit documentation of Cardiac Arrest with CPR in the patient note? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.'''
    }
}

json_directory = "complication_results/"
output_directory = 'refined_results/'
class QUALLM_JSON(BaseModel):
    reason: str
    option: str
llm = models.vllm(
        "/home/ec2-user/llama3.1/Meta-Llama-3.1-8B-Instruct",
        trust_remote_code = True,
        max_model_len=8192,
        gpu_memory_utilization=0.95
    )
llm_engine = generate.json(llm, QUALLM_JSON)
sampling_params = vllm.SamplingParams(
                        temperature=0,
                        max_tokens=512, 
                        use_beam_search=True, 
                        n=5
                        )

def get_llm_response(rationale, disease_info):
    # Fill in the code to call the LLM and analyze the rationale
    # Example LLM call placeholder
    prompt = generate_prompt(rationale, disease_info)
    response = llm_engine(prompt, sampling_params)
    if response['option'] == 'Y':
        return True
    else:
        return False

def generate_prompt(rationale, disease_info):
    # Prompt to ask the LLM
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a Trauma quality abstractor. Given the following rationale for diagnosing a disease, evaluate whether the rationale is aligned with the provided medical documentation.
    If the rationale is based on just superficial reasoning or vague language like 'I infer' or 'I think.', then return option as "N". If it loosely makes sense then return option as "Y". Explain why it aligns or doesn't align with the documentation.<|eot_id|><|start_header_id|>user<|end_header_id|>

    Rationale: {rationale}
    
    Medical Documentation: {disease_info}

    OUTPUT FORMAT: Return the answer as a JSON object following the format,
    {{"reason": str, "option": str}}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    """
    return prompt

def process_json_file(file_path, output_directory):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        filtered_questions = {}

        # Loop over the found_questions to extract rationale and corresponding disease
        for question_key, question_content in data.get('found_questions', {}).items():
            rationale = question_content.get('rationale', "")
            disease = question_key  # Assuming the question key represents the disease name

            # Get the relevant disease documentation
            disease_info = question_dictionary.get(disease).get('corpus')
            
            # Call LLM with rationale and disease documentation
            if get_llm_response(rationale, disease_info):
                # If LLM validates the rationale, add it to the filtered questions
                filtered_questions[question_key] = question_content

        # If there are any filtered questions, save the output JSON
        if filtered_questions:
            output_data = {"found_questions": filtered_questions}
            output_filename = os.path.basename(file_path)
            output_filepath = os.path.join(output_directory, output_filename)
            with open(output_filepath, 'w') as output_file:
                json.dump(output_data, output_file, indent=4)

def process_all_json_files(directory_path, output_directory):
    # Iterate over all JSON files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            process_json_file(file_path, output_directory)

# Call the function to process all JSON files in the directory
process_all_json_files(json_directory, output_directory)