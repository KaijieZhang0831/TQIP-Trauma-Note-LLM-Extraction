question_dictionary = {

    'ALCOHOL WITHDRAWAL SYNDROME': {
        'corpus': '''Consider any withdrawal symptoms resulting from the cessation or reduction of alcohol use following a period of heavy, prolonged use.''',
        'question': '''Is there documentation of potential alcohol withdrawal, alcohol abuse, or CIWA in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale. \n\n EXAMPLES: \n "Alcohol abuse- CIWA" = yes \n "It is possible patient is withrawing" = yes \n "Recovering from intoxication" = yes \n "Alcohol withdrawal" = yes''',
        'filters': [
            'ciwa',  
            'withdrawal', 
            'etoh',   
            'alcohol',  
            r'delirium\s*tremens',
            r'\bDTs\b', 
            r'\bAWS\b',  
            'librium', 
            'chlordiazepoxide', 
        ],
        'filter_case': [False, False, False, False, False, True, True, False, False]
    },

    'DELIRIUM': {
        'corpus': '''Delirium is characterized by acute onset of behaviors characterized by restlessness, illusions, and incoherence of thought and speech. Delirium can often be traced to one or more contributing factors, such as a severe or chronic medical illness, changes in your metabolic balance (such as low sodium), medication, infection, surgery, or alcohol or drug withdrawal. Do not consider if delirium is due to alcohol withdrawal.''',
        'question': '''Is there explicit documentation of Delirium in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale. \n\n EXAMPLES:\n  "BID given sedation/delirium in the ICU" = Yes''',
        'filters': [
            r'delir',
            r'CAM.?ICU',
            r'encephalopath',
            r'acute\s*confus', 
            r'sundown',  
            r'\bAMS\b',   
        ],
        'filter_case': [False, False, False, False, False, True]
    },

    'DVT/THROMBOPHLEBITIS': {
        'corpus': '''Diagnosis of Deep Vein Thrombosis (DVT) must be documented in the patient's medical record, which may be confirmed by venogram, ultrasound, or CT. The patient must also have been treated with anticoagulation therapy and/or placement of a vena cava filter or clipping of the vena cava. The use of DVT prophylaxis doesn't mean DVT is diagnosed.''',
        'question': '''Is there explicit documentation of a confirmed diagnosis of Deep Vein Thrombosis (DVT) in the patient notes? Do not consider prophylaxis (ppx) or "concern for" (c/f) or "could be" (c/b). Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            r'DVT',
            r'deep\s*vein',
            r'\bVTE\b', 
            r'thrombophlebitis',
            r'venous\s*thrombo', 
            r'IVC\s*filter', 
            r'vena\s*cava\s*filter',
        ],
        'excluders': [
            r'\bppx\b', 'prophyla', 
            r'VTE\s*ppx', r'VTE\s*prophyla', 
            r'DVT\s*screen', 
        ]
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
        'filters': [
            'stroke', 
            r'\bCVA\b', 
            r'\bTIA\b', 
            r'cerebrovascular', 
            r'cerebral\s*infarct',
            r'intracranial\s*hemorrhage',
        ],
        'filter_case': [False, True, True, False, False, False]
    },

    'UNPLANNED INTUBATION': {
        'corpus': '''Unplanned Intubation is characterized by patients who require placement of an endotracheal tube and mechanical or assisted ventilation manifested by severe respiratory distress, hypoxia, hypercarbia, or respiratory acidosis. It is also characterized by patients who were intubated in the field or emergency department, or those intubated for surgery, an unplanned intubation occurs if they require reintubation > 24 hours after they were extubated. Make sure to use explicit evidence when abstracting.''',
        'question': '''Is there explicit documentation of Unplanned Invasive Intubation captured in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.\n\n EXAMPLES: bipap = No \n reintubation = Yes''',
        'filters': [
            'intub', 
            'ventilat', 
            r'\bETT\b',  
            r'emergent\s*airway', 
        ]
    },

    'UNPLANNED ADMISSION TO ICU': {
        'corpus': '''Unplanned Admission to ICU is characterized by patients admitted to the ICU after initial transfer to the floor, and/or patients with an unplanned return to the ICU after initial ICU discharge.''',
        'question': '''Is there documentation of any Unplanned Admission to ICU, escalation of care to the ICU, or upgraded level of care to ICU in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale. \n\n POSITIVE EXAMPLES: Answer "Yes" if any of the following are present \n "Patient worsening, upgraded to ICU" \n "Rapid called, transferred to ICU"  \n "Admit to ICU because of aspiration event" \n "Transferred to ICU for concern of" \n "upgraded to ICU level of care" \n''',
        'filters': [
            '(?-i:ICU).*(?i:wors)',
            '(?i:wors).*(?-i:ICU)',
            '(?-i:ICU).*(?i:escalat)',
            '(?i:escalat).*(?-i:ICU)',
            '(?i:deterio).*(?-i:ICU)',
            '(?-i:ICU).*(?i:deterio)',
            '(?i:upgrad).*(?-i:ICU)',
            '(?-i:ICU).*(?i:upgrad)',
            '(?-i:ICU).*(?i:transfer)',
            '(?i:transfer).*(?-i:ICU)',
            r'rapid\s*response.*(?-i:ICU)', 
            r'(?-i:ICU).*rapid\s*response',
            r'(?-i:RRT).*(?-i:ICU)',
            r'(?-i:ICU).*(?-i:RRT)',  
            r'decompens.*(?-i:ICU)',  
            r'(?-i:ICU).*decompens',      
            r'step.*up.*(?-i:ICU)',  
        ]
    },


    'SEVERE SEPSIS': {
        'corpus': '''Severe sepsis is characterized by sepsis plus organ dysfunction, hypotension (low blood pressure), or hypoperfusion (insufficient blood flow) to 1 or more organs. It is also characterized by sepsis with persisting arterial hypotension or hypoperfusion despite adequate fluid resuscitation. Account severe sepsis when patients are treated or suspected for sepsis.''',
        'question': '''Is there documentation of Severe Sepsis in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            'sepsis', 
            'septic',    
            r'\bSIRS\b', 
            'bacteremia',  
            'septicemia', 
        ]
    },

    'PRESSURE ULCER': {
        'corpus': '''Pressure ulcer is characterized by a localized injury to the skin and/or underlying tissue usually over a bony prominence, as a result of pressure, or pressure in combination with shear. A number of contributing or confounding factors are also associated with pressure ulcers; the significance of these factors is yet to be elucidated. Equivalent to NPUAP Stages II-IV, Unstageable/Unclassified, and Suspected Deep Tissue Injury. Do not infer random information as potentital indicators for Pressure Ulcer. Use only concrete evidence.''',
        'question': '''Is there documentation of Pressure Ulcer in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            'ulcer',
            r'bed\s*sore', 
            r'pressure\s*sore',
            r'pressure\s*injury', 
            'decub', 
            r'skin\s*breakdown', 
            r'stage\s*(II|III|IV|2|3|4)',
            r'(sacral|coccyx|coccygeal)\s*wound', 
            r'deep\s*tissue\s*injury', 
        ]
    },

    'CARDIAC ARREST WITH CPR': {
        'corpus': '''Cardiac arrest is the sudden cessation of cardiac activity after hospital arrival. The patient becomes unresponsive with no normal breathing and no signs of circulation. If corrective measures are not taken rapidly, this condition progresses to sudden death. Only consider patients who, after arrival at your hospital, have had an episode of cardiac arrest evaluated by hospital personnel, and received compressions or defibrillation or cardioversion or cardiac pacing to restore circulation. Do not consider patients whose ONLY episode of cardiac arrest with CPR was on arrival to your hospital.''',
        'question': '''Is there explicit documentation of Cardiac Arrest with CPR in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            'cpr',
            'defib', 
            'cardioversion', 
            'compress',  
            r'cardiac\s*arrest', 
            r'code\s*blue',
            r'\bROSC\b',
            'pulseless', 
            'asystole',
            r'v.?fib',    
            r'\bACLS\b',
            r'cardiac\s*pacing',  
        ]
    },

    'ACUTE KIDNEY INJURY': {
        'corpus': '''Acute Kidney Injury, AKI (stage 3), is characterized by abrupt decrease in kidney function. It does not include Patients with renal failure that were requiring chronic renal replacement therapy such as
periodic peritoneal dialysis, hemodialysis, hemofiltration, or hemodiafiltration prior to injury.''',
        'question': '''Is there explicit documentation of Acute Kidney Injury (AKI) in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            r'\bAKI\b',
            r'kidney\s*injury', 
            r'renal\s*failure', 
            r'renal\s*insufficiency',
            r'hemodialysis', 
            r'\bdialysis\b', 
            r'anuria',  
            r'oliguria',
        ],
        'excluders': [
            r'chronic\s*renal', 
            r'\bCKD\b', 
            r'\bESRD\b',
            r'prior\s*to\s*(injury|admission)', 
        ]
    },

    'UNPLANNED VISIT TO THE OR': {
        'corpus': '''An unplanned visit to the operating room includes an unplanned operative procedure or patients returned to the operating room after initial operative management of a related previous procedure. EXCLUDE: • Non-urgent tracheostomy and percutaneous endoscopic gastrostomy. • Pre-planned, staged and/or procedures for incidental findings. • Operative management related to a procedure that was initially performed prior to  arrival at your center. INCLUDE: "To OR Emergently" \n "Decision made to take patient to OR"''',
        'question': '''Is there explicit documentation of an unplanned visit to the operating room (OR) in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            r'\bto\s*(?-i:OR)\b',
            r'operating\s*room',
            r'emergent.*surg', 
            r'emergent.*operat',
            r'return.*operat', 
            r'washout', 
            r'ex.?lap', 
        ],
        'filter_case': [True, False, False, False, False, False, False],
        'excluders': [
            r'tracheostomy',
            r'\bPEG\b', 
            r'planned', 
            r'staged',
        ]
    },

    'PULMONARY EMBOLISM': {
        'corpus': '''A pulmonary embolism (PE) is a lodging of a blood clot in a pulmonary artery with subsequent obstruction of blood supply to the lung parenchyma. The blood clots usually originate from the deep leg veins or the pelvic venous system. Only consider PEs where the onset of symptoms began after arrival to your ED/hospital. Consider the condition present if the patient has a VQ scan interpreted as high probability of pulmonary embolism or a positive pulmonary arteriogram or positive CT angiogram and/or a diagnosis of PE is documented in the patient's medical record.''',
        'question': '''Is there explicit documentation of a PE in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            r'\bPE\b', 
            r'pulm.*emb', 
            r'thromboembol',
            r'saddle\s*embol',  
            r'\bVQ\s*scan\b', 
            r'pulmonary\s*arteriogram', 
            r'\bCTA\b.*chest', 
        ],
        'filter_case': [True, False, False, False, False, False, False]
    },

    'MYOCARDIAL INFARCTION': {
        'corpus': '''A diagnosis of myocardial infarction prior to injury must be documented in the patient's medical record.''',
        'question': '''Is there history of a myocardial infarction (MI) in the six months prior to injury? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            r'\bMI\b',
            r'myo.{0,20}infarc', 
            r'\bSTEMI\b',    
            r'\bNSTEMI\b',  
            r'heart\s*attack',   
            'troponin',  
        ],
        'filter_case': [True, False, True, True, False, False]
    },

    'VAP': {
        'corpus': '''Ventilator associated pneumonia (VAP) is a pneumonia where the patient is on mechanical ventilation for > 2 calendar days on the date of event, with day of ventilator placement being Day 1, amd the ventilator was in place on the date of event or the day before.''',
        'question': '''Is there explicit documentation of pneuomonia and mechanical ventilation in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            r'\bVAP\b', 
            'pneu(?!mothorax)',                
            'ventilat',  
        ]
    },

    'ARDS': {
        'corpus': '''A diagnosis of ARDS must be documented in the patient's medical record. Onset of symptoms must have begun after arrival to your ED/hospital.''',
        'question': '''Is there explicit diagnosis of ARDS in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            r'\bARDS\b', 
            r'acute.*resp.*distress',    
            r'resp.*distress.*syndrome', 
        ]
    },

    'CAUTI': {
        'corpus': '''Only consider an explicit mention of both a UTI and a catheter (such as a Foley) in the patient notes.''',
        'question': '''Is there explicit diagnosis of catheter associated urinary tract infection (CAUTI) in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            r'\bUTI\b', 
            'catheter', 
            'foley',           
            r'\bCAUTI\b',       
            r'urinary\s*tract\s*infect', 
            'bacteriuria',   
            'pyuria',           
            'urosepsis',             
            r'indwelling.*catheter',     
        ],
        'filter_case': [True, False, False, True, False, False, False, False, False]
    },

    'OSTEOMYELITIS': {
        'corpus': '''Osteomyelitis must meet at least one of the following criteria: (1) Patient has organism(s) identified from bone by culture or non-culture based microbiologic testing method, which is performed for purposes of clinical diagnosis and treatment, for example, not Active Surveillance Culture/Testing (ASC/AST). (2) Patient has evidence of osteomyelitis on gross anatomic or histopathologic exam.''',
        'question': '''Is there explicit diagnosis of osteomyelitis in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            'osteomyelitis',     
            r'bone\s*infect',   
            r'\bOM\b',                
        ],
        'filter_case': [False, False, True]
    },

    # =========================================================================
    'SUPERFICIAL INCIS SURG SITE INF': {
        'corpus': '''The superficial SSI must meet the following conditions: (1) Infection occurs within 30 days after any NHSN operative procedure (where day 1 = the procedure date). (2) the infection involves only skin and subcutaneous tissue of the incision, (3) Diagnosis of a superficial incisional SSI by the surgeon or attending physician.''',
        'question': '''Is there explicit diagnosis of a superficial surgical site infection (SSI) in the patient notes? Answer with Yes or No for option and with appropriate evidence supporting the option from the medical note for rationale.''',
        'filters': [
            r'\bSSI\b',      
            r'site\s*infect',  
            r'wound\s*infect',    
            r'surgical.*infect',       
            r'incision.*infect',      
            r'incision.*(erythema|cellulitis|drainage|puru)',  
            r'wound.*dehiscen',       
        ]
    },
}


