import os
import json
import pandas as pd
from collections import Counter, defaultdict
import pdb


COMPLICATIONS = [
    'ALCOHOL WITHDRAWAL SYNDROME',
    'DELIRIUM',
    'DVT/THROMBOPHLEBITIS',
    'STROKE/CVA',
    'UNPLANNED INTUBATION',
    'UNPLANNED ADMISSION TO ICU',
    'SEVERE SEPSIS',
    'PRESSURE ULCER',
    'CARDIAC ARREST WITH CPR',
    'ACUTE KIDNEY INJURY',
    'UNPLANNED VISIT TO THE OR',
    'PULMONARY EMBOLISM',
    'MYOCARDIAL INFARCTION',
    'VAP',
    'ARDS',
    'CAUTI',
    'OSTEOMYELITIS',
    'SUPERFICIAL INCIS SURG SITE INF'
]

def aggregate_results(output_dir):
    aggregated_data = {}

    # Loop through all JSON files in the output directory
    for filename in os.listdir(output_dir):
        if filename.endswith('.json'):
            csn = filename.replace('.json', '')
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r') as json_file:
                data = json.load(json_file)
                aggregated_data[csn] = data

    return aggregated_data

def parse_complications(complication_str):
    return complication_str.split('-') if isinstance(complication_str, str) else []

def compare_complications(df, aggregated_results):
    individual_results = []
    missed_complications = []
    FNs = defaultdict(int)
    TPs = defaultdict(int)
    FPs = defaultdict(int)
    TNs = defaultdict(int)
    totals = defaultdict(int)

    total_true_complications = 0
    total_true_positives = 0
    total_false_positives = 0
    total_predicted_complications = 0
    total_time = 0
    
    predicted_complication_counter = Counter()
    
    N = 0
    for idx, row in df.iterrows():
        csn = row['csn']
        true_complications = parse_complications(row['comp'])
        predicted_complications = aggregated_results.get(str(csn), {}).get('found_questions', {}).keys()
        #####
        if str(csn) not in aggregated_results.keys():
            continue
        else:
            N+=1
        
        time = aggregated_results.get(str(csn), {}).get('processing_time', 0)
        
        total_time += time

        # if "CAUTI" in true_complications:
        #     continue

        # if not predicted_complications:
        #     continue

        true_positives = len(set(true_complications).intersection(predicted_complications))
        false_negatives = len(set(true_complications) - set(predicted_complications))
        false_positives = len(set(predicted_complications) - set(true_complications))
        
        # Sensitivity calculation for each MRN
        sensitivity = (true_positives / len(true_complications)) * 100 if len(true_complications) > 0 else 0
        
        # Additional complications percentage
        additional_complications_percentage = (false_positives / len(predicted_complications)) * 100 if len(predicted_complications) > 0 else 0
        
        # Update totals for aggregated metrics
        total_true_complications += len(true_complications)
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_predicted_complications += len(predicted_complications)
        
        # Count the predicted complications for identifying the top 3
        predicted_complication_counter.update(predicted_complications)
        
        # Store individual results
        individual_results.append({
            'csn': csn,
            'true_complications': true_complications,
            'predicted_complications': list(predicted_complications),
            'sensitivity (%)': sensitivity,
            'additional_complications (%)': additional_complications_percentage
        })

        if false_negatives:
            missed_complications.append({
                'csn': csn,
                'true_complications': true_complications,
                'predicted_complications': list(predicted_complications)
            })

        for miss in set(true_complications) - set(predicted_complications):
            FNs[miss] += 1

        for hit in set(true_complications).intersection(predicted_complications):
            TPs[hit] += 1
            # os.system(f"rm ./complication_results_wt/{csn}.json")

        for hit in set(predicted_complications) - set(true_complications):
            FPs[hit] +=1

        for hit in set(true_complications):
            totals[hit] +=1

        for miss in set(COMPLICATIONS) - set(true_complications) - set(predicted_complications):
            TNs[miss] +=1
            
    # Calculate overall metrics
    overall_sensitivity = (total_true_positives / total_true_complications) * 100 if total_true_complications > 0 else 0
    
    average_additional_complications = (total_false_positives / N) * 100 if total_predicted_complications > 0 else 0
    
    # Identify the top 3 most predicted complications
    top_3_predicted_complications = predicted_complication_counter.most_common(3)
    
    # Aggregate results
    aggregated_metrics = {
        "overall_sensitivity (%)": overall_sensitivity,
        "average_additional_complications (%)": average_additional_complications,
        "top_3_predicted_complications": top_3_predicted_complications,
        "total_time_taken": total_time
    }
    
    # Convert individual results to DataFrame
    individual_results_df = pd.DataFrame(individual_results)
    
    return individual_results_df, aggregated_metrics, missed_complications, {'FN': FNs, 'TP':TPs, 'FP': FPs, 'TN': TNs, 'totals': totals}


# Use the function to aggregate results
output_dir = 'complication_results_wt_test/'
aggregated_results = aggregate_results(output_dir)
df = pd.read_csv('/home/kaz034/tqip/tqip/ground_truth.csv', dtype={'csn':str})
keep = set(aggregated_results.keys())
df = df[df['csn'].astype(str).isin(keep)]
print("Eval cases:", len(df))
# Use the modified function to compare complications and calculate metrics
individual_results_df, aggregated_metrics, missed_cases, per_complication_stats = compare_complications(df, aggregated_results)

individual_results_df.to_csv(os.path.join(output_dir, "eval_individual.csv"), index=False)
with open(os.path.join(output_dir, "eval_aggregated.json"), "w") as f:
    json.dump(aggregated_metrics, f, indent=2)
with open(os.path.join(output_dir, "eval_per_complication.json"), "w") as f:
    json.dump({k: {kk: int(vv) for kk, vv in d.items()} for k, d in per_complication_stats.items()}, f, indent=2)


# temp_df = pd.read_csv('result.csv')
# temp_df['mrn'] = temp_df['mrn'].astype(str)
# temp_df['results'] = temp_df['mrn'].map(aggregated_results)
# pdb.set_trace()
# Display the individual results
print("Individual Results:")
print(individual_results_df)

# Display the aggregated metrics
print("\nAggregated Metrics:")
print(f"Overall Sensitivity: {aggregated_metrics['overall_sensitivity (%)']:.2f}%")
print(f"Average % of Additional Complications: {aggregated_metrics['average_additional_complications (%)']:.2f}%")
print(f"Top 3 Most Predicted Complications: {aggregated_metrics['top_3_predicted_complications']}")
print(f"Total time in seconds: {aggregated_metrics['total_time_taken']}")

TP = per_complication_stats['TP']
TN = per_complication_stats['TN']
FP = per_complication_stats['FP']
FN = per_complication_stats['FN']

TP_total = 0
TN_total = 0
FP_total = 0
FN_total = 0

for v in COMPLICATIONS:
    print("=========================")
    print(v)
    den = TP[v] + FN[v]
    if den == 0:
        print(f"Sensitivity: {TP[v]}/{den} (NA)")
    else:
        print(f"Sensitivity: {TP[v]}/{den} ({TP[v] / den})")

    den = TN[v] + FN[v]
    if den == 0:
        print(f"NPV: {TN[v]}/{den} (NA)")
    else:
        print(f"NPV: {TN[v]}/{den} ({TN[v] / den})")

    den = TP[v] + FP[v]
    if den == 0:
        print(f"PPV: {TP[v]}/{den} (NA)")
    else:
        print(f"PPV: {TP[v]}/{den} ({TP[v] / den})")

    print("=========================")
    TP_total += TP[v]
    TN_total += TN[v]
    FP_total += FP[v]
    FN_total += FN[v]

print("TP TOTAL: " + str(TP_total)   )
print("TN TOTAL: " + str(TN_total)   )
print("FP TOTAL: " + str(FP_total)   )
print("FN TOTAL: " + str(FN_total)   )

# def parse_complications(complication_str):
#     """
#     Splits the complication string into a list of complications if multiple complications are present.
    
#     Args:
#         complication_str (str): The complication string from the DataFrame.
        
#     Returns:
#         list: A list of complications.
#     """
#     return complication_str.split('-') if isinstance(complication_str, str) else []
# def load_json_results(output_dir):
#     """
#     Load all JSON files in the output directory and return a dictionary with MRN as keys.
    
#     Args:
#         output_dir (str): Directory where the JSON files are stored.
        
#     Returns:
#         dict: A dictionary with MRN as keys and the simplified JSON content as values.
#     """
#     json_results = {}
    
#     for filename in os.listdir(output_dir):
#         if filename.endswith('.json'):
#             mrn = filename.replace('.json', '')
#             filepath = os.path.join(output_dir, filename)
#             with open(filepath, 'r') as json_file:
#                 data = json.load(json_file)
                
#                 # Extract found complications and their respective rationales
#                 found_complications = list(data.get('found_questions', {}).keys())
#                 rationales = {comp: data['found_questions'][comp]['rationale'] for comp in found_complications}
#                 processing_time = data.get('processing_time', None)
                
#                 # Store in the json_results dictionary
#                 json_results[mrn] = {
#                     'found_complications': found_complications,
#                     'rationales': rationales,
#                     'processing_time': processing_time
#                 }
    
#     return json_results

# def merge_json_to_dataframe(df, json_results):
#     """
#     Merge the JSON results into the DataFrame as new columns: 'found_complications', 'rationales', and 'processing_time'.
    
#     Args:
#         df (pd.DataFrame): The original DataFrame.
#         json_results (dict): Dictionary with MRN as keys and simplified JSON data as values.
        
#     Returns:
#         pd.DataFrame: The DataFrame with new columns 'found_complications', 'rationales', and 'processing_time'.
#     """
#     # Parse the complication column into a list
#     df['complication'] = df['complication'].apply(parse_complications)
    
#     # Map the JSON results to the DataFrame using the MRN as the key
#     df['found_complications'] = df['mrn'].map(lambda mrn: json_results.get(mrn, {}).get('found_complications', []))
#     df['rationales'] = df['mrn'].map(lambda mrn: json_results.get(mrn, {}).get('rationales', {}))
#     # df['processing_time'] = df['mrn'].map(lambda mrn: json_results.get(mrn, {}).get('processing_time', None))
    
#     return df

# # Directory where the JSON files are stored
# output_dir = 'refined_results'
# df = pd.read_csv('result.csv')
# df['mrn'] = df['mrn'].astype(str)
# # Load the JSON results
# json_results = load_json_results(output_dir)

# # Merge the JSON results into the DataFrame
# df = merge_json_to_dataframe(df, json_results)
# df = df.drop('pat_id', axis=1)
# # Display the DataFrame with the new columns
# df = df[['mrn', 'csn', 'complication', 'found_complications', 'rationales', 'notes']]#, 'processing_time']]
# df = df[~df['complication'].apply(lambda x: 'CAUTI' in x)]
# print(len(df))
#print(missed_stats)
# pdb.set_trace()
# Optionally, save the updated DataFrame to a CSV file
# df_with_results.to_csv('df_with_results.csv', index=False)
