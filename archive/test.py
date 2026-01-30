import os
import json
import pandas as pd
from collections import Counter
import pdb
def aggregate_results(output_dir):
    aggregated_data = {}

    # Loop through all JSON files in the output directory
    for filename in os.listdir(output_dir):
        if filename.endswith('.json'):
            mrn = filename.replace('.json', '')
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r') as json_file:
                data = json.load(json_file)
                aggregated_data[mrn] = data

    return aggregated_data

def parse_complications(complication_str):
    return complication_str.split('-') if isinstance(complication_str, str) else []

def compare_complications(df, aggregated_results):
    individual_results = []
    
    total_true_complications = 0
    total_true_positives = 0
    total_false_positives = 0
    total_predicted_complications = 0
    total_time = 0
    
    predicted_complication_counter = Counter()
    
    for idx, row in df.iterrows():
        
        mrn = row['mrn']
        true_complications = parse_complications(row['comp'])
        predicted_complications = aggregated_results.get(str(mrn), {}).get('found_questions', {}).keys()
        time = aggregated_results.get(str(mrn), {}).get('processing_time', 0)
        
        total_time += time

        if "CAUTI" in true_complications:
            continue

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
            'mrn': mrn,
            'true_complications': true_complications,
            'predicted_complications': list(predicted_complications),
            'sensitivity (%)': sensitivity,
            'additional_complications (%)': additional_complications_percentage
        })
    
    # Calculate overall metrics
    overall_sensitivity = (total_true_positives / total_true_complications) * 100 if total_true_complications > 0 else 0
    print(total_false_positives)
    average_additional_complications = (total_false_positives / (len(df)-1)) * 100 if total_predicted_complications > 0 else 0
    
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
    
    return individual_results_df, aggregated_metrics


# Use the function to aggregate results
output_dir = 'complication_results_wt/'
aggregated_results = aggregate_results(output_dir)
df = pd.read_csv('/home/ec2-user/fhir/ground_truth.csv')
# Use the modified function to compare complications and calculate metrics
individual_results_df, aggregated_metrics = compare_complications(df, aggregated_results)


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
# pdb.set_trace()
# Optionally, save the updated DataFrame to a CSV file
# df_with_results.to_csv('df_with_results.csv', index=False)
