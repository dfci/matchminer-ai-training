import json
import pandas as pd

def parse_clinical_trials_json(json_file_path):
    """
    Parse clinical trials JSON file into a pandas DataFrame.
    
    Args:
        json_file_path: Path to the JSON file containing clinical trial data
        
    Returns:
        pandas DataFrame with columns: nct_id, title, brief_summary, 
        eligibility_criteria, and trial_text
    """
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # List to store parsed trial data
    trials = []
    
    # Parse each trial entry
    for trial in data:
        try:
            # Extract the protocol section
            protocol = trial.get('protocolSection', {})
            
            # Extract NCT ID
            nct_id = protocol.get('identificationModule', {}).get('nctId', '')
            
            # Extract title (using briefTitle, could also use officialTitle)
            title = protocol.get('identificationModule', {}).get('briefTitle', '')
            
            # Extract brief summary
            brief_summary = protocol.get('descriptionModule', {}).get('briefSummary', '')
            
            # Extract eligibility criteria
            eligibility_criteria = protocol.get('eligibilityModule', {}).get('eligibilityCriteria', '')
            
            # Create trial_text as concatenation
            trial_text = f"{title}\n{brief_summary}\n{eligibility_criteria}"
            
            # Append to list
            trials.append({
                'nct_id': nct_id,
                'title': title,
                'brief_summary': brief_summary,
                'eligibility_criteria': eligibility_criteria,
                'trial_text': trial_text
            })
            
        except Exception as e:
            print(f"Error parsing trial: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(trials)
    
    return df


if __name__ == "__main__":
    # Example usage
    input_file = "ctgov_interventional_phased_cancer_trials_11-3-25.json"  # Replace with your JSON file path
    
    # Parse the JSON file
    df = parse_clinical_trials_json(input_file)
    
    # Display basic info
    print(f"Parsed {len(df)} clinical trials")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Optionally save to CSV
    output_file = "../data/ctgov_trials.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved DataFrame to {output_file}")