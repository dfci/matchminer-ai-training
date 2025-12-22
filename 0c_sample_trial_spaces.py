import pandas as pd

trial_space_lineitems = pd.read_csv('trial_space_lineitems.csv')

print(trial_space_lineitems.info())

def clean_trial_spaces(line_item_space_dataframe):
    search_list = ["Age", "Cancer type allowed", "Histology allowed", "Cancer burden allowed", "Prior treatment required", "Prior treatment excluded", "Biomarkers required", "Biomarkers excluded"]
    return line_item_space_dataframe.this_space.apply(lambda x: all(term in x for term in search_list))


trial_space_lineitems['good_space'] = clean_trial_spaces(trial_space_lineitems)
print(trial_space_lineitems.good_space.value_counts())

trial_space_lineitems = trial_space_lineitems[trial_space_lineitems.good_space]
print(trial_space_lineitems.info())

dfci_enrollments = pd.read_csv('dfci_enrolled_nctids.csv')

trial_space_lineitems = trial_space_lineitems[~trial_space_lineitems.nct_id.isin(dfci_enrollments.nct_id)]

sample_trials = trial_space_lineitems.groupby('nct_id').first().reset_index()[['nct_id']].sample(n=3000, random_state=42)

output = pd.merge(sample_trials, trial_space_lineitems, on='nct_id').reset_index(drop=True)
output['space_index'] = output.index

print(output.info())

output.to_csv('sample_trial_space_lineitems.csv')
