# note: vllm steps can fail for parallelized scripts if there is no pre-compiled cache yet for that vllm config. if this happens, restarting script at that stage should allow training to proceed.

# pull a JSON file from ctgov 
# clinicaltrials.gov API can be challenging to work with, so we just did it manually by going to this link:
# https://clinicaltrials.gov/search?cond=cancer%20OR%20lymphoma%20OR%20carcinoma%20OR%20leukemia%20OR%20sarcoma%20OR%20melanoma%20OR%20myeloma%20OR%20myelodysplastic%20OR%20myeloproliferative&aggFilters=phase:0%201%202%203%204,status:not%20rec,studyType:int
# and then selecting "Download" and a JSON download option.

# to work with the same file we started with, do:
wget https://huggingface.co/datasets/ksg-dfci/mmai-synthetic/resolve/main/ctgov_interventional_phased_cancer_trials_11-3-25.json

# try to pre-compile relevant vllm configurations
aggregator=$(cat << EOF
from vllm import LLM
llm = LLM(
        model='openai/gpt-oss-120',
        tensor_parallel_size=1,
        download_dir="./meta_ai",
        gpu_memory_utilization=0.94,
        max_model_len=10000
    )

EOF
)
python -c "$aggregator"

aggregator=$(cat << EOF
from vllm import LLM
llm = LLM(
        model='openai/gpt-oss-120',
        tensor_parallel_size=1,
        download_dir="./meta_ai",
        gpu_memory_utilization=0.95,
        max_model_len=15000
    )

EOF
)
python -c "$aggregator"


aggregator=$(cat << EOF
from vllm import LLM
llm = LLM(
        model='openai/gpt-oss-120',
        tensor_parallel_size=2,
        download_dir="./meta_ai",
        gpu_memory_utilization=0.95,
        max_model_len=30000
    )

EOF
)
python -c "$aggregator"


aggregator=$(cat << EOF
from vllm import LLM
llm = LLM(
        model='openai/gpt-oss-120',
        tensor_parallel_size=2,
        download_dir="./meta_ai",
        gpu_memory_utilization=0.93,
        max_model_len=120000
    )

EOF
)
python -c "$aggregator"

aggregator=$(cat << EOF
from vllm import LLM
llm = LLM(
        model='openai/gpt-oss-120',
        tensor_parallel_size=1,
        download_dir="./meta_ai",
        gpu_memory_utilization=0.95,
        max_model_len=20000
    )

EOF
)
python -c "$aggregator"

aggregator=$(cat << EOF
from vllm import LLM
llm = LLM(
        model='openai/gpt-oss-120',
        tensor_parallel_size=1,
        download_dir="./meta_ai",
        gpu_memory_utilization=0.95,
        max_model_len=10000
    )

EOF
)
python -c "$aggregator"



python 0a_parse_ctgov_json.py 

python 0b_create_trial_spaces.py \
   --input ctgov_trials.csv \
   --gpus 0,1,2,3,4,5,6,7 \
   --gpus-per-instance 1

echo 0 done

python 0c_sample_trial_spaces.py

python 1a_make_synthetic_enrollee_prompts.py

echo 1a done

python 1b_make_synthetic_negative_enrollee_prompts.py

echo 1b done


python 2_make_synthetic_notes_sharded.py \
  --input_csv trial_spaces_with_positive_prompts.csv \
  --out_dir ./synthetic_notes \
  --model openai/gpt-oss-120b \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --download_dir ../meta_ai \
  --max_model_len 10000 --max_new_tokens 5000 \
  --batch_size 1000 --tp 1 --temperature 0.75 --top_p 0.5 \
  --perturb_prob 0.3

echo 2a done



python 2_make_synthetic_notes_sharded.py \
  --input_csv trial_spaces_with_negative_prompts.csv \
  --out_dir ./synthetic_negative_notes \
  --model openai/gpt-oss-120b \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --download_dir ../meta_ai \
  --max_model_len 10000 --max_new_tokens 5000 \
  --batch_size 1000 --tp 1 --temperature 0.75 --top_p 0.5 \
  --perturb_prob 0.3

echo 2b done


python 3a_tag_synthetic_patient_notes.py \
  --input_parquet ./synthetic_notes/synthetic_notes.parquet \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --chunk_size 10000 \
  --prompt_batch_size 5000 \
  --max_model_len 20000 \
  --note_sample_n 120000 \
  --final_output tagged_chunks_enrolled_pt_reports_with_boilerplate_positive.parquet

echo 3a done

mv ./tagging_out ./tagging_out_positive


python 3a_tag_synthetic_patient_notes.py \
  --input_parquet ./synthetic_negative_notes/synthetic_notes.parquet \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --chunk_size 10000 \
  --prompt_batch_size 5000 \
  --max_model_len 20000 \
  --note_sample_n 120000 \
  --final_output tagged_chunks_enrolled_pt_reports_with_boilerplate_negative.parquet

echo 3a done

mv ./tagging_out ./tagging_out_negative

aggregator=$(cat << EOF
import pandas as pd

positive_notes = pd.read_parquet("./synthetic_notes/synthetic_notes.parquet")

print(positive_notes.info())

negative_notes = pd.read_parquet("./synthetic_negative_notes/synthetic_notes.parquet")

print(negative_notes.info())

spaces = pd.read_csv('trial_spaces_with_positive_prompts.csv')

negative_notes['pseudo_mrn'] = negative_notes.pseudo_mrn * 1000000

output = pd.concat([positive_notes, negative_notes], ignore_index=True)

output = pd.merge(output, spaces, on='space_index')

output.to_parquet("all_synthetic_notes.parquet")

EOF
)

python -c "$aggregator"



accelerate launch 3b_train_tiny_oncbert.py \
    --data trial_space_lineitems.csv:trial_text \
    trial_space_lineitems.csv:this_space \
    trial_space_lineitems.csv:trial_boilerplate_text \
    all_synthetic_notes.parquet:synthetic_note \
    --output_dir ./onc_bert_tiny \
    --per_device_train_batch_size 64

echo 3b done


aggregator_tags=$(cat << EOF
import pandas as pd

positive_tags = pd.read_parquet("./tagging_out_positive/tagged_chunks_enrolled_pt_reports_with_boilerplate_positive.parquet")

print(positive_tags.info())

negative_tags = pd.read_parquet("./tagging_out_negative/tagged_chunks_enrolled_pt_reports_with_boilerplate_negative.parquet")

print(negative_tags.info())

spaces = pd.read_csv('trial_spaces_with_positive_prompts.csv')[['space_index','split']]

output = pd.concat([positive_tags, negative_tags], ignore_index=True)

output = pd.merge(output, spaces, on='space_index')

output.to_parquet("tagged_chunks_enrolled_pt_reports_with_boilerplate.parquet")

EOF
)

python -c "$aggregator_tags"


accelerate launch 4_train_tiny_bert_tagger_automodel.py

echo 4 done

#### Tagger synthetic validation metrics
# AUROC 0.9313
# AUPRC 0.6643
# Best F1 score: 0.6499
# Threshold at best F1: 0.3311
# Precision at Best F1: 0.5637
# Recall at Best F1: 0.7672
# 
# Confusion Matrix at Best Threshold:                                                                                                                            
# True Negatives:  895488                                                        
# False Positives: 96322                                                                                                                                         
# False Negatives: 37770                                                                                                                                         
# True Positives:  124440     

python 5_make_patient_longnotes.py \
  --input_parquet all_synthetic_notes.parquet \
  --spaces_csv trial_spaces_lineitems.csv \
  --model_id ./auto-tiny-bert-tagger-with-pretraining \
  --max_length 128 \
  --prob_threshold 0.1 \
  --batch_size 1024 \
  --out_parquet useful_trial_enrollments_with_longnotes.parquet \
  --tmp_dir ./longnotes_shards \
  --gpus 0,1,2,3,4,5,6,7

echo 5 done

python 6_summarize_longnotes.py \
  --input_parquet useful_trial_enrollments_with_longnotes.parquet \
  --output_parquet patient_summaries_and_their_spaces.parquet \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --max_model_len 120000 \
  --prompt_batch_size 2000

echo 6 done

python llm_check_trials.py \
 --input_parquet patient_summaries_and_their_spaces.parquet \
 --out_dir ./initial_trialcheck_outputs \
 --final_output space_specific_eligibility_checks.parquet \
 --gpus 0,1,2,3,4,5,6,7 \
 --gpus_per_kernel 1 \
 --prompt_batch_size 2000 \
 --model openai/gpt-oss-120b \
 --download_dir ../meta_ai \
 --max_model_len 20000 \
 --gpu_memory_utilization 0.95

echo 7 done

accelerate launch finetune_embedder.py -i ./initial_trialcheck_outputs/space_specific_eligibility_checks.parquet \
-c ./initial_embedder_training -m Qwen/Qwen3-Embedding-0.6B -o ./pt_trial_summary_perspace_finetuned.model

echo 8 done

python make_top_matches.py \
  --parquet ./initial_trialcheck_outputs/space_specific_eligibility_checks.parquet \
  --model pt_trial_summary_perspace_finetuned.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 \
  --top_k_patients 40 \
  --encode_batch_size 128 \
  --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet top_cohorts_tocheck_round1.parquet \
  --out_patients_parquet top_patients_tocheck_round1.parquet

echo 9a done

python llm_check_trials.py \
  --input_parquet top_cohorts_tocheck_round1.parquet \
  --out_dir ./round1_patientcentric_checks \
  --final_output top_cohorts_checked_round1.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --max_model_len 10000 \
  --gpu_memory_utilization 0.95

echo 9b done

python llm_check_trials.py \
  --input_parquet top_patients_tocheck_round1.parquet \
  --out_dir ./round1_trialcentric_checks \
  --final_output top_patients_checked_round1.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --max_model_len 10000 \
  --gpu_memory_utilization 0.95

echo 9c done

accelerate launch finetune_embedder.py \
   -i ./round1_trialcentric_checks/top_patients_checked_round1.parquet \
   -i ./round1_patientcentric_checks/top_cohorts_checked_round1.parquet \
   -c ./reranker1_training \
   -m ./pt_trial_summary_perspace_finetuned.model \
   -o ./reranker_round1.model

echo 10 done

python make_top_matches.py \
  --parquet ./initial_trialcheck_outputs/space_specific_eligibility_checks.parquet \
  --model reranker_round1.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 \
  --top_k_patients 40 \
  --encode_batch_size 128 \
  --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet top_cohorts_tocheck_round2.parquet \
  --out_patients_parquet top_patients_tocheck_round2.parquet

echo 11a done

python llm_check_trials.py \
  --input_parquet top_cohorts_tocheck_round2.parquet \
  --out_dir ./round2_patientcentric_checks \
  --final_output top_cohorts_checked_round2.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --max_model_len 10000 \
  --gpu_memory_utilization 0.95

echo 11b done

python llm_check_trials.py \
  --input_parquet top_patients_tocheck_round2.parquet \
  --out_dir ./round2_trialcentric_checks \
  --final_output top_patients_checked_round2.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.95

echo 11c done

accelerate launch finetune_embedder.py \
   -i ./round2_trialcentric_checks/top_patients_checked_round2.parquet \
   -i ./round2_patientcentric_checks/top_cohorts_checked_round2.parquet \
   -c ./reranker2_training \
   -m ./reranker_round1.model \
   -o ./reranker_round2.model

echo 12 done


python make_top_matches.py \
  --parquet ./initial_trialcheck_outputs/space_specific_eligibility_checks.parquet \
  --model reranker_round2.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 \
  --top_k_patients 40 \
  --encode_batch_size 128 \
  --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet top_cohorts_tocheck_round3.parquet \
  --out_patients_parquet top_patients_tocheck_round3.parquet

echo 13a done

python llm_check_trials.py \
  --input_parquet top_cohorts_tocheck_round3.parquet \
  --out_dir ./round3_patientcentric_checks \
  --final_output top_cohorts_checked_round3.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --max_model_len 10000 \
  --gpu_memory_utilization 0.95

echo 13b done

python llm_check_trials.py \
  --input_parquet top_patients_tocheck_round3.parquet \
  --out_dir ./round3_trialcentric_checks \
  --final_output top_patients_checked_round3.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --max_model_len 10000 \
  --gpu_memory_utilization 0.95

echo 13c done

python 14_check_boilerplate.py \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 1000 \
  --max_model_len 10000 \
  --max_new_tokens 5000 \
  --gpu_memory_utilization 0.95 \
  --out_dir ./boilerplate_checks

echo 14 done

accelerate launch --num_processes 8 15_train_modernbert_trial_checker.py

echo 15 done

accelerate launch --num_processes 8 16_train_modernbert_boilerplate_checker.py

echo 16 done





