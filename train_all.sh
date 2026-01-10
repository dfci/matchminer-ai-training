# note: vllm steps can fail for parallelized scripts if there is no pre-compiled cache yet for that vllm config. if this happens, restarting script at that stage should allow training to proceed.

# pull a JSON file from ctgov 
# clinicaltrials.gov API can be challenging to work with, so we just did it manually by going to this link:
# https://clinicaltrials.gov/search?cond=cancer%20OR%20lymphoma%20OR%20carcinoma%20OR%20leukemia%20OR%20sarcoma%20OR%20melanoma%20OR%20myeloma%20OR%20myelodysplastic%20OR%20myeloproliferative&aggFilters=phase:0%201%202%203%204,status:not%20rec,studyType:int
# and then selecting "Download" and a JSON download option.

# to work with the same file we started with, do:
wget -P ../data https://huggingface.co/datasets/ksg-dfci/mmai-synthetic/resolve/main/ctgov_interventional_phased_cancer_trials_11-3-25.json


# try to pre-compile relevant vllm configurations
aggregator=$(cat << EOF
from vllm import LLM
llm = LLM(
        model='openai/gpt-oss-120b',
        tensor_parallel_size=1,
        download_dir="../models",
        gpu_memory_utilization=0.94,
        max_model_len=10000
    )

EOF
)
python -c "$aggregator"

aggregator=$(cat << EOF
from vllm import LLM
llm = LLM(
        model='openai/gpt-oss-120b',
        tensor_parallel_size=1,
        download_dir="../models",
        gpu_memory_utilization=0.95,
        max_model_len=15000
    )

EOF
)
python -c "$aggregator"


aggregator=$(cat << EOF
from vllm import LLM
llm = LLM(
        model='openai/gpt-oss-120b',
        tensor_parallel_size=2,
        download_dir="../models",
        gpu_memory_utilization=0.95,
        max_model_len=30000
    )

EOF
)
python -c "$aggregator"


aggregator=$(cat << EOF
from vllm import LLM
llm = LLM(
        model='openai/gpt-oss-120b',
        tensor_parallel_size=2,
        download_dir="../models",
        gpu_memory_utilization=0.93,
        max_model_len=120000
    )

EOF
)
python -c "$aggregator"

aggregator=$(cat << EOF
from vllm import LLM
llm = LLM(
        model='openai/gpt-oss-120b',
        tensor_parallel_size=1,
        download_dir="../models",
        gpu_memory_utilization=0.95,
        max_model_len=20000
    )

EOF
)
python -c "$aggregator"

aggregator=$(cat << EOF
from vllm import LLM
llm = LLM(
        model='openai/gpt-oss-120b',
        tensor_parallel_size=1,
        download_dir="../models",
        gpu_memory_utilization=0.95,
        max_model_len=10000
    )

EOF
)
python -c "$aggregator"



python 0a_parse_ctgov_json.py 

python 0b_create_trial_spaces.py \
   --input ../data/no_phi/ctgov_trials.csv \
   --gpus 0,1,2,3,4,5,6,7 \
   --gpus-per-instance 1

echo 0 done

python 0c_sample_trial_spaces.py

python 1a_make_synthetic_enrollee_prompts.py

echo 1a done

python 1b_make_synthetic_negative_enrollee_prompts.py

echo 1b done


python 2_make_synthetic_notes_sharded.py \
  --input_csv ../data/no_phi/trial_spaces_with_positive_prompts.csv \
  --out_dir ../data/no_phi/synthetic_notes \
  --model openai/gpt-oss-120b \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --download_dir ../models \
  --max_model_len 10000 --max_new_tokens 5000 \
  --batch_size 1000 --tp 1 --temperature 0.75 --top_p 0.5 \
  --perturb_prob 0.3

echo 2a done



python 2_make_synthetic_notes_sharded.py \
  --input_csv ../data/no_phi/trial_spaces_with_negative_prompts.csv \
  --out_dir ../data/no_phi/synthetic_negative_notes \
  --model openai/gpt-oss-120b \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --download_dir ../models \
  --max_model_len 10000 --max_new_tokens 5000 \
  --batch_size 1000 --tp 1 --temperature 0.75 --top_p 0.5 \
  --perturb_prob 0.3

echo 2b done



aggregator=$(cat << EOF
import pandas as pd

positive_notes = pd.read_parquet("../data/no_phi/synthetic_notes/synthetic_notes.parquet")

print(positive_notes.info())

negative_notes = pd.read_parquet("../data/no_phi/synthetic_negative_notes/synthetic_notes.parquet")

print(negative_notes.info())

spaces = pd.read_csv('trial_spaces_with_positive_prompts.csv')

negative_notes['pseudo_mrn'] = negative_notes.pseudo_mrn * 1000000

output = pd.concat([positive_notes, negative_notes], ignore_index=True)

output = pd.merge(output, spaces, on='space_index')

output.to_parquet("../data/no_phi/all_synthetic_notes.parquet")

EOF
)

python -c "$aggregator"






python 6_summarize_patients.py \
  --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
  --output_parquet ../data/no_phi/patient_serial_summaries.parquet \
  --shard_dir ../data/no_phi/summary_shards \
  --model openai/gpt-oss-120b \
  --download_dir ../models \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --max_model_len 20000 \
  --prompt_batch_size 1000 \
  --generate_dates \
  --synthetic_start_date 2017-01-01 \
  --synthetic_min_days 0 \
  --synthetic_max_days 180 

echo 6 done


aggregator=$(cat << EOF
import pandas as pd

spaces = pd.read_csv('../data/no_phi/sample_trial_space_lineitems.csv')
summaries = pd.read_parquet('../data/no_phi/patient_summaries.parquet')

summaries = pd.merge(summaries, spaces, on='space_index')

summaries.to_parquet('../data/no_phi/patient_summaries_with_spaces.parquet')

EOF
)

python -c "$aggregator"





python llm_check_trials.py \
 --input_parquet ../data/no_phi/patient_summaries_with_spaces.parquet \
 --out_dir ../data/no_phi/initial_trialcheck_outputs \
 --final_output space_specific_eligibility_checks.parquet \
 --gpus 0,1,2,3,4,5,6,7 \
 --gpus_per_kernel 1 \
 --prompt_batch_size 2000 \
 --model openai/gpt-oss-120b \
 --download_dir ../models \
 --max_model_len 20000 \
 --gpu_memory_utilization 0.95

echo 7 done

mv ../data/no_phi/initial_trialcheck_outputs/space_specific_eligibility_checks.parquet ../data/no_phi/space_specific_eligibility_checks.parquet

accelerate launch finetune_embedder.py -i ../data/no_phi/space_specific_eligibility_checks.parquet \
-c ../models/initial_embedder_training -m Qwen/Qwen3-Embedding-0.6B -o ../models/pt_trial_summary_perspace_finetuned.model

echo 8 done

python make_top_matches.py \
  --parquet ../data/no_phi/space_specific_eligibility_checks.parquet \
  --model ../models/pt_trial_summary_perspace_finetuned.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 \
  --top_k_patients 40 \
  --encode_batch_size 128 \
  --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet ../data/no_phi/top_cohorts_tocheck_round1.parquet \
  --out_patients_parquet ../data/no_phi/top_patients_tocheck_round1.parquet

echo 9a done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_cohorts_tocheck_round1.parquet \
  --out_dir ../data/no_phi/round1_patientcentric_checks \
  --final_output top_cohorts_checked_round1.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir ../models \
  --max_model_len 10000 \
  --gpu_memory_utilization 0.95

echo 9b done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_patients_tocheck_round1.parquet \
  --out_dir ../data/no_phi/round1_trialcentric_checks \
  --final_output top_patients_checked_round1.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir ../models \
  --max_model_len 10000 \
  --gpu_memory_utilization 0.95

echo 9c done

accelerate launch finetune_embedder.py \
   -i ../data/no_phi/round1_trialcentric_checks/top_patients_checked_round1.parquet \
   -i ../data/no_phi/round1_patientcentric_checks/top_cohorts_checked_round1.parquet \
   -c ../models/reranker1_training \
   -m ../models/pt_trial_summary_perspace_finetuned.model \
   -o ../models/reranker_round1.model

echo 10 done

python make_top_matches.py \
  --parquet ../data/no_phi/space_specific_eligibility_checks.parquet \
  --model ../models/reranker_round1.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 \
  --top_k_patients 40 \
  --encode_batch_size 128 \
  --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet ../data/no_phi/top_cohorts_tocheck_round2.parquet \
  --out_patients_parquet ../data/no_phi/top_patients_tocheck_round2.parquet

echo 11a done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_cohorts_tocheck_round2.parquet \
  --out_dir ../data/no_phi/round2_patientcentric_checks \
  --final_output top_cohorts_checked_round2.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir ../models \
  --max_model_len 10000 \
  --gpu_memory_utilization 0.95

echo 11b done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_patients_tocheck_round2.parquet \
  --out_dir ../data/no_phi/round2_trialcentric_checks \
  --final_output top_patients_checked_round2.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir ../models \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.95

echo 11c done

accelerate launch finetune_embedder.py \
   -i ../data/no_phi/round2_trialcentric_checks/top_patients_checked_round2.parquet \
   -i ../data/no_phi/round2_patientcentric_checks/top_cohorts_checked_round2.parquet \
   -c ../models/reranker2_training \
   -m ../models/reranker_round1.model \
   -o ../models/reranker_round2.model

echo 12 done


python make_top_matches.py \
  --parquet ../data/no_phi/space_specific_eligibility_checks.parquet \
  --model ../models/reranker_round2.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 \
  --top_k_patients 40 \
  --encode_batch_size 128 \
  --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet ../data/no_phi/top_cohorts_tocheck_round3.parquet \
  --out_patients_parquet ../data/no_phi/top_patients_tocheck_round3.parquet

echo 13a done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_cohorts_tocheck_round3.parquet \
  --out_dir ../data/no_phi/round3_patientcentric_checks \
  --final_output top_cohorts_checked_round3.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir ../models \
  --max_model_len 10000 \
  --gpu_memory_utilization 0.95

echo 13b done

python llm_check_trials.py \
  --input_parquet ../data/no_phi/top_patients_tocheck_round3.parquet \
  --out_dir ../data/no_phi/round3_trialcentric_checks \
  --final_output top_patients_checked_round3.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir ../models \
  --max_model_len 10000 \
  --gpu_memory_utilization 0.95

echo 13c done

python 14_check_boilerplate.py \
  --model openai/gpt-oss-120b \
  --download_dir ../models \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 1 \
  --prompt_batch_size 1000 \
  --max_model_len 10000 \
  --max_new_tokens 5000 \
  --gpu_memory_utilization 0.95 \
  --out_dir ../data/no_phi/boilerplate_checks

echo 14 done

accelerate launch --num_processes 8 15_train_modernbert_trial_checker.py

echo 15 done

accelerate launch --num_processes 8 16_train_modernbert_boilerplate_checker.py

echo 16 done





