set -e

# 1. generate SQL template
python src/run.py \
       --run_name t5-3b \
       --model_name_or_path experimental_outputs/train/template_generator/BEST_MODEL \
       --dataset $1 \
       --output_dir experimental_outputs/inference/$1 \
       --cache_dir transformers_cache \
       --do_train false \
       --do_eval true \
       --fp16 false \
       --per_device_eval_batch_size 1 \
       --seed 1 \
       --logging_strategy steps \
       --logging_first_step true \
       --logging_steps 4 \
       --predict_with_generate true \
       --num_beams 8 \
       --num_beam_groups 1 \
       --overwrite_cache true \
       --overwrite_output_dir true \

# 2. align (SELECT, STRUCTURE) with the user question
python src/run_aligner.py \
       --model_name_or_path experimental_outputs/train/aligner/checkpoint_best.pkl \
       --tokenizer_path microsoft/deberta-v3-large \
       --do_train false \
       --do_test true \
       --learning_rate 5e-6 \
       --train_batch_size 4 \
       --eval_batch_size 4 \
       --epochs 20 \
       --test_file experimental_outputs/inference/$1/predictions_eval_None.json \
       --output_file experimental_outputs/inference/$1/align_select-structure.json \

# 3. prepare data for LLM inference
python src/utils/get_data.py \
       --test_set_name $1 \

# 4. text2sql using LLM
python LLM_test2sql.py \
       --key $2 \
       --test_set_name $1 \
       --value_match_method sbert \
