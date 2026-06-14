LR=2e-5

SEEDS=(22 42 66 99 111 1234) 

for SEED in "${SEEDS[@]}"
do
    work_path=exps/mlee/$SEED
    mkdir -p $work_path

    CUDA_VISIBLE_DEVICES=0 python -u engine.py \
        --dataset_type=MLEE \
        --batch_size=4  \
        --gradient_accumulation_steps=1    \
        --context_representation=decoder \
        --model_name_or_path=./roberta-large \
        --role_path=./data/MLEE/MLEE_role_name_mapping.json \
        --prompt_path=./data/prompts/prompts_MLEE_full.csv \
        --statistics_role_graph_path=./data/dset_meta/role_coref_weight_mlee.json \
        --seed=$SEED \
        --structural_type=biaffine \
        --output_dir=$work_path \
        --learning_rate=$LR \
        --max_steps=10000 \
        --max_enc_seq_length 512 \
        --max_dec_seq_length 512 \
        --window_size 250 \
        --bipartite 
done
