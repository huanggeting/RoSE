LR=2e-5

SEEDS=(22 42 66 99 111 1234)

for SEED in "${SEEDS[@]}"
do
    work_path=exps/rams/$SEED/$LR
    mkdir -p $work_path

    CUDA_VISIBLE_DEVICES=0 python -u engine.py \
        --model_type=RoSE \
        --batch_size=4  \
        --dataset_type=rams \
        --model_name_or_path=./roberta-large \
        --role_path=./data/dset_meta/description_rams.csv \
        --prompt_path=./data/prompts/prompts_rams_concat.csv \
        --statistics_role_graph_path=./data/dset_meta/role_coref_weight_rams.json \
        --seed=$SEED \
        --output_dir=$work_path \
        --structural_type=biaffine \
        --learning_rate=$LR \
        --max_steps=10000 \
        --max_enc_seq_length 512 \
        --max_prompt_seq_length 330 \
        --bipartite 
done
