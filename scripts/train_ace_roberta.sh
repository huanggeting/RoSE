LR=2e-5

SEEDS=(22 42 66 99 111 1234)

for SEED in "${SEEDS[@]}"
do
    work_path=exps/ace/$SEED/$LR
    mkdir -p $work_path

    CUDA_VISIBLE_DEVICES=0 python -u engine.py \
        --batch_size=4  \
        --model_type=RoSE \
        --dataset_type=ace_eeqa \
        --model_name_or_path=./roberta-large \
        --role_path=./data/dset_meta/description_ace.csv \
        --prompt_path=./data/prompts/prompts_ace_concat.csv \
        --statistics_role_graph_path=./data/dset_meta/role_coref_weight_ace.json \
        --structural_type=biaffine \
        --seed=$SEED \
        --output_dir=$work_path \
        --learning_rate=$LR \
        --max_steps=10000 \
        --max_enc_seq_length 512 \
        --max_prompt_seq_length 512 \
        --bipartite
done