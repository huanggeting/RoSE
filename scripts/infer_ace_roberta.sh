if [ $# == 0 ] 
then
    SEED=22
    LR=2e-5
    BatchSize=4
else
    SEED=$1
    LR=$2
fi

CUDA_VISIBLE_DEVICES=0 python -u engine.py \
    --dataset_type=ace_eeqa \
    --batch_size=$BatchSize \
    --context_representation=decoder \
    --model_name_or_path=./roberta-large \
    --inference_model_path=./exps/ace/22/checkpoint \
    --role_path=./data/dset_meta/description_ace.csv \
    --prompt_path=./data/prompts/prompts_ace_concat.csv \
    --statistics_role_graph_path=./data/dset_meta/role_coref_weight_ace.json \
    --seed=$SEED \
    --learning_rate=$LR \
    --max_steps=10000 \
    --max_enc_seq_length 500 \
    --max_dec_seq_length 360 \
    --window_size 250 \
    --bipartite \
    --inference_only