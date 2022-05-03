#!/usr/bin/env bash
CkptPath=experiments/RAMS_F1_46.50
python train.py \
  --data_path=data/RAMS \
  --dataset=RAMS \
  --ontology_file=aida_ontology_cleaned.csv \
  --lr=1e-4 \
  --train_batchsize=32 \
  --eval_batchsize=32 \
  --max_seq_len=512 \
  --model=bart \
  --model_name_or_path=./bart-base \
  --eval_only \
  --pipeline_decode \
  --ckptPath=$CkptPath
