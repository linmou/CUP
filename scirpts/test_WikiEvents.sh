#!/usr/bin/env bash
$CkptPath=experiments/WikiEvents/CL
python train.py \
  --data_path=data/wikievents \
  --dataset=wikievents \
  --ontology_file=aida_ontology_cleaned.csv \
  --lr=1e-4 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --max_seq_len=256 \
  --model=bart \
  --model_name_or_path=./bart-base \
  --eval_only \
  --pipeline_decode \
  --ckptPath=$CkptPath