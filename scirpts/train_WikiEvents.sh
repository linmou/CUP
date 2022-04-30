#!/usr/bin/env bash
STORAGE=experiments/WikiEvents/CL
python train.py \
  --data_path=data/wikievents/WikiwCoref \
  --dataset=wikievents \
  --ontology_file=wikievents_ontology.csv \
  --lr=1e-4 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --max_seq_len=256 \
  --model=bart \
  --model_name_or_path=./bart-base \
  --CL\
  --pipeline_decode \
  --storage_dir=$STORAGE
