#!/usr/bin/env bash
STORAGE=experiments/WikiEvents/CL
python train.py \
  --data_path=data/wikievents/informative \
  --dataset=wikievents \
  --ontology_file=wikievents_ontology.json \
  --lr=1e-4 \
  --train_batchsize=32 \
  --eval_batchsize=32 \
  --max_seq_len=512 \
  --model=bart \
  --model_name_or_path=./bart-base \
  --CL\
  --pipeline_decode \
  --storage_dir=$STORAGE
