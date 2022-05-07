#!/usr/bin/env bash
STORAGE=experiments/RAMS/CL
python train.py --augment \
  --data_path=data/RAMS \
  --dataset=RAMS \
  --ontology_file=aida_ontology_cleaned.csv \
  --lr=1e-4 \
  --train_batchsize=32 \
  --eval_batchsize=32 \
  --max_seq_len=512 \
  --model=bart \
  --model_name_or_path=./bart-base \
  --CL\
  --pipeline_decode \
  --storage_dir=$STORAGE
