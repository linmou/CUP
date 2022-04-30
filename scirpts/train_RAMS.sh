#!/usr/bin/env bash
STORAGE=experiments/RAMS/CL
python train.py --augment \
  --data_path=data/RAMS/RAMSwithcoref \
  --dataset=RAMS \
  --ontology_file=aida_ontology_cleaned.csv \
  --lr=1e-4 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --max_seq_len=256 \
  --model=bart \
  --model_name_or_path=./bart-base \
  --CL\
  --pipeline_decode \
  --storage_dir=$STORAGE
