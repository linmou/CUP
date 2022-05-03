#!/usr/bin/env bash
CkptPath=experiments/wikievents/CL
python RAMStest.py \
  --data_path=data/wikievents/informative \
  --dataset=wikievents \
  --ontology_file=aida_ontology_cleaned.csv \
  --lr=1e-4 \
  --train_batchsize=32 \
  --eval_batchsize=32 \
  --max_seq_len=256 \
  --model=bart \
  --model_name_or_path=./bart-base \
  --eval_only \
  --pipeline_decode \
  --ckptPath=$CkptPath