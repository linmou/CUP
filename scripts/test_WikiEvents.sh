#!/usr/bin/env bash
CkptPath=experiments/WikiEvents_F1_47.37
python RAMStest.py \
  --data_path=data/wikievents/informative \
  --dataset=wikievents \
  --ontology_file=wikievents_ontology.json \
  --lr=1e-4 \
  --train_batchsize=32 \
  --eval_batchsize=32 \
  --max_seq_len=512 \
  --model=bart \
  --model_name_or_path=./bart-base \
  --eval_only \
  --pipeline_decode \
  --ckptPath=$CkptPath