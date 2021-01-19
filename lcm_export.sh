#!/usr/bin/env bash
export BERT_BASE_DIR=./albert_tiny
export GLUE_DIR=./data
export MODEL_DIR=./output_lcm
export MODEL_PB_DIR=./model_pb_lcm


python lcm_export.py \
  --task_name=sentiment_cls \
  --do_predict=true \
  --data_dir=$GLUE_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
  --model_dir=$MODEL_DIR/ \
  --serving_model_save_path=$MODEL_PB_DIR/



