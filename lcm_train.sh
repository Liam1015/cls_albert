#!/usr/bin/env bash

export BERT_BASE_DIR=./albert_tiny
export GLUE_DIR=./data
export OUTPUT_DIR=./output_lcm


python run_classifier_LCM.py \
  --task_name=sentiment_cls \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
  --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt \
  --max_seq_length=32 \
  --train_batch_size=128 \
  --learning_rate=1e-3 \
  --num_train_epochs=90 \
  --output_dir=$OUTPUT_DIR/ \
  --lcm_stop_epochs=20
  --alpha=4.0
