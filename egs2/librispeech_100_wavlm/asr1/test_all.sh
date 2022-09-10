#!/bin/bash

# ./run_15db_wavlm_conformer_transformer_original.sh --stage 12 --stop_stage 13 || exit 1
./run_0db_wavlm_conformer_transformer_original.sh --stage 12 --stop_stage 13 || exit 1
./run_0db_wavlm_conformer_transformer_finetune_21.sh --stage 12 --stop_stage 13 || exit 1
./run_0db_wavlm_conformer_transformer_finetune_0_1_2.sh --stage 12 --stop_stage 13 || exit 1
./run_0db_wavlm_conformer_transformer_finetune_0_1_2_sl_loss.sh --stage 12 --stop_stage 13 || exit 1
