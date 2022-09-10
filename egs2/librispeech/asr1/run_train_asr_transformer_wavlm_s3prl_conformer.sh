#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
# test_sets="test_clean test_other"
test_sets="test_clean_original test_15db test_0db"

asr_tag=s3prl_wavlm_conformer_transformer_100_4
asr_config=conf/tuning/train_asr_conformer7_wavlm_large.yaml
inference_config=conf/decode_asr.yaml

# --inference_asr_model 44epoch.pth \
./asr.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --lang en \
    --ngpu 2 \
    --nj 32 \
    --inference_nj 32 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --audio_format "wav" \
    --feats_type raw \
    --use_lm false \
	--feats_normalize utt_mvn \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
	--asr_stats_dir exp/asr_stats_raw_en_bpe5000_utt_mvn_100 \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
