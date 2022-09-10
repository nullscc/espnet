#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train100"
valid_set="dev"
test_sets="test_0db"

asr_tag=0db_ds2
asr_stats_dir=asr_stats_raw_en_bpe5000_0db
dumpdir=dump0dB
asr_config=conf/tuning/train_asr_ds2.yaml
inference_config=conf/decode_asr_ds2.yaml

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
    --asr_tag "${asr_tag}" \
	--asr_stats_dir "${asr_stats_dir}" \
	--dumpdir "${dumpdir}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
	--feats_normalize utt_mvn \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
