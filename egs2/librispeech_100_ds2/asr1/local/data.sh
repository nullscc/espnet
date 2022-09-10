#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

l="train100 dev test"
for split in $l; do
	mkdir -p data/$split
	dir=/ssdhome/xzw521/data/libri_noisy/0dB/clean_$split
	<$dir/text sort -u > data/$split/text
	<$dir/wav.scp sort -u > data/$split/wav.scp
	<data/$split/wav.scp awk '{printf("%s %s\n", $1, $1)}' >  data/$split/utt2spk
	utils/utt2spk_to_spk2utt.pl <data/$split/utt2spk >data/$split/spk2utt
done

log "Successfully finished. [elapsed=${SECONDS}s]"
