#!/bin/bash
# Fill the account/QOS placeholders in every #SBATCH header, in place.
# Confirm the values first with:  sacctmgr show assoc user=$USER
#     bash slurm/configure.sh <A100_ACCOUNT> <A100_QOS> <H200_ACCOUNT> <H200_QOS>
# The previous project used different values from the cluster guide's examples,
# so do not copy either source blindly.
set -euo pipefail
if [ $# -ne 4 ]; then
    echo "usage: bash slurm/configure.sh A100_ACCOUNT A100_QOS H200_ACCOUNT H200_QOS" >&2; exit 1
fi
cd "$(dirname "$0")"
for f in [0-9][0-9]_*.sh; do
    sed -i -e "s|<A100_ACCOUNT>|$1|g" -e "s|<A100_QOS>|$2|g" \
           -e "s|<H200_ACCOUNT>|$3|g" -e "s|<H200_QOS>|$4|g" "$f"
done
left=$(grep -l -E "<(A100|H200)_(ACCOUNT|QOS)>" [0-9][0-9]_*.sh || true)
if [ -n "$left" ]; then echo "placeholders remain in: $left" >&2; exit 1; fi
echo "Filled account/QOS in $(ls [0-9][0-9]_*.sh | wc -l) scripts."
