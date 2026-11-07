#!/bin/bash
# PokeFlex download (Phase 13; pre-registered Oct 22, 2026). Run on a DEV SERVER with internet (crimv3mgpu015), not on
# the login node or a compute node:   nohup bash slurm/00d_download_pokeflex.sh <stage> > logs/pokeflex_<stage>.log 2>&1 &
# The personal link is read from ../PokeFlex/url.txt (never stored here). Files go to $BIG_ROOT/datasets/pokeflex/.
# Stages (sizes read from the server on Oct 29, header-only requests):
#   format   one drop take + one poke take of Sponge (0.53 + 4.70 GB)           ~ 5.2 GB   (fix the format first)
#   drops    all 54 drop takes (18 objects x 3)                                   25.9 GB
#   pokes    all 116 poke takes                                                  562.5 GB
# Each file is downloaded to <name>.part, its size checked against the server's Content-Length, then renamed: an
# interrupted run never leaves a truncated .zip behind (the official script would skip it). Re-running skips finished
# files. Nothing is unzipped here.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/cluster.env
STAGE="${1:?usage: 00d_download_pokeflex.sh format|drops|pokes}"
LINK=$(head -1 ../PokeFlex/url.txt | tr -d ' \r\n')
[ -n "$LINK" ] || { echo "no link in ../PokeFlex/url.txt"; exit 1; }
OUT="$BIG_ROOT/datasets/pokeflex"
mkdir -p "$OUT"
objects=(3dPrintedBunny 3dPrintedCylinder 3dPrintedHeart 3dPrintedPizza 3dPrintedPyramid Beanbag FoamCylinder FoamDice
         FoamHalfSphere MemoryFoam Pillow PlushDice PlushMoon PlushOctopus PlushTurtle PlushVolleyball Sponge ToiletPaperRoll)
takes_poking=(7 6 6 6 6 7 7 8 6 6 7 8 6 7 6 6 5 6)

files=()
case "$STAGE" in
  format) files=(dropping/Sponge/Sponge_T1.zip poking/Sponge/Sponge_T1.zip) ;;
  drops)  for o in "${objects[@]}"; do for i in 1 2 3; do files+=("dropping/$o/${o}_T$i.zip"); done; done ;;
  pokes)  for k in "${!objects[@]}"; do o=${objects[$k]}; for i in $(seq 1 ${takes_poking[$k]}); do files+=("poking/$o/${o}_T$i.zip"); done; done ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac

echo "stage $STAGE: ${#files[@]} files -> $OUT   ($(df -h "$BIG_ROOT" | tail -1 | awk '{print $4}') free)"
for f in "${files[@]}"; do
  dst="$OUT/$f"
  want=$(curl -sI -L --max-time 60 "${LINK}&file=/$f" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v+0}')
  if [ "$want" -le 0 ]; then echo "FAIL  $f: no size from the server (link expired?)"; exit 1; fi
  if [ -f "$dst" ] && [ "$(stat -c %s "$dst")" -eq "$want" ]; then echo "skip  $f"; continue; fi
  mkdir -p "$(dirname "$dst")"
  curl -s -L --retry 5 --retry-delay 20 --max-time 7200 -o "$dst.part" "${LINK}&file=/$f"
  got=$(stat -c %s "$dst.part")
  if [ "$got" -ne "$want" ]; then echo "FAIL  $f: got $got of $want bytes"; rm -f "$dst.part"; exit 1; fi
  mv "$dst.part" "$dst"
  echo "ok    $f ($((want / 1000000)) MB)"
done
echo "stage $STAGE done: $(du -sh "$OUT" | cut -f1) in $OUT"
