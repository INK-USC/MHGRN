: ${2?"Usage: $0 DATASET ENCODER"}
dataset=$1
model=$2
shift
shift
args=$@

declare -A param_search_range=(
  ["dlr"]="1e-4 3e-4 1e-3"
  ["unfreeze_epoch"]="0 1"
)

dlr_range=${param_search_range[dlr]}
unfreeze_epoch_range=${param_search_range[unfreeze_epoch]}

echo "***** param_search_rn.sh *****"
echo "dataset: $dataset"
echo "encoder: $model"
echo "decoder: rn"
echo "decoder_learning_rate: $dlr_range"
echo "unfreeze_epoch: $unfreeze_epoch_range"
echo "******************************"

for u_epoch in $unfreeze_epoch_range; do
  for dlr in $dlr_range; do
    for seed in 0; do
      python -u rn.py --dataset $dataset --encoder $model -dlr $dlr --unfreeze_epoch $u_epoch --seed $seed $args
    done
  done
done
