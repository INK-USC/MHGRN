: ${2?"Usage: $0 DATASET MODEL"}
dataset=$1
model=$2
shift
shift
args=$@

lr_range="1e-5 2e-5 3e-5 6e-5 1e-4"
bs=32
n_epochs=100

echo "***** param_search_lm.sh *****"
echo "dataset: $dataset"
echo "model_name: $model"
echo "batch_size: $bs"
echo "learning_rate: $lr_range"
echo "******************************"

for lr in $lr_range; do
  for seed in 0; do
    python -u lm.py --dataset $dataset --encoder $model -elr $lr -bs $bs --n_epochs $n_epochs --seed $seed $args
  done
done
