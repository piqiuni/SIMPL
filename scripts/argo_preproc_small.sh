echo "-- Processing val set..."
python data_argo/run_preprocess.py --mode val \
  --data_dir /mnt/data/Argoverse1.1/val/data/ \
  --save_dir data_argo/features/ \
  --small
  # --debug --viz

echo "-- Processing train set..."
python data_argo/run_preprocess.py --mode train \
  --data_dir /mnt/data/Argoverse1.1/train/data/ \
  --save_dir data_argo/features/ \
  --small

# echo "-- Processing test set..."
# python data_argo/run_preprocess.py --mode test \
#   --data_dir ~/data/dataset/argo_motion_forecasting/test_obs/data/ \
#   --save_dir data_argo/features/ \
#   --small