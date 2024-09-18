echo "-- Processing AV2 val set..."
python data_av2/run_preprocess.py --mode val \
  --data_dir /mnt/data/Argoverse2_Motion_Forecasting_Dataset/val/ \
  --save_dir data_av2/features/

echo "-- Processing AV2 train set..."
python data_av2/run_preprocess.py --mode train \
  --data_dir /mnt/data/Argoverse2_Motion_Forecasting_Dataset/train/ \
  --save_dir data_av2/features/