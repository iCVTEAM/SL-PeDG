CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py \
--config-file ./configs/DG/C3-D-M-MT.yml \
--eval-only \
MODEL.DEVICE "cuda:0" \
MODEL.WEIGHTS "./model_weights/C3-D-M-MT.pth" \
CUDNN_BENCHMARK "False" \
OUTPUT_DIR ./logs/DG/C3-D-M-MT