CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py \
--config-file ./configs/DG/C3-D-M-MT.yml \
MODEL.DEVICE "cuda:0" \
CUDNN_BENCHMARK "False" \
OUTPUT_DIR ./logs/DG/C3-D-M-MT
