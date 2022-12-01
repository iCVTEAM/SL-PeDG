CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py \
--config-file ./configs/DG/C2-C3-D-M-CS.yml \
MODEL.DEVICE "cuda:0" \
CUDNN_BENCHMARK "False" \
OUTPUT_DIR ./logs/DG/C2-C3-D-M-CS
