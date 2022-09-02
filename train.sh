export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 128 --moco-k 65536\
  --mlp --moco-t 0.2 --aug-plus --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /data1/ImageData/ILSVRC2012-100

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_lincls.py \
  -a resnet50 \
  --lr 10.0 \
  --batch-size 256 \
  --pretrained checkpoint_0099.pth.tar \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  /data1/ImageData/ILSVRC2012-100

