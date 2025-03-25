# NTIRE2025-CDFSOD-AI4EarthLab

### Installation Environment

The experimental environment is based on [mmdetection](https://github.com/open-mmlab/mmdetection/blob/main/docs/zh_cn/get_started.md), the installation environment reference mmdetection's [installation guide](https://github.com/open-mmlab/mmdetection/blob/main/docs/zh_cn/get_started.md).
```
conda create --name lae python=3.8 -y
conda activate lae
cd LAE-DINO/mmdetection_lae
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# 开发并直接运行 mmdet
pip install -v -e .
pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
```
Then download the BERT weights `bert-base-uncased` into the weights directory,
```
cd LAE-DINO
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir weights/bert-base-uncased
```
### Get weights
Baidu Disk: 

### Train GroundingDINO Model

```
cd ./mmdetection

./tools/dist_train_muti.sh configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py "0,1,2,3,4,5,6,7" 50
```



### Infer GroundingDINO Model


```
cd ./mmdetection

bash tools/dist_test_out.sh configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py ./weight/1-shot-dataset1.pth 4 ./CDFSOD_PKL/dataset1_1shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset1/annotations/test.json --pkl_file ./CDFSOD_PKL/dataset1_1shot.pkl --output_json ./CDFSOD_PKL/dataset1_1shot_coco.json --annotations_json ../commit/dataset1_1shot.json
```

