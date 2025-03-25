# NTIRE2025-CDFSOD-AI4EarthLab

### Installation Environment

The experimental environment is based on [mmdetection](https://github.com/open-mmlab/mmdetection/blob/main/docs/zh_cn/get_started.md), the installation environment reference mmdetection's [installation guide](https://github.com/open-mmlab/mmdetection/blob/main/docs/zh_cn/get_started.md).
```
conda create --name lae python=3.8 -y
conda activate lae
cd ./mmdetection_lae
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
cd NTIRE2025-CDFSOD-AI4EarthLab/
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir weights/bert-base-uncased
```

### Catalogue

```
.
├── configs
├── data
├── LICENSE
├── mmdetection
├── pkl2coco.py
├── pkls
├── README.md
├── submit
├── submit_codalab
└── weights
```

### Get weights
Download the checkpoint files to dir `./weights`.
> Baidu Disk: [[link]](https://pan.baidu.com/s/1r_xR4F6eLq5pXocgZc8-Ww?pwd=mpnc)

### Train GroundingDINO Model
50 groups of experiments were carried out on the 8-card A100, a total of 50*8 groups of experiments.
```
cd ./mmdetection

./tools/dist_train_muti.sh configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py "0,1,2,3,4,5,6,7" 50
```

### Infer GroundingDINO Model
Save to `*.pkl` file and convert to submit `.json` format. Infer stage only need run on single card

```
cd ./mmdetection

## 1-shot-dataset1
bash tools/dist_test_out.sh ../configs/1-shot-dataset1.py ../weights/1-shot-dataset1-db4c5ebf.pth 1 ../pkls/dataset1_1shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset1/annotations/test.json --pkl_file ../pkls/dataset1_1shot.pkl --output_json ../pkls/dataset1_1shot_coco.json --annotations_json ../submit/dataset1_1shot.json

## 1-shot-dataset2
bash tools/dist_test_out.sh ../configs/1-shot-dataset2.py ../weights/1-shot-dataset2-0bd5d280.pth 1 ../pkls/dataset2_1shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset2/annotations/test.json --pkl_file ../pkls/dataset2_1shot.pkl --output_json ../pkls/dataset2_1shot_coco.json --annotations_json ../submit/dataset2_1shot.json

## 1-shot-dataset3
bash tools/dist_test_out.sh ../configs/1-shot-dataset3.py ../weights/1-shot-dataset3-433149f8.pth 1 ../pkls/dataset3_1shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset3/annotations/test.json --pkl_file ../pkls/dataset3_1shot.pkl --output_json ../pkls/dataset3_1shot_coco.json --annotations_json ../submit/dataset3_1shot.json

## 5-shot-dataset1
bash tools/dist_test_out.sh ../configs/5-shot-dataset1.py ../weights/5-shot-dataset1-ad2ac5f0.pth 1 ../pkls/dataset1_5shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset1/annotations/test.json --pkl_file ../pkls/dataset1_5shot.pkl --output_json ../pkls/dataset1_5shot_coco.json --annotations_json ../submit/dataset1_5shot.json

## 5-shot-dataset2
bash tools/dist_test_out.sh ../configs/5-shot-dataset2.py ../weights/5-shot-dataset2-0bfccba8.pth 1 ../pkls/dataset2_5shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset2/annotations/test.json --pkl_file ../pkls/dataset2_5shot.pkl --output_json ../pkls/dataset2_5shot_coco.json --annotations_json ../submit/dataset2_5shot.json

## 5-shot-dataset3
bash tools/dist_test_out.sh ../configs/5-shot-dataset3.py ../weights/5-shot-dataset3-0011f4b1.pth 1 ../pkls/dataset3_5shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset3/annotations/test.json --pkl_file ../pkls/dataset3_5shot.pkl --output_json ../pkls/dataset3_5shot_coco.json --annotations_json ../submit/dataset3_5shot.json

## 10-shot-dataset1
bash tools/dist_test_out.sh ../configs/10-shot-dataset1.py ../weights/10-shot-dataset1-33caf03b.pth 1 ../pkls/dataset1_10shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset1/annotations/test.json --pkl_file ../pkls/dataset1_10shot.pkl --output_json ../pkls/dataset1_10shot_coco.json --annotations_json ../submit/dataset1_10shot.json

## 10-shot-dataset2
bash tools/dist_test_out.sh ../configs/10-shot-dataset2.py ../weights/10-shot-dataset2-46b5584c.pth 1 ../pkls/dataset2_10shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset2/annotations/test.json --pkl_file ../pkls/dataset2_10shot.pkl --output_json ../pkls/dataset2_10shot_coco.json --annotations_json ../submit/dataset2_10shot.json

## 10-shot-dataset3
bash tools/dist_test_out.sh ../configs/10-shot-dataset3.py ../weights/10-shot-dataset3-7325994e.pth 1 ../pkls/dataset3_10shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset3/annotations/test.json --pkl_file ../pkls/dataset3_10shot.pkl --output_json ../pkls/dataset3_10shot_coco.json --annotations_json ../submit/dataset3_10shot.json
```

