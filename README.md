## ELAN

Codes for "Efficient Long-Range Attention Network for Image Super-resolution", [arxiv link](https://arxiv.org/abs/2203.06697).

## Dependencies & Installation
Please refer to the following simple steps for installation. Datas can be download from Baidu cloud disk [[url]](https://pan.baidu.com/s/15WjlGRhYOtVNRYTj3lfE6A) (pwd: al4m)
```
git clone https://github.com/xindongzhang/ELAN.git
cd ELAN
conda env create -f environment.yml
conda activate elan
```

## Training
```
cd ELAN
python train.py --config ./configs/elan_light_x4.yml
```

## Citation

If SimpleIR helps your research or work, please consider citing the following works:

----------
```BibTex
@article{zhang2022efficient,
  title={Efficient Long-Range Attention Network for Image Super-resolution},
  author={Zhang, Xindong and Zeng, Hui and Guo, Shi and Zhang, Lei},
  journal={arXiv preprint arXiv:2203.06697},
  year={2022}
}
```
