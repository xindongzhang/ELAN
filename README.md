## ELAN

### [Paper](https://www4.comp.polyu.edu.hk/~cslzhang/paper/ECCV_2022_ELAN.pdf) | [Supplementary Material](https://www4.comp.polyu.edu.hk/~cslzhang/paper/ECCV_2022_ELAN_supp.pdf)

Codes for "Efficient Long-Range Attention Network for Image Super-resolution", [arxiv link](https://arxiv.org/abs/2203.06697).


> **Efficient Long-Range Attention Network for Image Super-resolution** <br>
> [Xindong Zhang](https://github.com/xindongzhang), [Hui Zeng](https://huizeng.github.io/), [Shi Guo](https://scholar.google.com.hk/citations?user=5hsEmuQAAAAJ&hl=zh-CN), and [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/). <br>
> In ECCV 2022.

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

If ELAN helps your research or work, please consider citing the following works:

----------
```BibTex
@inproceedings{zhang2022efficient,
  title={Efficient Long-Range Attention Network for Image Super-resolution},
  author={Zhang, Xindong and Zeng, Hui and Guo, Shi and Zhang, Lei},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```
