# MORAN: A Multi-Object Rectified Attention Network for Scene Text Recognition

![](https://img.shields.io/badge/version-v2-orange.svg)

MORAN is a network with rectification mechanism for general scene text recognition. The paper in [arXiv]() version is available now. 

![](demo/MORAN_v2.gif)

## Improvements of MORAN v2:

- More stable rectification network for one-stage training
- Replace VGG backbone by ResNet
- Use bidirectional decoder (a trick borrowed from [ASTER](https://github.com/bgshih/aster))

| <center>Dataset</center> | <center>IIIT5K</center> | <center>SVT</center> | <center>IC03</center> | <center>IC13</center> | <center>SVT-P</center> | <center>CUTE80</center> | <center>IC15 (1811)</center> | <center>IC15 (2077)</center> |
| :---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:|
| MORAN v1 (curriculum training) | <center>91.2</center> | <center>**88.3**</center> | <center>**95.0**</center> | <center>92.4</center> | <center>76.1</center> | <center>77.4</center> | <center>74.7</center> | <center>68.8</center> |
| <center>MORAN v2 (one-stage training)</center> | <center>**93.4**</center> | <center>**88.3**</center> | <center>94.2</center> | <center>**93.2**</center> | <center>**79.7**</center> | <center>**81.9**</center> | <center>**77.8**</center> | <center>**73.9**</center> |

## Requirements

- [PyTorch](https://pytorch.org/) 0.3.*
- [TorchVision](https://pypi.org/project/torchvision/)
- [Python](https://www.python.org/) 2.7.*
- [OpenCV](https://opencv.org/) 2.4.*

Use [pip](https://pypi.org/project/pip/) to install the following libraries.

```bash
    pip install -r requirements.txt
```

- [Colour](https://pypi.org/project/colour/)
- [LMDB](https://pypi.org/project/lmdb/)
- [matplotlib](https://pypi.org/project/matplotlib/)

## Data Preparation
Please convert your own dataset to lmdb format by using the [tool](https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py) provided by [@Baoguang Shi](https://github.com/bgshih). You can also download the training and testing datasets prepared by us. The raw pictures of testing datasets can be found [here](https://github.com/chengzhanzhan/STR).

- [about 20G training datasets and testing datasets](https://pan.baidu.com/s/1TqZfvoEhyv57yf4YBjSzFg), password: l8em

## Training and Testing

Modify the path to dataset folder in `train_MORAN.sh`:

```bash
	--train_nips path_to_dataset \
	--train_cvpr path_to_dataset \
	--valroot path_to_dataset \
```

And start training: (manually decrease the learning rate for your task)

```bash
	sh train_MORAN.sh
```

## Demo

Download the model parameter file from the link above and put the `demo.pth` into root folder. Then, execute the `demo.py` for more visualizations.

```bash
	python demo.py
``` 

## Citation

```
@article{cluo2019moran,
  author  = {Canjie Luo, Lianwen Jin, Zenghui Sun},
  title   = {MORAN: A Multi-Object Rectified Attention Network for Scene Text Recognition},
  journal = {Pattern Recognition}, 
  volume  = {}, 
  number  = {}, 
  pages   = {},
  year    = {2019}, 
}
```

## Acknowledgment
The repo is developed based on [@Jieru Mei's](https://github.com/meijieru) [crnn.pytorch](https://github.com/meijieru/crnn.pytorch) and [@marvis'](https://github.com/marvis) [ocr_attention](https://github.com/marvis/ocr_attention). Thanks for your contribution.
