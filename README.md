# An non-CUDA demo for MORAN v2

## Requirements

The requirements are the same with the master version. 
(Except CPU version [PyTorch 0.3.*](https://pytorch.org/) without CUDA)

- [PyTorch CPU version](https://pytorch.org/) 0.3.*
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

## Demo

Download the model parameter file `demo.pth`.

- [BaiduYun](https://pan.baidu.com/s/1TqZfvoEhyv57yf4YBjSzFg) (password: l8em)
- [Google Drive](https://drive.google.com/file/d/1IDvT51MXKSseDq3X57uPjOzeSYI09zip/view?usp=sharing)
- [OneDrive](https://1drv.ms/u/s!Am3wqyDHs7r0hkAl0AtRIODcqOV3)

Put it into root folder. Then, execute the `demo.py` for more visualizations.

```bash
	python demo.py
``` 

![](demo/demo.png)

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

## Attention
The project is only free for academic research purposes.
