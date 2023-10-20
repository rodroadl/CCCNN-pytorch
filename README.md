# CCCNN
- Name: GunGyeom James Kim
- OS: Windows 11 / Colab(V100 GPU)

## Descripton
PyTorch implementation of Color Constancy Convolutional Network in [Color Constancy Using CNNs](https://arxiv.org/abs/1504.04548)

> S. Bianco, C. Cusano and R. Schettini, "Color constancy using CNNs," 2015 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Boston, MA, USA, 2015, pp. 81-89, doi: 10.1109/CVPRW.2015.7301275.

### TODO
- [ ] At testing time, generate a single illuminant estimation per image by pooling the predicted patch illuminants
- [ ] Optimize ```RandomPatches```

## How to
You can just run train and test on Colab using *colab.ipynb* or on your terminal by following below requiement and commands

### Requirements
- matplotlib                    3.5.3
- numpy                         1.21.5
- opencv-python                 4.7.0.72
- pandas                        1.3.5
- torch                         1.13.1
- torchvision                   0.14.1
- tqdm                          4.65.0

### Train
The [SimpleCube++](https://github.com/Visillect/CubePlusPlus) dataset

```bash
python src/train.py --train-file "data/91-image_x3.h5" \
                --eval-file "data/Set5_x3.h5" \
                --outputs-dir "pth" \
                --scale 3 \
                --lr 1e-4 \
                --batch-size 16 \
                --num-epochs 400 \
                --seed 123                
```

```bash
python train.py --train-images-dir ./SimpleCube++/train/PNG \
                --train-labels-file .SimpleCube++/train/gt.csv \
                --eval-images-dir ./SimpleCube++/test/PNG \
                --eval-labels-file ./SimpleCube++/test/gt.csv \
                --outputs-dir ./pth \
                --batch-size 32 \
                --num-epochs 10 \
                --lr 1e-3 \
                --num-patches 5
```

### Test
The [SimpleCube++](https://github.com/Visillect/CubePlusPlus) dataset

```bash
python src/test.py --weights-file "pth/srcnn_x3.pth" \
               --image-file "data/butterfly.png" \
```
