# CapsuleNLP
Capsule NLP

## Requirements
* [Anaconda](https://www.anaconda.com/download/)
* PyTorch
```
conda install pytorch torchvision -c pytorch
```
* PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```
* torchtext
```
pip install torchtext
```
* tqdm
```
conda install tqdm
```
* torchsummary
```
pip install torchsummary
```

## Usage
```
python -m visdom.server -logging_level WARNING & python main.py --data_type TREC --num_epochs 300
optional arguments:
--data_type              dataset type [default value is 'TREC'](choices:['TREC'])
--batch_size             train batch size [default value is 100]
--num_epochs             train epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097` in your browser.

## Results
The train loss、accuracy, test loss、accuracy, and confusion matrix are showed with visdom,
the best test accuracy is ~ 99.64%.
![result](results/result.png)
