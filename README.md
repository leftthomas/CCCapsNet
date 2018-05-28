# CCCapsNet
A PyTorch implementation of Compositional coding Capsule Network based on AAAI 2018 paper [Compositional coding capsule with random sample consensus]().

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
* PyTorch-NLP
```
pip install pytorch-nlp
```
* tqdm
```
conda install tqdm
```

## Datasets
The `AGNews`, `AmazonReview`, `DBPedia`, `SogouNews`, `YahooAnswers` and `YelpReview` datasets are coming from [here](http://goo.gl/JyCnZq).

The `Newsgroups`, `Reuters`, `Cade` and `WebKB` datasets can be found [here](http://ana.cachopo.org/datasets-for-single-label-text-categorization).

The `TREC`, `IMDB` and `SMT` datasets are downloaded by `PyTorch-NLP`.

You needn't download the datasets by yourself, the code will download them automatically. 

## Usage
```
python -m visdom.server -logging_level WARNING & python main.py --data_type TREC --num_epochs 300
optional arguments:
--data_type              dataset type [default value is 'TREC'](choices:['TREC', 'SMT', 'IMDB', 'Newsgroups', 'Reuters', 
                         'Cade', 'WebKB', 'DBPedia', 'AGNews', 'YahooAnswers', 'SogouNews', 'YelpReview', 'AmazonReview'])
--fine_grained           use fine grained class or not, it only works for TREC, SMT, Reuters, YelpReview and AmazonReview [default value is False]
--num_iterations         initial routing iterations number [default value is 1]
--batch_size             train batch size [default value is 30]
--num_epochs             train epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097/env/$data_type` in your browser, 
`$data_type` means the dataset type which you are training.

## Results
The train loss、accuracy, test loss、accuracy, and confusion matrix are showed with visdom,
the best test accuracy is ~ 99.64%.
![result](results/result.png)
