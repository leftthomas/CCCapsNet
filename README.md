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
* PorterStemmer
```
pip install PorterStemmer
```
* googledrivedownloader
```
pip install googledrivedownloader
```
* tqdm
```
conda install tqdm
```

## Datasets
The original `AGNews`, `AmazonReview`, `DBPedia`, `YahooAnswers` and `YelpReview` datasets are coming from [here](http://goo.gl/JyCnZq).

The original `Newsgroups`, `Reuters` and `WebKB` datasets can be found [here](http://ana.cachopo.org/datasets-for-single-label-text-categorization).

The original `IMDB` dataset is downloaded by `PyTorch-NLP` automatically.

We have uploaded all the original datasets into [BaiduYun](https://pan.baidu.com/s/1FrgwMzUFF8IMFY4d5_YJNA) and 
[GoogleDrive](https://drive.google.com/open?id=10n_eZ2ZyRjhRWFjxky7_PhcGHecDjKJ2).

We preprocessed these datasets and uploaded the preprocessed datasets into [BaiduYun](https://pan.baidu.com/s/1pCfF7xKQQmZ5XlrOFaSGrg) and 
[GoogleDrive](https://drive.google.com/open?id=1KDE5NJKfgOwc6RNEf9_F0ZhLQZ3Udjx5). If you want know how the original datasets
be preprocessed, you can reference our paper or look our code in this [release](https://github.com/leftthomas/CCCapsNet/tree/v0.0.1).
The preprocessed code has been removed in the current release for the simplify.

You needn't download the datasets by yourself, the code will download them automatically. If you encounter network issues, you can download 
all the datasets from the aforementioned cloud storage webs, and put the downloaded datasets into `data` directory.

## Usage

### Preprocess Data
```
python preprocess_data.py --data_type Newsgroups
optional arguments:
--data_type              dataset type [default value is 'IMDB'](choices:['IMDB', 'Newsgroups', 'Reuters', 'WebKB', 'DBPedia',
                         'AGNews', 'YahooAnswers', 'YelpReview', 'AmazonReview'])
--fine_grained           use fine grained class or not, it only works for Reuters, YelpReview and AmazonReview [default value is False]
```
The preprocessed datasets are in `data` directory.

### Train Model
```
python -m visdom.server -logging_level WARNING & python main.py --data_type newsgroups --num_epochs 300
optional arguments:
--data_type              dataset type [default value is 'imdb'](choices:['imdb', 'newsgroups', 'reuters', 'webkb', 
                         'dbpedia', 'agnews', 'yahoo', 'yelp', 'amazon'])
--fine_grained           use fine grained class or not, it only works for reuters, yelp and amazon [default value is False]
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
