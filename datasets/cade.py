import os

from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract


def cade_dataset(directory='data/', train=False, test=False, extracted_name='cade',
                 check_files=['cade/cade-train-stemmed.txt'],
                 url='https://drive.google.com/uc?export=download&id=19zEVR6ZgwZg80kwEAIBmtOBgqibJQQKX'):
    """
    Load the Cade12 dataset (Version 1).

    The Cade12 dataset is corresponding to a subset of web pages extracted from the CADÊ Web Directory,
    which points to Brazilian web pages classified by human experts. The total number of training samples
    is 27,322 and testing 13,661.

    **Reference:** http://www.cade.com.br/

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        extracted_name (str, optional): Name of the extracted dataset directory.
        check_files (str, optional): Check if these files exist, then this download was successful.
        url (str, optional): URL of the dataset `tar.gz` file.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training dataset and
        test dataset in order if their respective boolean argument is true.

    Example:
        >>> from datasets import cade_dataset
        >>> train = cade_dataset(train=True)
        >>> train[0:2]
        [{
          'label': '08_cultura',
          'text': 'br br email arvores arvores http www apoio mascote natureza vida...'}
         {
          'label': '02_sociedade',
          'text': 'page frames browser support virtual araraquara shop '}]
    """
    download_file_maybe_extract(url=url, directory=directory, filename='cade.tar.gz', check_files=check_files)

    ret = []
    splits = [file_name for (requested, file_name) in
              [(train, 'cade-train-stemmed.txt'), (test, 'cade-test-stemmed.txt')] if requested]
    for file_name in splits:
        with open(os.path.join(directory, extracted_name, file_name), 'r', encoding='utf-8') as foo:
            examples = []
            for line in foo.readlines():
                label, text = line.split('\t')
                examples.append({'label': label, 'text': text.rstrip('\n')})
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
