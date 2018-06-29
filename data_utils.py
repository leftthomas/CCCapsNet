import os
import re
import shutil
import warnings
import zipfile
from os import makedirs
from os.path import exists
from sys import stdout

import requests

re_letter_number = re.compile(r'[^a-zA-Z0-9 ]')
re_number_letter = re.compile(r'^(\d+)([a-z]\w*)$')


def text_preprocess(text, data_type):
    if data_type == 'sogou' or data_type == 'yahoo' or data_type == 'yelp':
        # Remove \\n character
        text = text.replace('\\n', ' ')
    if data_type == 'imdb' or data_type == 'yahoo':
        # Remove <br /> character
        text = text.replace('<br />', ' ')
    if data_type not in ['newsgroups', 'reuters', 'webkb', 'cade']:
        # Turn punctuation, foreign word, etc. into SPACES word SPACES).
        text = re_letter_number.sub(lambda m: ' ' + m.group(0) + ' ', text)
        # Turn all letters to lowercase.
        text = text.lower()
        # Turn the number-letter word to single number and word (such as turn 2008year into 2008 year).
        text = ' '.join(' '.join(w for w in re_number_letter.match(word).groups())
                        if re_number_letter.match(word) else word for word in text.split())
        # Turn all numbers to single number (such as turn 789 into 7 8 9).
        text = ' '.join(' '.join(w for w in word) if word.isdigit() else word for word in text.split())
    # Substitute multiple SPACES by a single SPACE.
    text = ' '.join(text.split())
    return text


class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.
    """

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download"

    @staticmethod
    def download_file_from_google_drive(file_id, file_name, dest_path, overwrite=False):
        """
        Downloads a shared file from google drive into a given folder.
        Optionally unzips it.

        Args:
            file_id (str): the file identifier. You can obtain it from the sherable link.
            file_name (str): the file name.
            dest_path (str): the destination where to save the downloaded file.
            overwrite (bool): optional, if True forces re-download and overwrite.

        Returns:
            None
        """

        if not exists(dest_path):
            makedirs(dest_path)

        if not exists(os.path.join(dest_path, file_name)) or overwrite:

            session = requests.Session()

            print('Downloading {} into {}... '.format(file_name, dest_path), end='')
            stdout.flush()

            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)

            GoogleDriveDownloader._save_response_content(response, os.path.join(dest_path, file_name))
            print('Done.')

            try:
                print('Unzipping... ', end='')
                stdout.flush()
                with zipfile.ZipFile(os.path.join(dest_path, file_name), 'r') as zip_file:
                    for member in zip_file.namelist():
                        filename = os.path.basename(member)
                        # skip directories
                        if not filename:
                            continue
                        # copy file (taken from zipfile's extract)
                        source = zip_file.open(member)
                        target = open(os.path.join(dest_path, filename), 'wb')
                        with source, target:
                            shutil.copyfileobj(source, target)
                print('Done.')
            except zipfile.BadZipfile:
                warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(file_name))

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination):
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

