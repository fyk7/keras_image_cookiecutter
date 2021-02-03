# -*- coding: utf-8 -*-
import io
import click
import logging
import zipfile
import requests
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    # logger.info('making final data set from raw data')
    logger.info('download data')

    data_url = 'https://mytlab.org/wp/wp-content/uploads/2020/08/stationery.zip'
    # This time data was already preproccessed so just only download to data/proccessed
    _download_zip(data_url, output_filepath)


def _download_zip(url, save_path):
    res = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(res.content))
    z.extractall(save_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
