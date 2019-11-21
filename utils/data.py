import os
import requests
import io
import zipfile
import shutil

import urllib.request

from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_unzip(url, data_name, output_path, zip_file):
    with DownloadProgressBar(unit='B', unit_scale=True, ncols = 70, ascii=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=zip_file, reporthook=t.update_to)
    print("Begin to unzip.")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    print("Finished unzipping.")
    os.remove(zip_file)

def get_data(data_name,url,output_path):
#    assert os.path.isdir("/storage"), "You need a folder /storage!"
    if not os.path.isdir(output_path):
        os.system("mkdir -p {}".format(output_path))
        os.system("ls -lah /storage")
    assert os.path.isdir("{}".format(output_path)), "{} existiert nicht. Abbruch.".format(output_path,data_name)
    download_unzip(url, data_name, output_path, "{}/{}.zip".format(output_path,data_name))
    os.system("ls -lah {}".format(output_path))

def zip_up(folder):
    if os.path.isdir(folder):
        shutil.make_archive(folder, 'zip', folder)

    
if __name__=='__main__':
    url = "https://scanx.s3.eu-central-1.amazonaws.com/abc3.zip"
    url = "https://mntl.de/share/abc1.zip"
    data_name = "abc3"
    data_name = "abc1_short"
    output_path = "./data/blender_full"
    get_data(data_name,url,output_path)
