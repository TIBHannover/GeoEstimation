from urllib.request import urlopen
import tarfile
import os
import sys

cur_dir = os.path.dirname(__file__)
if not os.path.exists(os.path.join(cur_dir, 'resources')):
    os.makedirs(os.path.join(cur_dir, 'resources'))
if not os.path.exists(os.path.join(cur_dir, 'models')):
    os.makedirs(os.path.join(cur_dir, 'models'))

urls = {
    # places365 model, weights & scene hierarchy
    'resnet152_places365.caffemodel':
    'http://places2.csail.mit.edu/models_places365/resnet152_places365.caffemodel',
    'deploy_resnet152_places365.prototxt':
    'https://raw.githubusercontent.com/CSAILVision/places365/master/deploy_resnet152_places365.prototxt',
    'scene_hierarchy_places365.csv':
    'https://spreadsheets.google.com/feeds/download/spreadsheets/Export?key=1H7ADoEIGgbF_eXh9kcJjCs5j_r3VJwke4nebhkdzksg&exportFormat=csv',
    # geo models
    'base_L_m.tar.gz':
    'https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/base_L_m.tar.gz',
    'base_M.tar.gz':
    'https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/base_M.tar.gz',
    'ISN_M_indoor.tar.gz':
    'https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/ISN_M_indoor.tar.gz',
    'ISN_M_natural.tar.gz':
    'https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/ISN_M_natural.tar.gz',
    'ISN_M_urban.tar.gz':
    'https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/ISN_M_urban.tar.gz'
}

BUFFER_SIZE = 256 * 1024

for filename, url in urls.items():
    print('#####')
    print('Get {} ...'.format(filename))

    if 'places' in filename:
        out_path = os.path.join(cur_dir, 'resources', filename)
    else:
        out_path = os.path.join(cur_dir, 'models', filename)

    if os.path.isfile(out_path):
        sys.stdout.write('File already exists. Overwrite? [Y/n]: ')
        choice = input().lower()
        if choice == 'n':
            continue

    response = urlopen(url)
    length = response.getheader('content-length')

    if length:
        length = int(length)
        blocksize = max(4096, length // 100)
    else:
        blocksize = BUFFER_SIZE  # if response does not contain content-length

    print('Downloading from URL: {}'.format(url))

    if 'places' in filename:
        out_path = os.path.join(cur_dir, 'resources', filename)
    else:
        out_path = os.path.join(cur_dir, 'models', filename)

    size = 0

    with open(out_path, 'wb') as f:
        while True:
            chunk = response.read(blocksize)
            if (not chunk):
                break
            f.write(chunk)
            size += len(chunk)

            if length:
                print(
                    '\tProgress: {:.0f}% ({:.2f} MB / {:.2f} MB)\r'.format(100 * size / length, size / (1024 * 1024),
                                                                           length / (1024 * 1024)),
                    end='')

        print()

        if 'models' in out_path:
            print('Extracting {}'.format(out_path))
            tf = tarfile.open(out_path)
            tf.extractall(path=os.path.dirname(out_path))

        print('DONE!')
