import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path_files):
    text_file_loc = os.path.join(path_files, 'name.txt')
    paths, size = make_dataset_txt(text_file_loc)
    '''
    if path_files.find('name.txt') != -1:
        paths, size = make_dataset_txt(path_files)
    else:
        paths, size = make_dataset_dir(path_files)
    '''
    return paths, size


def make_dataset_txt(files):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        img_paths.append(path)

    return img_paths, len(img_paths)


def make_dataset_dir(dir):
    """
    :param dir: directory paths that store the image
    :return: image paths and sizes
    """
    img_paths = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img_paths.append(path)

    return img_paths, len(img_paths)
