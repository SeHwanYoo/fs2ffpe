import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

from uvcgan2.consts import SPLIT_TRAIN

class ImageDomainFolder(Dataset):
    """Dataset structure introduced in a CycleGAN paper.

    Modified for FS->FFPE:
    - recursively scans nested slide folders;
    - follows symlinks;
    - supports linked datasets such as:
        trainA/TCGA-...-TS1/*.png
        trainA/TCGA-...-BS1/*.png
        trainA/BS/TCGA-...-TS1/*.png
    """

    def __init__(
        self, path,
        domain        = 'a',
        split         = SPLIT_TRAIN,
        transform     = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        subdir = split + domain.upper()

        self._path      = os.path.join(path, subdir)
        self._imgs      = ImageDomainFolder.find_images_in_dir(self._path)
        self._transform = transform

    @staticmethod
    def find_images_in_dir(path):
        extensions = {ext.lower() for ext in IMG_EXTENSIONS}

        if not os.path.exists(path):
            fallback = os.path.join(
                os.path.dirname(os.path.dirname(path)),
                os.path.basename(path)
            )
            if os.path.exists(fallback):
                path = fallback

        if not os.path.exists(path):
            return []

        result = []
        for root, _, fnames in os.walk(path, followlinks=True):
            for fname in fnames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in extensions:
                    continue
                result.append(os.path.join(root, fname))

        result.sort()
        return result

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, index):
        path   = self._imgs[index]
        result = default_loader(path)

        if self._transform is not None:
            result = self._transform(result)

        return result
