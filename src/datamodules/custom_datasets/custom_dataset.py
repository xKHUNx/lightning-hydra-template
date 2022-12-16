import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.composition import Compose


class CustomDataset(Dataset):
    """Custom image dataset with support of YAML-defined transforms, the dataset directory
    consist of images grouped in folders with it's corresponding class.
    """
    def __init__(self, dataset_dirs, transform=None, target_transform=None, 
                 get_img_path=False, color_space="RGB", exclude=None):
        self.dataset_dirs = dataset_dirs
        self.class_dict = {'class_A': 0, 'class_B': 1, 'class_C': 2}
        self.img_paths = []
        self.img_labels = []
        self.transform = transform
        self.target_transform = target_transform
        self.get_img_path = get_img_path
        self.color_space = color_space
        self.exclude = exclude
        
        # No exclusions, two similar loops are created for
        # performance in no exclusion case
        if self.exclude is None:
            # Iterate all class
            for class_name, class_id in (self.class_dict.items()):
                # Iterate all folders and files
                for dataset_dir in dataset_dirs:
                    img_folder_path = "{}/{}".format(dataset_dir, class_name)
                    for (path, b, files) in os.walk(img_folder_path):
                        for fn in files:
                            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                                img_path = "{}/{}".format(path, fn)
                                # Store image path
                                self.img_paths.append(img_path)
                                # Store image class
                                self.img_labels.append(self.class_dict[class_name])
        else:
            # Iterate all class
            for class_name, class_id in (self.class_dict.items()):
                # Iterate all folders and files
                for dataset_dir in dataset_dirs:
                    img_folder_path = "{}/{}".format(dataset_dir, class_name)
                    for (path, b, files) in os.walk(img_folder_path):
                        for fn in files:
                            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                                img_path = "{}/{}".format(path, fn)
                                # Exclude images from exclusion list
                                if img_path not in self.exclude:
                                    # Store image path
                                    self.img_paths.append(img_path)
                                    # Store image class
                                    self.img_labels.append(self.class_dict[class_name])
                    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert(self.color_space)
        # Load label
        label = self.img_labels[idx]
        # Transform image
        if self.transform:
            for transform in self.transform:
                # Albumentation's transforms
                if isinstance(transform, Compose) or isinstance(transform, BasicTransform):
                    # Convert to numpy array
                    image = np.asarray(image)
                    # Apply albumentation transform
                    image = transform(image=image)['image']
                    # Convert back to PIL Image
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(image)
                # torchvision's transforms
                else:
                    image = transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.get_img_path:
            return img_path, image, label
        else:
            return image, label
        
    def add_sample(self, img_path, class_name):
        """Add a single sample to the data loader."""
        # Store image path
        self.img_paths.append(img_path)
        # Store image class
        self.img_labels.append(self.class_dict[class_name])
        
    def class_count(self, idxs=None):
        if isinstance(idxs, list):
            # Filter by indices
            ls = np.array(self.img_labels)[idxs]
            return Counter(ls)
        else:
            return Counter(self.img_labels)