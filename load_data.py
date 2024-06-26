
import os
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch
import json
from tqdm import tqdm


DATA_DIR = "../../data/biopsies_s1.0_anon_data/"

P53_CLASS_NAMES = ["Wildtype", "Overexpression", "Nullmutation", "Doubleclone"]


###############################################################################
# UTILS
###############################################################################

def convert_status_to_presence(labels):
    """
    Convert labels, which are 0: wildtype, 1: overexpression, 2: nullmutation, 3: doubleclone
    to an array of size 2, where the first element is 1 if overexpression is present, and the second element is 1 if nullmutation is present.
    Wildtype is [0, 0], overexpression is [1, 0], nullmutation is [0, 1], doubleclone is [1, 1].
    """
    return torch.stack([(labels == 1) | (labels == 3), (labels == 2) | (labels == 3)], dim=1).float()

def convert_presence_to_status(outputs):
    """
    Convert presence, which is an array of size 2, where the first element is 1 if overexpression is present, and the second element is 1 if nullmutation is present,
    to pred labels, where 0: wildtype, 1: overexpression, 2: nullmutation, 3: doubleclone.
    """
    status = torch.zeros((len(outputs)), dtype=torch.uint8, device=outputs.device)
    status[outputs[:, 0] > 0.5] += 1
    status[outputs[:, 1] > 0.5] += 2
    return status

def convert_presence_probs_to_status_probs(presence_probs):
    """Presence probabilities has a last dim of 2, representing the probabilities of the two mutations.
    This function converts these probabilities to status probabilities, where the last dim 
    is 4, and represents the probability of the status 
    (neither, only the first, only the second, both)."""
    status_probs = torch.zeros(presence_probs.shape[:-1] + (4,), device=presence_probs.device)
    status_probs[..., 0] = (1-presence_probs[..., 0]) * (1-presence_probs[..., 1])
    status_probs[..., 1] = presence_probs[..., 0] * (1-presence_probs[..., 1])
    status_probs[..., 2] = (1-presence_probs[..., 0]) * presence_probs[..., 1]
    status_probs[..., 3] = presence_probs[..., 0] * presence_probs[..., 1]
    return status_probs


###############################################################################
# TRANSFORMS
###############################################################################

class RandomRotation90:
    def __init__(self):
        self.degrees = [0., 90., 180., 270.]

    def __call__(self, img):
        degree = np.random.choice(self.degrees)
        return torchvision.transforms.functional.rotate(img, degree)


BASIC_TRANSFORM = torchvision.transforms.Compose([
    # Convert from tensor to PIL image
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomVerticalFlip(),
    RandomRotation90(),
    torchvision.transforms.ToTensor(),
])

def get_rotation_transform(fill=128):
    return torchvision.transforms.Compose([
        # Convert from tensor to PIL image
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(360, fill=fill),
        torchvision.transforms.ToTensor(),
    ])

EXTRA_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.RandomHorizontalFlip(),
    RandomRotation90(),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.05),
    torchvision.transforms.RandomAdjustSharpness(sharpness_factor=0), # Adjusts sharpness of the image (1.0 is the original image, 2.0 is very sharp, 0.0 is blurry)
    torchvision.transforms.ToTensor(),
])

def get_extra_rotation_transform(fill=128):
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(360, fill=fill),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.05),
        torchvision.transforms.RandomAdjustSharpness(sharpness_factor=0), # Adjusts sharpness of the image (1.0 is the original image, 2.0 is very sharp, 0.0 is blurry)
        torchvision.transforms.ToTensor(),
])

NORMALIZE = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])
IMGNET_NORMALIZE = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TRANSFORMS = {
    "basic": BASIC_TRANSFORM,
    "normalize": NORMALIZE,
    "imgnet_normalize": IMGNET_NORMALIZE,
    "basic_normalize": torchvision.transforms.Compose([BASIC_TRANSFORM, NORMALIZE]),
    "extra": EXTRA_TRANSFORM,
    "extra_normalize": torchvision.transforms.Compose([EXTRA_TRANSFORM, NORMALIZE]),
    "rotation_grey": get_rotation_transform(fill=128),
    "rotation_black": get_rotation_transform(fill=0),
    "rotation_grey_normalize": torchvision.transforms.Compose([get_rotation_transform(fill=128), NORMALIZE]),
    "rotation_black_normalize": torchvision.transforms.Compose([get_rotation_transform(fill=0), NORMALIZE]),
    "extra_rotation_grey": get_extra_rotation_transform(fill=128),
    "extra_rotation_black": get_extra_rotation_transform(fill=0),
    "extra_rotation_grey_normalize": torchvision.transforms.Compose([get_extra_rotation_transform(fill=128), NORMALIZE]),
    "extra_rotation_black_normalize": torchvision.transforms.Compose([get_extra_rotation_transform(fill=0), NORMALIZE]),
}


###############################################################################
# DATALOADER UTILS
###############################################################################

def make_weights_for_balanced_classes(dataset, weights_factor=None):
    """Return a list of weights for each image in the dataset, based on the class distribution.
    dataset must have .num_classes attribute. dataset labels must be integers in the range [0, num_classes-1].
    
    weights_factor is an array of length num_classes that is multiplied with the weights of each class to scale them manually in addition to the balancing.
    If weights_factor is None, the weights are not manually scaled."""
    # Make a list of the class labels and count the number of images in each class
    count = [0] * dataset.num_classes
    for tup in dataset:
        label = tup[1]
        count[label] += 1
    
    # Compute the class weights
    weight_per_class = np.zeros(dataset.num_classes)
    N = float(sum(count))
    for i in range(dataset.num_classes):
        if count[i] == 0:
            weight_per_class[i] = 0
        else:
            weight_per_class[i] = N/float(count[i])
    
    # Multiply the weights by weights_factor
    if weights_factor is not None:
        weight_per_class = weights_factor * weight_per_class
    
    # Compute the weights for each image (this will be used in the sampler)
    weights = np.zeros(len(dataset))
    for i, tup in enumerate(dataset):
        label = tup[1]
        weights[i] = weight_per_class[label]
    
    return weights


def make_weights_for_balanced_masks(dataset, weights_factor=None):
    """Return a list of weights for each image in the dataset, based on the mask distribution.
    Labels are masks, and we separate them based on whether they're empty or not.
    
    weights_factor is an array of length num_classes that is multiplied with the weights of each class to scale them manually in addition to the balancing.
    If weights_factor is None, the weights are not manually scaled."""
    # Make a list of the class labels and count the number of images in each class
    assert dataset.num_classes == 2, "Only works for binary masks"
    count = [0] * dataset.num_classes
    for tup in dataset:
        mask = tup[1]
        label = 0 if mask.mean() == 0 else 1
        count[label] += 1
    
    # Compute the class weights
    weight_per_class = np.zeros(dataset.num_classes)
    N = float(sum(count))
    for i in range(dataset.num_classes):
        if count[i] == 0:
            weight_per_class[i] = 0
        else:
            weight_per_class[i] = N/float(count[i])
    
    # Multiply the weights by weights_factor
    if weights_factor is not None:
        weight_per_class = weights_factor * weight_per_class
    
    # Compute the weights for each image (this will be used in the sampler)
    weights = np.zeros(len(dataset))
    for i, tup in enumerate(dataset):
        mask = tup[1]
        label = 0 if mask.mean() == 0 else 1
        weights[i] = weight_per_class[label]
    
    return weights


def get_balanced_dataloader(dataset, batch_size=32, weights_factor=None, num_workers=0, masks=False):
    """Return a dataloader that keeps the class balance. weights_factor is an array of length num_classes that
    is multiplied with the weights of each class to scale them manually in addition to the balancing.
    If weights_factor is None, the weights are not manually scaled.
    
    Shuffle is set to False because the sampler already shuffles the data."""
    if not masks:
        weights = make_weights_for_balanced_classes(dataset, weights_factor=weights_factor)
    else:
        weights = make_weights_for_balanced_masks(dataset, weights_factor=weights_factor)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, shuffle=False)
    return dataloader


###############################################################################
# DATASETS
###############################################################################

# Write a Dataset class for the biopsies
class BiopsyDataset(torch.utils.data.Dataset):
    """Biopsy dataset. getitem returns (image, label) tuples."""
    def __init__(self, root_dir, labels_filename="train", transform=None, class_names=None, size=256, data_limit=None, latents_path=None, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the biopsies.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.latents_path = latents_path
        if latents_path:
            print("Loading from latents")
            print("transforms are ignored")
        
        # Read the labels
        labels_file = os.path.join(root_dir, f"{labels_filename}.csv")
        self.labels = np.loadtxt(labels_file, delimiter=",", skiprows=1)
        self.labels = self.labels.astype(int)
        if data_limit:
            self.labels = self.labels[:data_limit]

        self.num_classes = len(np.unique(self.labels[:, 1]))
        if class_names and self.num_classes != len(class_names):
            self.num_classes = len(class_names)
        self.class_names = class_names

        self.class_distribution = {i: np.sum(self.labels[:, 1] == i) for i in range(self.num_classes)}
        if class_names:
            self.class_distribution = {class_names[i]: self.class_distribution[i] for i in range(self.num_classes)}
        print("Class distribution: ", self.class_distribution)

        if latents_path:
            self.latents = torch.load(os.path.join(root_dir, latents_path)) # (n, 4, ftrs)
            self.latents = self.latents[self.labels[:, 0]] # Only keep the latents for the images in the dataset
            return

        # If self.imgs is already pickled, load it
        imgs_file = os.path.join(root_dir, f"imgs_{labels_filename}_{size}.pt")
        if os.path.exists(imgs_file):
            self.imgs = torch.load(imgs_file)
            print("Loaded images from file")
        else:
            print("Processing images")
            # Read the images
            imgs_dir = os.path.join(root_dir, "biopsies")
            self.imgs = []
            for idx in tqdm(self.labels[:, 0]):
                img_file = os.path.join(imgs_dir, f"{idx}.png")
                img = plt.imread(img_file)
                img = torch.tensor(img).permute(2, 0, 1).float()
                img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(size, size)).squeeze(0)
                self.imgs.append(img)
            
            # Save the images to a file
            torch.save(self.imgs, imgs_file)

        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = int(self.labels[idx, 1])

        if self.latents_path:
            return (self.latents[idx], label)

        image = self.imgs[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return (image, label)

    def plot_example_grid(self, n=5, random=True, figsize=(15, 15)):
        """Plot a grid with n*n examples."""
        if self.latents_path:
            print("Cannot plot examples from latents")
            return

        if random:
            indices = np.random.choice(len(self), size=n*n, replace=False)
        else:
            indices = np.arange(n*n)
        
        fig, axs = plt.subplots(n, n, figsize=figsize)
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(self[indices[i]][0].permute(1, 2, 0))
            ax.set_title(self.class_names[self[indices[i]][1]])
            ax.axis("off")
        plt.show()


def validate_label_from_patch_labels(label, patch_labels):
    """Based on patch labels, validate the label of the bag."""
    if label == 0:
        # assert np.all(patch_labels == 0), "Wildtype bag should have all wildtype patches"
        if not np.all(patch_labels == 0):
            print("Wildtype bag should have all wildtype patches")
    elif label == 1:
        # assert np.all(patch_labels[:, 1] == 0), "Overexpression bag should have no nullmutation patches"
        if not np.all(patch_labels[:, 1] == 0):
            print("Overexpression bag should have no nullmutation patches")
    elif label == 2:
        # assert np.any(patch_labels[:, 1] > 0), "Nullmutation bag should have at least one nullmutation patch"
        if not np.any(patch_labels[:, 1] > 0):
            print("Nullmutation bag should have at least one nullmutation patch")
    # elif label == 3: # We actually don't have annotations for doubleclone
    #     assert np.any(patch_labels[:, 1] > 0), "Doubleclone bag should have at least one nullmutation patch"


class BagDataset(torch.utils.data.Dataset):
    """
    Biopsy patch bag dataset. getitem returns (bag, label, patch_labels) tuples.
    grid_spacing: the distance between the centers of the patches. default 256
    size: the size of the patches. default 64

    If grid_spacing and size are the same, wsi spacing is 1.0 (because the source data is spacing 1.0)
    If grid_spacing is 256 and size is 64, wsi spacing is 4.0

    getitem returns (bag, label, patch_labels) tuples, 
        where bag is a tensor of shape (n_patches, 3, size, size),
        label is the label of the bag,
        patch_labels is a tensor of shape (n_patches, 2) with the patch labels (a percentage score for how much of the patch is abnormal).
    """
    def __init__(self, root_dir, labels_filename="train", transform=None, class_names=None, 
                 size=64, grid_spacing=256, data_limit=None, latents_path=None, 
                 mix_bags=0,
                 **kwargs):
        """
        Args:
            root_dir (string): Directory with all the biopsies.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.quantile_range_threshold = 0.10
        self.latents_path = latents_path
        self.mix_bags = mix_bags
        if latents_path:
            print("Loading from latents")
            print("transforms are ignored")

        assert size <= grid_spacing, "size must be less than or equal to grid_spacing"

        # Read the labels
        labels_file = os.path.join(root_dir, f"{labels_filename}.csv")
        self.labels = np.loadtxt(labels_file, delimiter=",", skiprows=1)
        self.labels = self.labels.astype(int)
        if data_limit:
            self.labels = self.labels[:data_limit]
        
        self.num_classes = len(np.unique(self.labels[:, 1]))
        if class_names and self.num_classes != len(class_names):
            self.num_classes = len(class_names)
        self.class_names = class_names

        self.class_distribution = {i: np.sum(self.labels[:, 1] == i) for i in range(self.num_classes)}
        if class_names:
            self.class_distribution = {class_names[i]: self.class_distribution[i] for i in range(self.num_classes)}
        print("Class distribution: ", self.class_distribution)

        if latents_path:
            # self.indices,self.patch_labels,self.bag_sizes = torch.load(os.path.join(root_dir, f"bags_meta_{labels_filename}_gs{grid_spacing}.pt"))
            # self.labels = self.labels[self.indices]

            # Validate the labels based on the patch labels
            # for i in range(len(self.labels)):
            #    validate_label_from_patch_labels(self.labels[i, 1], self.patch_labels[i])

            latents = torch.load(os.path.join(root_dir, latents_path)) # bag_idx: (n, 1, ftrs)
            self.latents = []
            for i, idx in enumerate(self.labels[:, 0]):
                patch_latents = latents[idx]
                self.latents.append(patch_latents) # Only keep the latents for the images in the dataset
            assert len(self.latents) == len(self.labels), f"Latents and labels don't match: {len(self.latents)} != {len(self.labels)}"

            if mix_bags:
                self.original_labels = self.labels.copy()
                # Synthetically add doubleclone bags by adding overexpression and nullmutation bags together
                # However, we don't want to store unnecessary bags, so we only store the indices of the bags
                # So we make self.latents_indices of shape (N+M, K), where N is the number of bags, M is the number of doubleclone bags,
                # and K is the number of bag indices that are concatenated together (so 1 for all bags except doubleclone, 2 for doubleclone)
                self.bag_indices = [[i] for i in range(len(self.labels))]

                oe_indices = np.where(self.labels[:, 1] == 1)[0]
                nm_indices = np.where(self.labels[:, 1] == 2)[0]

                mixed_bags = []
                for i in range(len(oe_indices)):
                    for j in range(len(nm_indices)):
                        mixed_bags.append((oe_indices[i], nm_indices[j]))

                # Choose self.mix_bags random mixed bags
                if self.mix_bags > len(mixed_bags):
                    print(f"Warning: only {len(mixed_bags)} mixed bags available")
                    self.mix_bags = len(mixed_bags)
                else:
                    np.random.seed(0)
                    # a must be 1-D
                    mixed_bag_indices = np.random.choice(len(mixed_bags), self.mix_bags, replace=False)
                    mixed_bags = [mixed_bags[i] for i in mixed_bag_indices]
                
                self.bag_indices += mixed_bags
                # Add the doubleclone labels (-1, 3)
                self.labels = np.concatenate((self.labels, np.array([[-1, 3]]*len(mixed_bags))), axis=0)

                print(f"Added {len(mixed_bags)} doubleclone bags")           

            return
        
        # If self.bags is already pickled, load it
        bags_file = os.path.join(root_dir, f"bags_{labels_filename}_{size}_{grid_spacing}.pt")
        if os.path.exists(bags_file):
            self.bags, self.indices, self.patch_labels, self.bag_sizes = torch.load(bags_file)
            print("Loaded bags from file")
        else:
            print("Creating bags")

            resize_factor = size / grid_spacing

            # Read the images and turn into patches
            self.bags = []
            self.indices = []
            self.patch_labels = []
            self.bag_sizes = []
            for i, (idx, label) in enumerate(tqdm(self.labels)):
                img_file = os.path.join(root_dir, "biopsies", f"{idx}.png")
                img = plt.imread(img_file)
                img = torch.tensor(img).permute(2, 0, 1).float() # (3, h, w)

                # Resize the img so that the height and width are multiples of grid_spacing, 
                # rounded to the nearest multiple.
                # This prevents the last row and column of patches from being cut off.
                # We also resize the image by the resize_factor instead of each patch separately
                new_size = (max(round(img.shape[1]*resize_factor/size), 1)*size, 
                            max(round(img.shape[2]*resize_factor/size), 1)*size)
                img = torch.nn.functional.interpolate(img.unsqueeze(0), size=new_size, 
                    mode='bilinear', align_corners=False)[0]

                patches = torch.nn.functional.unfold(img.unsqueeze(0), 
                    kernel_size=(size, size), 
                    stride=(size, size)) # (1, 3*size*size, n_patches)
                patches = patches.permute(0, 2, 1).reshape(-1, 3, size, size) # (n_patches, 3, size, size)
                
                # Remove patches with low difference between the 75th and 1st percentile
                qranges = torch.quantile(patches.view(patches.shape[0], -1), 0.75, dim=-1) - \
                          torch.quantile(patches.view(patches.shape[0], -1), 0.01, dim=-1)
                non_empty_idx = torch.where(qranges >= self.quantile_range_threshold)[0]
                patches = patches[non_empty_idx]
                patches = torch.nn.functional.interpolate(patches, size=(size, size))

                if len(patches) == 0:
                    print(f"Empty bag: {idx}")
                    continue
                self.bags.append(patches)
                self.indices.append(i)
                self.bag_sizes.append(len(patches))

                patch_labels = np.zeros((len(patches), 2))
                patch_labels.fill(np.nan)

                if label == 0: # If wildtype, all patches are wildtype
                    patch_labels[:, :] = 0
                elif label == 1: # If overexpression, no patches are nullmutation
                    patch_labels[:, 1] = 0
                elif label in [2,3]: 
                    # If nullmutation, no patches are overexpression
                    if label == 2:
                        patch_labels[:, 0] = 0
                    
                    # Use annotation masks for more accurate labels
                    mask_file = os.path.join(root_dir, "masks", f"{idx}.png")
                    if os.path.exists(mask_file):
                        mask = plt.imread(mask_file)
                        mask = torch.tensor(mask).float() # (h, w)
                        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                            size=new_size, mode='nearest')[0][0]
                        mask_patches = torch.nn.functional.unfold(mask.unsqueeze(0), 
                            kernel_size=(size, size), 
                            stride=(size, size)) # (size*size, n_patches)
                        mask_patches = mask.permute(1, 0).reshape(-1, size, size) # (n_patches, size, size)
                        mask_patches = mask_patches[non_empty_idx]
                        # Set the patch labels to the mean of the mask
                        patch_labels[:, 1] = mask_patches.mean(dim=(1, 2))
                self.patch_labels.append(patch_labels)

            # Save the bags to a file
            torch.save((self.bags, self.indices, self.patch_labels, self.bag_sizes), bags_file)
                
        # self.bags is potentially shorter than self.labels, so we need to filter it
        self.labels = self.labels[self.indices]
        assert len(self.bags) == len(self.labels), f"Bags and labels don't match: {len(self.bags)} != {len(self.labels)}"
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = int(self.labels[idx, 1])

        if self.latents_path:
            if self.mix_bags:
                bag_indices = self.bag_indices[idx]
                bag = torch.cat([self.latents[i] for i in bag_indices], dim=0)
                return (bag, label)
            return (self.latents[idx], label)

        bag = self.bags[idx]
        patch_labels = self.patch_labels[idx]
        
        if self.transform:
            # Apply the same transform to all patches in the bag
            bag = torch.stack([self.transform(patch) for patch in bag])
        
        return (bag, label, patch_labels)

    def plot_example_grid(self, n=5, random=True, figsize=(15, 15)):
        """Plot a grid with n*n examples from different bags."""
        if self.latents_path:
            print("Cannot plot examples from latents")
            return

        if random:
            indices = np.random.choice(len(self), size=n*n, replace=False)
        else:
            indices = np.arange(n*n)
        
        fig, axs = plt.subplots(n, n, figsize=figsize)
        for i, ax in enumerate(axs.flatten()):
            bag, label, patch_labels = self[indices[i]]
            idx = np.random.choice(len(bag))
            patch = bag[idx]
            patch_label = patch_labels[idx]
            ax.imshow(patch.permute(1, 2, 0))
            ax.set_title(self.class_names[label]+f"\n[{patch_label[0]:.2f}, {patch_label[1]:.2f}]")
            ax.axis("off")
        plt.show()


class Nullmutation(torch.utils.data.Dataset):
    """
    Biopsy patch dataset with nullmutation annotations. getitem returns (patch, patch_label) tuples.
    grid_spacing: the distance between the centers of the patches. default 256
    size: the size of the patches. default 64

    If grid_spacing and size are the same, wsi spacing is 1.0 (because the source data is spacing 1.0)
    If grid_spacing is 256 and size is 64, wsi spacing is 4.0

    getitem returns (bag, label, patch_labels) tuples, 
        where patch is a tensor of shape (3, size, size),
        patch_label is a binary mask denoting nullmutation (size, size)
    """
    def __init__(self, root_dir, labels_filename="train", transform=None, size=64, grid_spacing=256, data_limit=None,
                 return_mask_label=True, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the biopsies.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.quantile_range_threshold = 0.10
        self.return_mask_label = return_mask_label

        assert size <= grid_spacing, "size must be less than or equal to grid_spacing"

        # Read the labels
        labels_file = os.path.join(root_dir, f"{labels_filename}.csv")
        self.labels = np.loadtxt(labels_file, delimiter=",", skiprows=1)
        self.labels = self.labels.astype(int)
        if data_limit:
            self.labels = self.labels[:data_limit]
        
        # If self.patches is already pickled, load it
        patches_file = os.path.join(root_dir, f"NM-patches_{labels_filename}_{size}_{grid_spacing}.pt")
        if os.path.exists(patches_file):
            self.patches, self.patch_labels = torch.load(patches_file)
            print("Loaded patches from file")
        else:
            print("Processing patches and annotations")

            resize_factor = size / grid_spacing

            # Read the images and turn into patches
            self.patches = []
            self.patch_labels = []
            for idx, label in tqdm(self.labels):
                img_file = os.path.join(root_dir, "biopsies", f"{idx}.png")
                mask_file = os.path.join(root_dir, "masks", f"{idx}.png")
                # if label not in [0,2,3]:
                #     continue
                if label in [2,3] and not os.path.exists(mask_file):
                    continue

                img = plt.imread(img_file)
                img = torch.tensor(img).permute(2, 0, 1).float() # (3, h, w)

                # Resize the img so that the height and width are multiples of grid_spacing, 
                # rounded to the nearest multiple.
                # This prevents the last row and column of patches from being cut off.
                # We also resize the image by the resize_factor instead of each patch separately
                new_size = (max(round(img.shape[1]*resize_factor/size), 1)*size, 
                            max(round(img.shape[2]*resize_factor/size), 1)*size)
                img = torch.nn.functional.interpolate(img.unsqueeze(0), size=new_size, 
                    mode='bilinear', align_corners=False)[0]

                patches = torch.nn.functional.unfold(img.unsqueeze(0), 
                    kernel_size=(size, size), 
                    stride=(size, size)) # (1, 3*size*size, n_patches)
                patches = patches.permute(0, 2, 1).reshape(-1, 3, size, size) # (n_patches, 3, size, size)
                
                # Remove patches with low difference between the 75th and 1st percentile
                qranges = torch.quantile(patches.view(patches.shape[0], -1), 0.75, dim=-1) - \
                          torch.quantile(patches.view(patches.shape[0], -1), 0.01, dim=-1)
                non_empty_idx = torch.where(qranges >= self.quantile_range_threshold)[0]
                patches = patches[non_empty_idx]

                if len(patches) == 0:
                    print(f"Empty bag: {idx}")
                    continue

                patch_indices = []
                if label in [0,1]: # If wildtype or overexpression, no patches are nullmutation
                    mask_patches = torch.zeros_like(patches)[:,0:1,:,:]
                elif label in [2,3]:
                    mask = plt.imread(mask_file)
                    mask = torch.tensor(mask).float() # (h, w)
                    mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                        size=new_size, mode='nearest')[0][0]
                    mask_patches = torch.nn.functional.unfold(mask.unsqueeze(0), 
                        kernel_size=(size, size), 
                        stride=(size, size)) # (size*size, n_patches)
                    mask_patches = mask_patches.permute(1, 0).reshape(-1, 1, size, size) # (n_patches, 1, size, size)
                    mask_patches = mask_patches[non_empty_idx]

                    if label == 3:
                        # For doubleclone only select patches with nullmutation
                        patch_indices = torch.nonzero(mask_patches.mean(dim=(1, 2)) > 0)[0]
                        patches = patches[patch_indices]
                        mask_patches = mask_patches[patch_indices]
                self.patches.append(patches)
                self.patch_labels.append(mask_patches)
            self.patches = torch.cat(tuple(self.patches), dim=0) # (total n_patches, 3, size, size)
            self.patch_labels = torch.cat(tuple(self.patch_labels), dim=0) # (total n_patches, 1, size, size)

            # Save the bags to a file
            torch.save((self.patches, self.patch_labels), patches_file)

        means = self.patch_labels.mean(dim=(-1,-2))[:,0]
        self.wt_indices = torch.nonzero(means == 0)[:,0]
        self.nm_indices = torch.nonzero(means)[:,0]
        self.class_distribution = {
            "wildtype": len(self.wt_indices),
            "any nullmutation": len(self.nm_indices)
        }
        print("Class distribution: ", self.class_distribution)
        
        
    def __len__(self):
        return len(self.patch_labels)
    
    def __getitem__(self, idx):
        patch = self.patches[idx] # (3, size, size)
        patch_label = self.patch_labels[idx] # (1, size, size)
        
        if self.transform:
            patch = self.transform(patch)
            if self.return_mask_label:
                print("WARNING: If using masks as label, rotation transforms will make it incorrect")
                # TODO: Make this work for the pair, because rotation would ruin the mask
            # pair = torch.cat((patch, patch_label), dim=0)
        
        if not self.return_mask_label:
            patch_label = patch_label.mean(dim=(-2,-1))

        return (patch, patch_label)

    def plot_example_grid(self, n=4, random=True, figsize=(15, 15)):
        """Plot a grid with n*n examples."""
        assert n%2 == 0, "n must be even"
        if random:
            wt_indices = self.wt_indices[np.random.choice(len(self.wt_indices), size=n*n//2, replace=False)]
            nm_indices = self.nm_indices[np.random.choice(len(self.nm_indices), size=n*n//2, replace=False)]
        else:
            wt_indices = self.wt_indices[:n*n//2]
            nm_indices = self.nm_indices[:n*n//2]
        
        fig, axs = plt.subplots(n, n, figsize=figsize)
        for i, ax in enumerate(axs.flatten()):
            if i < n*n//2:
                label = "wildtype"
                indices = wt_indices
                idx = i
            else:
                label = "nullmutation"
                indices = nm_indices
                idx = i - n*n//2
                # print(indices[idx], self[indices[idx]][1].mean())
                
            patch, patch_label = self[indices[idx]]
            ax.imshow(patch.squeeze().permute(1, 2, 0))
            ax.set_title(f"Wildtype")
            if label == "nullmutation":
                ax.imshow(patch_label[0], alpha=0.5, vmin=0, vmax=1, cmap="Reds")
                ax.set_title(f"Nullmutation [{patch_label.mean():.2f}]")
            ax.axis("off")
        plt.show()



class PatchDataset(torch.utils.data.Dataset):
    """
    Biopsy patch dataset with overexpression and nullmutation labels.

    getitem returns (patch latent, patch label) tuples, 
        where patch latent is a vector of shape (2048),
        patch label is a vector of shape (2) with the overexpression and nullmutation labels.
    """
    def __init__(self, root_dir, labels_filename="train", data_limit=None, int_labels=True, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the biopsies.
        """
        self.root_dir = root_dir
        self.int_labels = int_labels

        # Read the labels
        labels_file = os.path.join(root_dir, f"{labels_filename}.csv")
        self.labels = np.loadtxt(labels_file, delimiter=",", skiprows=1)
        self.labels = self.labels.astype(int)
        if data_limit:
            self.labels = self.labels[:data_limit]
        
        # bag_latents_file = os.path.join(root_dir, f"bag_latents_gs256_retccl_relaxed.pt")
        # bag_patch_indices_file = os.path.join(root_dir, f"non_empty_patch_indices_gs256_relaxed.pt")
        bag_latents_file = os.path.join(root_dir, f"bag_latents_gs256_retccl__backup.pt")
        bag_patch_indices_file = os.path.join(root_dir, f"non_empty_patch_indices_gs256.pt")
        oe_patch_indices_file = os.path.join(root_dir, f"oe_patch_indices_gs256.pt")
        nm_patch_indices_file = os.path.join(root_dir, f"nm_patch_indices_gs256.pt")
        for file in [bag_latents_file, bag_patch_indices_file, oe_patch_indices_file, nm_patch_indices_file]:
            if not os.path.exists(file):
                print("File does not exist at", file)
                return
        
        bag_latents       = torch.load(bag_latents_file)        # dict of idx: [(ftrs,), (ftrs,), ...]
        bag_patch_indices = torch.load(bag_patch_indices_file)  # dict of idx: [patch_idx, patch_idx, ...]
        oe_patch_indices  = torch.load(oe_patch_indices_file)   # dict of idx: [patch_idx, patch_idx, ...]
        nm_patch_indices  = torch.load(nm_patch_indices_file)   # dict of idx: [patch_idx, patch_idx, ...]

        self.patch_latents = []
        self.patch_labels = []
        for idx, label in tqdm(self.labels):
            if label == 0: # Wildtype, use all patches
                self.patch_latents.extend(bag_latents[idx])
                self.patch_labels.extend([(0, 0)] * len(bag_latents[idx]))
            elif label == 1: # Overexpression, use only overexpression patches
                oe_indices = [bag_patch_indices[idx].tolist().index(oe_idx) for oe_idx in oe_patch_indices[idx] if oe_idx in bag_patch_indices[idx]]
                self.patch_latents.extend([bag_latents[idx][i] for i in oe_indices])
                self.patch_labels.extend([(1, 0)] * len(oe_indices))
            elif label == 2: # Nullmutation, use only nullmutation patches
                nm_indices = [bag_patch_indices[idx].tolist().index(nm_idx) for nm_idx in nm_patch_indices[idx] if nm_idx in bag_patch_indices[idx]]
                self.patch_latents.extend([bag_latents[idx][i] for i in nm_indices])
                self.patch_labels.extend([(0, 1)] * len(nm_indices))
            elif label == 3: # Both, use patches from both overexpression and nullmutation
                oe_indices = [bag_patch_indices[idx].tolist().index(oe_idx) for oe_idx in oe_patch_indices[idx] if oe_idx in bag_patch_indices[idx]]
                nm_indices = [bag_patch_indices[idx].tolist().index(nm_idx) for nm_idx in nm_patch_indices[idx] if nm_idx in bag_patch_indices[idx]]
                self.patch_latents.extend([bag_latents[idx][i] for i in oe_indices + nm_indices])
                self.patch_labels.extend([(1, 0)] * len(oe_indices) + [(0, 1)] * len(nm_indices))
        self.patch_latents = torch.stack(self.patch_latents) # (n, ftrs)
        self.patch_labels = torch.tensor(self.patch_labels).float() # (n, 2)

        self.wt_indices = torch.nonzero((self.patch_labels[:, 0] == 0) & (self.patch_labels[:, 1] == 0)).squeeze()
        self.oe_indices = torch.nonzero(self.patch_labels[:, 0] == 1).squeeze()
        self.nm_indices = torch.nonzero(self.patch_labels[:, 1] == 1).squeeze()
        self.class_distribution = {
            "wildtype": len(self.wt_indices),
            "any overexpression": len(self.oe_indices),
            "any nullmutation": len(self.nm_indices),
            "total": len(self.patch_labels)
        }
        print("Class distribution: ", self.class_distribution)

        if self.int_labels: # change (0,0) to 0, (1,0) to 1, (0,1) to 2
            self.patch_labels = (self.patch_labels[:, 0] + 2*self.patch_labels[:, 1]).long()
        
        
    def __len__(self):
        return len(self.patch_labels)
    
    def __getitem__(self, idx):
        patch_latent = self.patch_latents[idx] # (ftrs,)
        patch_label = self.patch_labels[idx] # (2,)

        return (patch_latent, patch_label)

    def plot_example_grid(self, n=3, random=True, figsize=(15, 15)):
        """Plot a grid with n*n examples."""
        print("Cannot plot examples from latents")
        return

        assert n%3 == 0, "n must be divisible by 3"
        if random:
            wt_indices = self.wt_indices[np.random.choice(len(self.wt_indices), size=n*n//2, replace=False)]
            oe_indices = self.oe_indices[np.random.choice(len(self.oe_indices), size=n*n//2, replace=False)]
            nm_indices = self.nm_indices[np.random.choice(len(self.nm_indices), size=n*n//2, replace=False)]
        else:
            wt_indices = self.wt_indices[:n*n//2]
            oe_indices = self.oe_indices[:n*n//2]
            nm_indices = self.nm_indices[:n*n//2]
        
        fig, axs = plt.subplots(n, n, figsize=figsize)
        for i, ax in enumerate(axs.flatten()):
            if i < n*n//3:
                label = "wildtype"
                indices = wt_indices
                idx = i
            elif i < 2*n*n//3:
                label = "overexpression"
                indices = oe_indices
                idx = i - n*n//3
            else:
                label = "nullmutation"
                indices = nm_indices
                idx = i - 2*n*n//3
                # print(indices[idx], self[indices[idx]][1].mean())
                
            patch, patch_label = self[indices[idx]]
            ax.imshow(patch.squeeze().permute(1, 2, 0))
            ax.set_title(f"Wildtype")
            if label == "nullmutation":
                ax.imshow(patch_label[0], alpha=0.5, vmin=0, vmax=1, cmap="Reds")
                ax.set_title(f"Nullmutation [{patch_label.mean():.2f}]")
            ax.axis("off")
        plt.show()



class BiopsyKeepScaleDataset(torch.utils.data.Dataset):
    """Biopsy dataset. getitem returns (image, label) tuples."""
    def __init__(self, root_dir, labels_filename="test", transform=None, class_names=None, spacing=4, size_limit=4096, data_limit=None, latents_path=None, fill_bg=0.5, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the biopsies.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.latents_path = latents_path
        if latents_path:
            print("Loading from latents")
            print("transforms are ignored")

        self.spacing = spacing
        self.size_limit = size_limit
        self.fill_bg = fill_bg
        
        # Read the labels
        labels_file = os.path.join(root_dir, f"{labels_filename}.csv")
        self.labels = np.loadtxt(labels_file, delimiter=",", skiprows=1)
        self.labels = self.labels.astype(int)
        if data_limit:
            self.labels = self.labels[:data_limit]

        self.num_classes = len(np.unique(self.labels[:, 1]))
        if class_names and self.num_classes != len(class_names):
            self.num_classes = len(class_names)
        self.class_names = class_names

        # Filter out indices of biopsies that exceed the size limit
        with open(os.path.join(DATA_DIR, "biopsy_dims.json"), 'r') as f:
            biopsy_dims = json.load(f)
        filtered_labels = []
        max_dim = 0
        for idx, label in self.labels:
            w_h = biopsy_dims[str(idx)]
            if size_limit and max(w_h) > size_limit:
                continue
            max_dim = max(max_dim, max(w_h))
            filtered_labels.append([idx, label])
        self.labels = np.array(filtered_labels)
        resize_factor = 1 / spacing

        self.size = int(size_limit * resize_factor)

        self.class_distribution = {i: np.sum(self.labels[:, 1] == i) for i in range(self.num_classes)}
        if class_names:
            self.class_distribution = {class_names[i]: self.class_distribution[i] for i in range(self.num_classes)}
        print("Class distribution: ", self.class_distribution)

        if latents_path:
            self.latents = torch.load(os.path.join(root_dir, latents_path)) # (n, 4, ftrs)
            self.latents = self.latents[self.labels[:, 0]] # Only keep the latents for the images in the dataset
            return

        # If self.imgs is already pickled, load it
        imgs_file = os.path.join(root_dir, f"imgs_{labels_filename}_sp{spacing}_sl{size_limit}.pt")
        if os.path.exists(imgs_file):
            self.imgs = torch.load(imgs_file)
            print("Loaded images from file")
        else:
            print("Processing images")
            # Read the images
            imgs_dir = os.path.join(root_dir, "biopsies")
            self.imgs = []
            for idx in tqdm(self.labels[:, 0]):
                img_file = os.path.join(imgs_dir, f"{idx}.png")
                img = plt.imread(img_file)
                img = torch.tensor(img).permute(2, 0, 1).float()
                dims = int(img.shape[1] * resize_factor), int(img.shape[2] * resize_factor)
                img = torch.nn.functional.interpolate(img.unsqueeze(0), size=dims).squeeze(0)
                self.imgs.append(img)
            
            # Save the images to a file
            torch.save(self.imgs, imgs_file)

        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = int(self.labels[idx, 1])

        if self.latents_path:
            return (self.latents[idx], label)

        image = self.imgs[idx]
        # Place image in self.size x self.size square without resizing
        h, w = image.shape[1], image.shape[2]
        frame = torch.zeros(3, self.size, self.size).fill_(self.fill_bg)
        # img_mean = image.mean(dim=(1, 2))
        # frame[:, :, :] = img_mean.view(3, 1, 1)
        # Center image in frame
        x_start = (self.size - w) // 2
        y_start = (self.size - h) // 2
        frame[:, y_start:y_start+h, x_start:x_start+w] = image
        image = frame
        
        if self.transform:
            image = self.transform(image)
        
        return (image, label)

    def plot_example_grid(self, n=5, random=True, figsize=(15, 15)):
        """Plot a grid with n*n examples."""
        if self.latents_path:
            print("Cannot plot examples from latents")
            return

        if random:
            indices = np.random.choice(len(self), size=n*n, replace=False)
        else:
            indices = np.arange(n*n)
        
        fig, axs = plt.subplots(n, n, figsize=figsize)
        for i, ax in enumerate(axs.flatten()):
            img = self[indices[i]][0].permute(1, 2, 0)
            ax.imshow(img)
            ax.set_title(self.class_names[self[indices[i]][1]] + f"\n{list(img.shape[:-1])}")
            ax.axis("off")
        plt.show()


