import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from load_data import DATA_DIR
BOLERO_DIR = os.path.join(DATA_DIR, "..", "BOLERO")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("Device: {}".format(device))


# Load RetCCL model (feature extractor)
from RetCCL.custom_objects import retccl_resnet50, HistoRetCCLResnet50_Weights

retccl = retccl_resnet50(weights=HistoRetCCLResnet50_Weights.RetCCLWeights).to(device)
# retccl is pretrained on images with size 256x256 at 1 micron per pixel
num_ftrs = retccl.fc.in_features
print("Number of features: {}".format(num_ftrs))
retccl.fc = nn.Identity()


# ImageNet normalization
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225]) # What this does is it subtracts the mean from each channel and divides by the std
])

TRANSFORM_NORMAL = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                      std=[0.5, 0.5, 0.5]) # What this does is it subtracts the mean from each channel and divides by the std
])

model_configs = {
    "retccl": {
        "model": retccl,
        "size": 256,
        "num_ftrs": 2048,
        "transform": TRANSFORM,
        "spacing": 1.0,
    },
}


def process_bag_latents(model_name="retccl", root_dir=DATA_DIR):
    print(f"Processing bag latents with {model_name} in {root_dir}...")
    model_config = model_configs[model_name]
    model = model_config["model"]
    model.eval()
    model = model.to(device)

    size = model_config["size"]
    resize_factor = 1 / model_config["spacing"]
    transform = model_config["transform"]
    num_ftrs = model_config["num_ftrs"]
    quantile_range_threshold = 0.1

    biopsy_ids = [f.split(".")[0] for f in os.listdir(os.path.join(root_dir, "biopsies"))]

    indices_filename = os.path.join(root_dir, "non_empty_patch_indices_gs256.pt")
    non_empty_patch_indices_by_biopsy = {}
    if os.path.exists(indices_filename):
        non_empty_patch_indices_by_biopsy = torch.load(indices_filename)

    # Load latents
    latent_file = os.path.join(root_dir, f"bag_latents_gs{size}_{model_name}.pt")
    bag_latents = {}
    if os.path.exists(latent_file):
        bag_latents = torch.load(latent_file)

    for i, idx in enumerate(tqdm(biopsy_ids)):
        if idx in bag_latents:
            continue
        # if idx in non_empty_patch_indices_by_biopsy:
        #     continue

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
        non_empty_idx = torch.where(qranges >= quantile_range_threshold)[0]
        patches = transform(patches[non_empty_idx])
        patches = patches.to(device)

        non_empty_patch_indices_by_biopsy[idx] = non_empty_idx

        with torch.no_grad():
            patch_latents = model(patches) # (n_patches, num_ftrs)
        bag_latents[idx] = patch_latents.cpu()[:, None, :] # (n_patches, 1, num_ftrs)

        if i % 10 == 0:
            torch.save(bag_latents, latent_file)
            torch.save(non_empty_patch_indices_by_biopsy, indices_filename)

        # Clear memory
        del img, patches, patch_latents
        torch.cuda.empty_cache()

    torch.save(bag_latents, latent_file)
    torch.save(non_empty_patch_indices_by_biopsy, indices_filename)


if __name__ == "__main__":
    process_bag_latents(model_name="retccl", root_dir=DATA_DIR)
    process_bag_latents(model_name="retccl", root_dir=BOLERO_DIR)

