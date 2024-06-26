import os
import torch
import torch.nn as nn
import torchvision

from RetCCL.custom_objects import retccl_resnet50, HistoRetCCLResnet50_Weights
from clam_model.model_clam import CLAM_SB

CLAM_DIR = os.path.join('..', 'CLAM')
CHECKPOINT_PATH = os.path.join(CLAM_DIR, 'results', 'test_s1', 's_0_checkpoint.pt')

TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225]), # What this does is it subtracts the mean from each channel and divides by the std
])

def load_encoder():
    # Load the pretrained encoder model
    retccl = retccl_resnet50(weights=HistoRetCCLResnet50_Weights.RetCCLWeights)
    # retccl is pretrained on images with size 256x256 at 1 micron per pixel
    num_ftrs = retccl.fc.in_features
    retccl.fc = nn.Identity()
    retccl.eval()
    return retccl, num_ftrs

def initiate_model(ckpt_path, dropout=True, n_classes=3, encoding_size=2048):    
    model = CLAM_SB(dropout=dropout,
                    n_classes=n_classes,
                    encoding_size=encoding_size)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    # model.relocate()
    model.eval()
    return model

def load_clam_model(device='cuda'):
    """Return encoder and clam model on device"""
    encoder, encoding_size = load_encoder()
    model = initiate_model(CHECKPOINT_PATH, encoding_size=encoding_size)
    encoder.to(device)
    model.attention_net = model.attention_net.to(device)
    model.classifiers = model.classifiers.to(device)
    model.instance_classifiers = model.instance_classifiers.to(device)
    return encoder, model

def process_image(img, patch_size=256, quantile_range_threshold=0.1, transform=TRANSFORM):
    size = patch_size

    # Resize the img so that the height and width are multiples of grid_spacing, 
    # rounded to the nearest multiple.
    # This prevents the last row and column of patches from being cut off.
    new_size = (max(round(img.shape[1]/size),1)*size, 
                max(round(img.shape[2]/size),1)*size)
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
    return patches, non_empty_idx

def clam_predict(encoder, model, img, patch_size=256):
    patches, _ = process_image(img, patch_size=patch_size)
    patches = patches.to(encoder.parameters().__next__().device)
    with torch.no_grad():
        # First process the patches into latents with the retccl model
        patch_latents = encoder(patches)
        # Then process the latents with the CLAM model
        # tuple of logits, Y_prob, Y_hat, A_raw, results_dict
        # shapes:  (1, 3), (1, 3), (1,1), (1, n_patches), dict
        logits, Y_prob, Y_hat, A_raw, results_dict = model(patch_latents)
    return Y_hat.item(), Y_prob.squeeze().cpu().numpy(), A_raw.squeeze().cpu().numpy()
