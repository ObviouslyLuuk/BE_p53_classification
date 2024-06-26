import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle as pkl
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from sklearn.model_selection import KFold

from RetCCL.custom_objects import retccl_resnet50, HistoRetCCLResnet50_Weights

from load_data import BagDataset, get_balanced_dataloader, DATA_DIR, TRANSFORMS, P53_CLASS_NAMES

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print("Device: {}".format(device))


def patch_presence_2_status(patch_labels, bag_label):
    """
    Convert patch presence labels (B, N, 2) to patch status labels (B, N)
    where 0 = wildtype, 1 = overexpression, 2 = nullmutation, 3 = doubleclone
    status is np.nan if unknown
    """
    # If bag label is wildtype, all patches are wildtype
    if bag_label == 0:
        return torch.zeros_like(patch_labels[:,:,0])
    # If bag label is overexpression, all patches are unknown
    if bag_label == 1:
        return torch.full_like(patch_labels[:,:,0], np.nan)
    # If bag label is nullmutation, all patches are known based on the presence labels
    # If bag label is doubleclone, patches are unknown if no nullmutation present
    if bag_label in [2,3]:
        status = (patch_labels[:,:,1] > 0).float()
        status *= 2 # 0 or 2 because 0 = wildtype, 2 = nullmutation
        if bag_label == 3:
            status[status == 0] = np.nan
        return status


class ResNetModelMIL(pl.LightningModule):
    def __init__(self, num_classes=4, latents_path=None, rot_inv=True,
                    mil_aggregator='max', patch_loss_weight=0.5,
                    lr=1e-3, weight_decay=0.0005, lr_step_size=30, lr_gamma=0.1, 
                    encoder_model="resnet18",
                    **kwargs):
        super(ResNetModelMIL, self).__init__()
        if encoder_model == "resnet18":
            self.model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        elif encoder_model == "resnet50":
            self.model = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        elif encoder_model == "retccl":
            self.model = retccl_resnet50(weights=HistoRetCCLResnet50_Weights.RetCCLWeights)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc = nn.Linear(num_ftrs, num_classes)

        if latents_path:
            # Freeze all layers except model.fc
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

        self.num_classes = num_classes
        self.mil_aggregator = mil_aggregator
        self.patch_loss_weight = patch_loss_weight

        self.criterion = nn.CrossEntropyLoss()
        self.latents_path = latents_path
        self.rotation_invariant = rot_inv
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        print("MODEL: ResNetModel (MIL)")
        print(f"MODEL ARGS: num_classes={num_classes}, latents_path={latents_path}, rotation_invariant={rot_inv}, " +\
              f"mil_aggregator={mil_aggregator}, patch_loss_weight={patch_loss_weight}, " + \
              f"lr={lr}, weight_decay={weight_decay}, lr_step_size={lr_step_size}, lr_gamma={lr_gamma}")

        self.outputs = {'train': [], 'val': [], 'test': []}
        self.class_proportions = {i: 0.0 for i in range(self.num_classes)}
        self.batch_count = 0
        
    def update_class_proportions(self, y):
        unique, counts = np.unique(y, return_counts=True)
        batch_proportions = {i: 0.0 for i in range(self.num_classes)}
        for u, c in zip(unique, counts):
            batch_proportions[u] = c / len(y)
        for i in range(self.num_classes):
            self.class_proportions[i] = (self.batch_count * self.class_proportions[i] + batch_proportions[i]) / (self.batch_count + 1)
        self.batch_count += 1

    def get_class_weights(self):
        class_weights = [1.0 / self.class_proportions[i] if self.class_proportions[i] > 0 else 0 for i in range(self.num_classes)]
        # Normalize the class weights
        class_weights = [w / sum(class_weights) for w in class_weights]
        return class_weights

    def forward(self, x):
        B = x.shape[0]
        assert B == 1, "Batch size must be 1, more bags not supported yet"

        if self.latents_path:
            # x is a bag of patch_latents (B, N, 4, num_ftrs) with B=1 and 4 rotations
            if self.rotation_invariant:
                x = torch.mean(x, dim=2) # (B, N, num_ftrs) (average over rotations)
            else:
                x = x[:,:,0] # (B, N, num_ftrs) (take only one rotation)
            patch_latents = x.view(-1, x.shape[-1]) # (N, num_ftrs)
        else:
            # x is a bag of patches (B, N, C, H, W) with B=1
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4]) # (N, C, H, W)
            patch_latents = self.model(x) # (N, num_ftrs)
        patch_preds = self.fc(patch_latents) # (N, num_classes)

        if self.mil_aggregator == 'avg':
            pred = patch_preds.mean(dim=0, keepdim=True) # (1, num_classes)
        elif self.mil_aggregator == 'max':
            # Get the maximum overall prediction index (regardless of class)
            m = patch_preds.view(1, -1).argmax(dim=1) # (1) with m < N*num_classes
            # assert m < patch_preds.shape[0]*patch_preds.shape[1], f"Max index out of bounds: {m} >= {patch_preds.shape[0]}*{patch_preds.shape[1]}"

            # Get the corresponding patch prediction
            max_patch_idx = m // self.num_classes # (1) with max_patch_idx < N
            # assert max_patch_idx < patch_preds.shape[0], f"Max patch index out of bounds: {m}//{self.num_classes}={max_patch_idx} >= {patch_preds.shape[0]}"

            pred = patch_preds[max_patch_idx] # (1, num_classes)
        else:
            raise ValueError(f"Unknown MIL aggregator: {self.mil_aggregator}")

        return pred, patch_preds

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def step(self, batch, batch_idx, mode):
        images, labels, patch_labels = batch # (B, N, C, H, W), (B), (B, N, 2) with B=1
        patch_labels = patch_presence_2_status(patch_labels, labels[0]) # (B, N)
        outputs, patch_outputs = self(images) # (B, num_classes), (N, num_classes)

        # assert outputs.shape[0] == 1, f"Output shape is {outputs.shape}, expected (1, num_classes)"
        # assert labels.shape[0] == 1, f"Labels shape is {labels.shape}, expected (1)"
        # assert labels[0] < self.num_classes, f"Label is {labels[0]}, expected < {self.num_classes}"

        loss = self.criterion(outputs, labels)

        # Patch loss
        patch_labels = patch_labels.view(-1) # (N)
        not_nan_indices = torch.nonzero(~torch.isnan(patch_labels)).view(-1)
        patch_labels = patch_labels[not_nan_indices]
        patch_outputs = patch_outputs[not_nan_indices]
        patch_loss = torch.zeros(1, device=device)
        if len(not_nan_indices) > 0 and self.patch_loss_weight > 0:
            self.update_class_proportions(patch_labels.cpu().numpy())
            class_weights = torch.tensor(self.get_class_weights(), device=device)
            for i in range(self.num_classes):
                class_mask = patch_labels == i
                if class_mask.sum() == 0:
                    continue

                class_loss = self.criterion(patch_outputs[class_mask].float(), patch_labels[class_mask].long())
                patch_loss += class_weights[i] * class_loss

        # Combine the losses
        loss = (1 - self.patch_loss_weight) * loss + self.patch_loss_weight * patch_loss

        # Log the outputs
        preds = outputs.argmax(dim=-1) # (1)
        patch_preds = patch_outputs.argmax(dim=-1) # (N)
        self.outputs[mode].append({'loss': loss, 'preds': preds, 'labels': labels,
            'patch_loss': patch_loss, 'patch_preds': patch_preds, 'patch_labels': patch_labels})
        return loss
    
    def on_epoch_end(self, mode):
        outputs = self.outputs[mode]
        self.outputs[mode] = []
        # Log loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log(f'loss/{mode}', avg_loss)
        # Log patch loss
        avg_patch_loss = torch.stack([x['patch_loss'] for x in outputs]).mean()
        self.log(f'patch_loss/{mode}', avg_patch_loss)

        # Log accuracy
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        assert len(preds) == len(labels), f"Preds and labels have different lengths: {len(preds)} != {len(labels)}\n" + \
            f"Preds: {[x['preds'].shape for x in outputs]}\nLabels: {[x['labels'].shape for x in outputs]}"
        accuracy_per_class, _ = self.get_accuracy_per_class(preds, labels)
        avg_accuracy = torch.mean(accuracy_per_class)
        self.log(f'accuracy/{mode}', avg_accuracy)
        # Log patch accuracy
        patch_preds = torch.cat([x['patch_preds'] for x in outputs])
        patch_labels = torch.cat([x['patch_labels'] for x in outputs])
        assert len(patch_preds) == len(patch_labels), f"Patch preds and labels have different lengths: {len(patch_preds)} != {len(patch_labels)}"
        patch_accuracy_per_class, _ = self.get_accuracy_per_class(patch_preds, patch_labels)
        avg_patch_accuracy = torch.mean(patch_accuracy_per_class)
        self.log(f'patch_accuracy/{mode}', avg_patch_accuracy)

        # Log confusion matrix
        wandb.log({f"{mode}_confusion_matrix": wandb.plot.confusion_matrix(
            preds=preds.cpu().numpy(), 
            y_true=labels.cpu().numpy(),
            class_names=P53_CLASS_NAMES, title=f"{mode.capitalize()} confusion matrix")})
        # Log patch confusion matrix
        wandb.log({f"{mode}_patch_confusion_matrix": wandb.plot.confusion_matrix(
            preds=patch_preds.cpu().numpy(), 
            y_true=patch_labels.cpu().numpy(),
            class_names=P53_CLASS_NAMES, title=f"{mode.capitalize()} patch confusion matrix")})
        return {f'accuracy/{mode}': avg_accuracy}

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    
    def on_train_epoch_end(self):
        return self.on_epoch_end('train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def on_validation_epoch_end(self):
        return self.on_epoch_end('val')
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')
    
    def on_test_epoch_end(self):
        return self.on_epoch_end('test')

    def get_accuracy_per_class(self, preds, labels):
        confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=self.device)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        accuracy_per_class = torch.diag(confusion_matrix) / confusion_matrix.sum(1)
        accuracy_per_class[torch.isnan(accuracy_per_class)] = 0
        return accuracy_per_class, confusion_matrix



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



class ResNetModelMILDoubleBinary(ResNetModelMIL):
    def __init__(self, **kwargs):
        kwargs['num_classes'] = 4
        super(ResNetModelMILDoubleBinary, self).__init__(**kwargs)
        num_ftrs = self.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2),
            nn.Sigmoid()
        )

        self.criterion = nn.BCELoss()
        # No need for see_opposite_class_data, since that only applies when two classes are present
        # in the same image (because it learns mutual exclusivity), which is not the case for the patch model

        print("SUBMODEL: ResNetModelMILDoubleBinary")

    def forward(self, x):
        B = x.shape[0]
        assert B == 1, "Batch size must be 1, more bags not supported yet"

        if self.latents_path:
            # x is a bag of patch_latents (B, N, 4, num_ftrs) with B=1 and 4 rotations
            if self.rotation_invariant:
                x = torch.mean(x, dim=2) # (B, N, num_ftrs) (average over rotations)
            else:
                x = x[:,:,0] # (B, N, num_ftrs) (take only one rotation)
            patch_latents = x.view(-1, x.shape[-1]) # (N, num_ftrs)
        else:
            # x is a bag of patches (B, N, C, H, W) with B=1
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4]) # (N, C, H, W)
            patch_latents = self.model(x) # (N, num_ftrs)
        patch_preds = self.fc(patch_latents) # (N, 2)

        if self.mil_aggregator == 'avg':
            pred = patch_preds.mean(dim=0, keepdim=True) # (1, 2)
        elif self.mil_aggregator == 'max':
            pred = patch_preds.max(dim=0, keepdim=True)[0] # (1, 2) (max returns both values and indices, we only want the values)
        else:
            raise ValueError(f"Unknown MIL aggregator: {self.mil_aggregator}")

        return pred, patch_preds
        
    def step(self, batch, batch_idx, mode):
        images, labels, patch_labels = batch # (B, N, C, H, W), (B), (B, N, 2) with B=1
        presence_patch_labels = patch_labels.view(-1, 2) # (N, 2)
        patch_labels = patch_presence_2_status(patch_labels, labels[0]) # (B, N)
        outputs, patch_outputs = self(images) # (B, 2), (N, 2)

        presence_labels = convert_status_to_presence(labels) # (B, 2)

        loss1 = self.criterion(outputs[:, 0], presence_labels[:, 0])
        loss2 = self.criterion(outputs[:, 1], presence_labels[:, 1])
        loss = loss1 + loss2

        # Patch loss

        # Where presence is > 0 and not nan, move the range from [0, 1] to [.5, 1]
        # This is because we want any presence to be considered as a positive label
        presence_patch_labels[presence_patch_labels > 0] = 0.5 + presence_patch_labels[presence_patch_labels > 0] * 0.5

        patch_labels = patch_labels.view(-1) # (N)
        not_nan_indices = torch.nonzero(~torch.isnan(patch_labels)).view(-1)
        presence_patch_labels = presence_patch_labels[not_nan_indices]
        patch_labels = patch_labels[not_nan_indices]
        patch_outputs = patch_outputs[not_nan_indices]
        patch_loss = torch.zeros(1, device=device)
        if len(not_nan_indices) > 0 and self.patch_loss_weight > 0:
            # If nan in presence_patch_labels, print the count
            nan_count = torch.isnan(presence_patch_labels).sum()
            if nan_count > 0:
                print(f"Label: {labels[0]}")
                print(f"NaN count: {nan_count}")
                print(f"Presence patch labels: {presence_patch_labels}")
                print(f"Patch labels: {patch_labels}")
            assert presence_patch_labels.min() >= 0 and presence_patch_labels.max() <= 1, f"Presence patch labels out of range: {presence_patch_labels.min()} - {presence_patch_labels.max()}"
            
            self.update_class_proportions(patch_labels.cpu().numpy())
            class_weights = torch.tensor(self.get_class_weights(), device=device)
            # print("Class weights: ", class_weights)
            for i in range(self.num_classes):
                class_mask = patch_labels == i
                if class_mask.sum() == 0:
                    continue

                # print(patch_outputs[class_mask], presence_patch_labels[class_mask]) # (class_n, 2), (class_n, 2)

                class_loss = self.criterion(patch_outputs[class_mask].float(), presence_patch_labels[class_mask].float())
                patch_loss += class_weights[i] * class_loss

        # Combine the losses
        loss = (1 - self.patch_loss_weight) * loss + self.patch_loss_weight * patch_loss

        # Log the outputs
        preds = convert_presence_to_status(outputs) # Shape: (B)
        patch_preds = convert_presence_to_status(patch_outputs) # Shape: (N)
        self.outputs[mode].append({'loss': loss, 'preds': preds, 'labels': labels,
            'patch_loss': patch_loss, 'patch_preds': patch_preds, 'patch_labels': patch_labels})
        return loss



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a model on the biopsies dataset.")
    # General arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=8, help="Accumulate gradients every n batches.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay.")
    parser.add_argument("--lr_step_size", type=int, default=30, help="Step size for the learning rate scheduler.")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Gamma for the learning rate scheduler.")

    # Specific arguments
    parser.add_argument("--model", type=str, default="mil_double-binary", help="Model to use.")
    parser.add_argument("--class_distr_factors", nargs="+", type=int, default=[1,1,1,0], help="Class distribution factors.")
    parser.add_argument("--img_size", type=int, default=64, help="Image size.")
    parser.add_argument("--transform", type=str, default="basic_normalize", help="Transform to apply to the images.")

    # Encoder arguments
    parser.add_argument("--encoder_model", type=str, default="retccl", help="Encoder model to use.")
    parser.add_argument("--latents", type=bool, default=True, help="Whether to use latents or not.")
    parser.add_argument("--rotation_invariant", type=bool, default=False, help="Whether to average the rotations or not. Only used if latents_path is given.")

    # Patching arguments
    parser.add_argument("--mil_aggregator", type=str, default="max", help="Aggregator for the MIL model.")
    parser.add_argument("--grid_spacing", type=int, default=256, help="Grid spacing for the MIL model.")
    parser.add_argument("--patch_loss_weight", type=float, default=0, help="Weight for the patch loss.")

    # Other arguments
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs to perform per fold.")
    parser.add_argument("--note", type=str, default="", help="Note to add to the run.")
    args = parser.parse_args()

    if args.latents:
        assert args.encoder_model in ["resnet18", "resnet50", "retccl"], "Encoder model must be one of 'resnet18', 'resnet50' or 'retccl'."
        args.latents_path = f"bag_latents_gs{args.grid_spacing}_{args.encoder_model}.pt"
        args.img_size = args.grid_spacing
        assert os.path.exists(os.path.join(DATA_DIR, args.latents_path)), f"Latents file {args.latents_path} does not exist."
    else:
        args.latents_path = None

    # Custom parsing
    args.class_distr_factors = np.array(args.class_distr_factors)
    weights_factor = args.class_distr_factors
    batch_size = args.batch_size
    train_transform = TRANSFORMS[args.transform]
    test_transform = None
    if "normalize" in args.transform:
        test_transform = TRANSFORMS["normalize"]

    if args.model == "mil":
        Model = ResNetModelMIL
    elif args.model == "mil_double-binary":
        Model = ResNetModelMILDoubleBinary
    else:
        raise ValueError(f"Unknown model: {args.model}")


    # Check if the class distribution factors are balanced
    if args.model == "mil_double-binary" \
    and (args.class_distr_factors[0] + args.class_distr_factors[2]  \
      != args.class_distr_factors[1] + args.class_distr_factors[3] \
     or args.class_distr_factors[0] + args.class_distr_factors[1]  \
      != args.class_distr_factors[2] + args.class_distr_factors[3]):
        # The only way to balance this is to show as many wildtype and doubleclone as overexpression and nullmutation
        # Since we have no doubleclone, we can't balance this without setting wildtype to 0
        print("Warning: class distribution factors are not balanced for the double binary model.")
        print("This may lead the model to learn to predict the majority class.")
        print("This might not be a problem however, because in the end the bag labels come from the patch labels.")


    # Load the datasets
    test_dataset = BagDataset(root_dir=DATA_DIR, class_names=P53_CLASS_NAMES,
        size=args.img_size, labels_filename="test", transform=test_transform,
        grid_spacing=args.grid_spacing, latents_path=args.latents_path)
    print("Test dataset size: ", len(test_dataset))
    train_dataset = BagDataset(root_dir=DATA_DIR, class_names=P53_CLASS_NAMES,
        size=args.img_size, labels_filename="train", transform=train_transform,
        grid_spacing=args.grid_spacing, data_limit=None, latents_path=args.latents_path)
    print("Train dataset size: ", len(train_dataset))


    # Initialize k-fold cross-validation (same seed each time for reproducibility)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Create the test loader
    num_workers = 0
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    # Set id for the multi-fold run
    id_nr = np.random.randint(1e6)
    args.id_nr = id_nr
    
    # norm_str = "norm" if "normalize" in args.transform else "nonorm"
    # name = f"s{args.img_size}_{norm_str}_e{args.epochs}" # We're defaulting to normalizing now
    model_str = args.model
    if args.latents:
        model_str += f"_latent"
    name = f"s{args.img_size}-{args.grid_spacing}_e{args.epochs}_p{args.patch_loss_weight}_{model_str}"
    args.group_name = name


    # Set torch.set_float32_matmul_precision('medium' | 'high') on cluster to make use of Tensor Cores
    torch.set_float32_matmul_precision('high')

    # Loop over the folds
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        fold = fold + 1 # Start from 1
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        train_subset.num_classes = len(P53_CLASS_NAMES)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)

        train_loader = get_balanced_dataloader(train_subset, batch_size=batch_size, weights_factor=weights_factor, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for run in range(args.num_runs):
            print(f"\nFOLD {fold} RUN {run+1}\n")
            # Add the fold number to the config
            args.fold = fold

            # Initialize wandb
            wandb.init(project='resnet_mil', config=args, group=name)
            wandb_logger = pl.loggers.WandbLogger(project='resnet_mil', config=args, group=name)
            # Give name to this specific run
            wandb_logger.experiment.name = name + f"_f{fold}"
            wandb_logger.experiment.save()

            # Initialize the model
            model = Model(num_classes=len(P53_CLASS_NAMES), lr=args.lr, weight_decay=args.weight_decay,
                                lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma,
                                mil_aggregator=args.mil_aggregator, patch_loss_weight=args.patch_loss_weight,
                                latents_path=args.latents_path, rot_inv=args.rotation_invariant,
                                encoder_model=args.encoder_model)
            model.to(device)

            # Define a checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                monitor='accuracy/val',
                dirpath='models/',
                filename='acc{accuracy/val:.2f}_epoch{epoch:02d}_'+f's{args.img_size}_{args.model}',
                auto_insert_metric_name=False,
                save_top_k=1,
                mode='max',
            )

            # Train the model
            trainer = pl.Trainer(max_epochs=args.epochs, 
                                logger=wandb_logger,
                                log_every_n_steps=5,
                                accumulate_grad_batches=args.accumulate_grad_batches,
                                callbacks=[checkpoint_callback],
                                accelerator="auto" if device.type == "cuda" else "cpu")
            trainer.fit(model, train_loader, val_loader)

            # Load the best model
            model = Model.load_from_checkpoint(checkpoint_callback.best_model_path,
                mil_aggregator=args.mil_aggregator, patch_loss_weight=args.patch_loss_weight,
                num_classes=len(P53_CLASS_NAMES), latent_path=args.latents_path, rot_inv=args.rotation_invariant,
                encoder_model=args.encoder_model)

            # Test the model
            trainer.test(model, test_loader)
            wandb.finish()
