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

from load_data import Nullmutation, get_balanced_dataloader, DATA_DIR, TRANSFORMS, P53_CLASS_NAMES

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))


class ResNetModelNM(pl.LightningModule):
    def __init__(self,
                    lr=1e-3, weight_decay=0.0005, lr_step_size=30, lr_gamma=0.1, **kwargs):
        super(ResNetModelNM, self).__init__()
        self.model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

        self.num_classes = 2

        self.criterion = nn.BCELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        print("MODEL: ResNetModel (nullmutation)")
        print(f"MODEL ARGS: lr={lr}, weight_decay={weight_decay}, lr_step_size={lr_step_size}, lr_gamma={lr_gamma}")

        self.outputs = {'train': [], 'val': [], 'test': []}

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def step(self, batch, batch_idx, mode):
        images, labels = batch # (32, 3, size, size), (32, 1, size, size) or (32, 1)
        # labels = labels.mean(dim=(-2,-1)) # (32, 1, size, size) -> (32, 1) This is done in the dataset now

        # Make nonzero labels range from 0.5 to 1.0 instead of 0.0 to 1.0
        # For classification thresholding purposes
        labels[labels > 0] = labels[labels > 0] * 0.5 + 0.5

        outputs = self(images) # (32, 1)
        loss = self.criterion(outputs, labels)
        preds = (outputs > 0.5).float()
        self.outputs[mode].append({'loss': loss, 'preds': preds, 'labels': (labels > 0.5).float()})
        return loss
    
    def on_epoch_end(self, mode):
        outputs = self.outputs[mode]
        self.outputs[mode] = []
        # Log loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log(f'loss/{mode}', avg_loss)

        # Log accuracy
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        accuracy_per_class, _ = self.get_accuracy_per_class(preds, labels)
        avg_accuracy = torch.mean(accuracy_per_class)
        self.log(f'accuracy/{mode}', avg_accuracy)
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



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a model on the biopsies dataset.")
    # General arguments
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients every n batches.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay.")
    parser.add_argument("--lr_step_size", type=int, default=30, help="Step size for the learning rate scheduler.")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Gamma for the learning rate scheduler.")

    # Specific arguments
    parser.add_argument("--model", type=str, default="nm", help="Model to use.")
    parser.add_argument("--img_size", type=int, default=64, help="Image size.")
    parser.add_argument("--transform", type=str, default="basic_normalize", help="Transform to apply to the images.")

    # Patching arguments
    parser.add_argument("--grid_spacing", type=int, default=256, help="Grid spacing for the MIL model.")

    # Other arguments
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs to perform per fold.")
    parser.add_argument("--note", type=str, default="", help="Note to add to the run.")
    args = parser.parse_args()

    # Custom parsing
    batch_size = args.batch_size
    train_transform = TRANSFORMS[args.transform]
    test_transform = None
    if "normalize" in args.transform:
        test_transform = TRANSFORMS["normalize"]

    if args.model == "nm":
        Model = ResNetModelNM
    else:
        raise ValueError(f"Unknown model: {args.model}")


    # Load the datasets
    test_dataset = Nullmutation(root_dir=DATA_DIR, grid_spacing=args.grid_spacing,
        size=args.img_size, labels_filename="test", transform=test_transform,
        return_mask_label=False)
    print("Test dataset size: ", len(test_dataset))
    train_dataset = Nullmutation(root_dir=DATA_DIR, grid_spacing=args.grid_spacing,
        size=args.img_size, labels_filename="train", transform=train_transform,
        data_limit=None, return_mask_label=False)
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
    name = f"s{args.img_size}_e{args.epochs}_{model_str}"
    args.group_name = name


    # Set torch.set_float32_matmul_precision('medium' | 'high') on cluster to make use of Tensor Cores
    torch.set_float32_matmul_precision('high')

    # Loop over the folds
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        fold = fold + 1 # Start from 1
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        train_subset.num_classes = 2
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)

        train_loader = get_balanced_dataloader(train_subset, batch_size=batch_size, weights_factor=None, num_workers=num_workers, masks=True)
        # train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for run in range(args.num_runs):
            print(f"\nFOLD {fold} RUN {run+1}\n")
            # Add the fold number to the config
            args.fold = fold

            # Initialize wandb
            wandb.init(project='resnet_nm', config=args, group=name)
            wandb_logger = pl.loggers.WandbLogger(project='resnet_nm', config=args, group=name)
            # Give name to this specific run
            wandb_logger.experiment.name = name + f"_f{fold}"
            wandb_logger.experiment.save()

            # Initialize the model
            model = Model(num_classes=2, lr=args.lr, weight_decay=args.weight_decay,
                                lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma)
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
                                callbacks=[checkpoint_callback])
            trainer.fit(model, train_loader, val_loader)

            # Load the best model
            model = Model.load_from_checkpoint(checkpoint_callback.best_model_path,
                                                     num_classes=len(P53_CLASS_NAMES))

            # Test the model
            trainer.test(model, test_loader)
            wandb.finish()
