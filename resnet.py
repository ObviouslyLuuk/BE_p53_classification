import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from sklearn.model_selection import KFold

from RetCCL.custom_objects import retccl_resnet50, HistoRetCCLResnet50_Weights

from load_data import BiopsyDataset, BiopsyKeepScaleDataset, get_balanced_dataloader, \
    DATA_DIR, TRANSFORMS, P53_CLASS_NAMES, convert_status_to_presence, convert_presence_to_status

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))


class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes=4, latents_path=None, rot_inv=True,
                    lr=1e-3, weight_decay=0.0005, lr_step_size=30, lr_gamma=0.1, 
                    encoder_model="resnet18",
                    **kwargs):
        super(ResNetModel, self).__init__()
        if encoder_model == "resnet18":
            self.model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        elif encoder_model == "resnet50":
            self.model = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        elif encoder_model == "retccl":
            self.model = retccl_resnet50(weights=HistoRetCCLResnet50_Weights.RetCCLWeights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        if latents_path:
            # Freeze all layers except model.fc
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.latents_path = latents_path
        self.rotation_invariant = rot_inv
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        print("MODEL: ResNetModel")
        print(f"MODEL ARGS: num_classes={num_classes}, latents_path={latents_path}, rotation_invariant={rot_inv}, " +\
              f"lr={lr}, weight_decay={weight_decay}, lr_step_size={lr_step_size}, lr_gamma={lr_gamma}")

        self.outputs = {'train': [], 'val': [], 'test': []}

    def forward(self, x):
        if self.latents_path: # x is a batch of latents (B, 4, 512) where 4 is the number of rotations
            if self.rotation_invariant:
                x = torch.mean(x, dim=1) # Average over the rotations
            else:
                x = x[:, 0, :] # Take the first rotation
            return self.model.fc(x)
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def step(self, batch, batch_idx, mode):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        self.outputs[mode].append({'loss': loss, 'preds': preds, 'labels': labels})
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

        # Log the confusion matrix into wandb
        if mode in ['val', 'test']:
            wandb.log({f"{mode}_confusion_matrix": wandb.plot.confusion_matrix(
                preds=preds.cpu().numpy(), 
                y_true=labels.cpu().numpy(),
                class_names=P53_CLASS_NAMES, title=f"{mode.capitalize()} confusion matrix")})
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


# Make a model that inherits from the above ResNetModel
class ResNetModelDoubleBinary(ResNetModel):
    def __init__(self, see_opposite_class_data="normal", **kwargs):
        kwargs['num_classes'] = 4
        super(ResNetModelDoubleBinary, self).__init__(**kwargs)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2),
            nn.Sigmoid()
        )

        self.criterion = nn.BCELoss()
        self.see_opposite_class_data = see_opposite_class_data

        print("SUBMODEL: ResNetModelDoubleBinary")
        print(f"MODEL ARGS: see_opposite_class_data={see_opposite_class_data}")

    def step(self, batch, batch_idx, mode):
        images, labels = batch

        presence_labels = convert_status_to_presence(labels)

        outputs = self(images) # Shape: (batch_size, 2)

        # In simple terms, this will ignore the opposite class data
        # by setting the output equal to the label if the opposite class is present and the own class is not present
        # This is helpful to avoid the model to learn to predict the majority class (wildtype) because it can't be
        # balanced with doubleclone
        if self.see_opposite_class_data != "normal":
            presence_labels = presence_labels.bool()
            opposite_class_mask = torch.stack([
                presence_labels[:,0] |~ presence_labels[:,1], # Mask for overexpression: 1 if wildtype, overexpression or doubleclone, 0 if nullmutation
                presence_labels[:,1] |~ presence_labels[:,0], # Mask for nullmutation: 1 if wildtype, nullmutation or doubleclone, 0 if overexpression
            ], dim=1) # Shape: (batch_size, 2)
            opposite_class_mask = opposite_class_mask.float()
            presence_labels = presence_labels.float()
            if self.see_opposite_class_data == "no_loss":
                # For every 1 in the mask, set the corresponding output to the same value as the label
                outputs = outputs * opposite_class_mask + presence_labels * (1 - opposite_class_mask)
            elif self.see_opposite_class_data == "random_labels":
                # For every 1 in the mask, set the corresponding label to a random value between 0 and 1
                # This is to attempt to actively decorrelate the presence of the opposite class with the output
                presence_labels = presence_labels * opposite_class_mask + torch.rand_like(presence_labels) * (1 - opposite_class_mask)

        loss1 = self.criterion(outputs[:, 0], presence_labels[:, 0])
        loss2 = self.criterion(outputs[:, 1], presence_labels[:, 1])
        loss = loss1 + loss2

        preds = convert_presence_to_status(outputs) # Shape: (batch_size)
        self.outputs[mode].append({'loss': loss, 'preds': preds, 'labels': labels})
        return loss



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a model on the biopsies dataset.")
    # General arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients every n batches.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay.")
    parser.add_argument("--lr_step_size", type=int, default=30, help="Step size for the learning rate scheduler.")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Gamma for the learning rate scheduler.")

    # Specific arguments
    # parser.add_argument("--model", type=str, default="end-to-end", help="Model to use.")
    parser.add_argument("--model", type=str, default="end-to-end_double-binary", help="Model to use.")
    parser.add_argument("--class_distr_factors", nargs="+", type=int, default=[1,1,1,0], help="Class distribution factors.")
    parser.add_argument("--see_opposite_class_data", type=str, default="no_loss", help="How to handle the opposite class data in the double binary model.")
    parser.add_argument("--transform", type=str, default="basic_normalize", help="Transform to apply to the images.")
    # parser.add_argument("--transform", type=str, default="rotation_grey_normalize", help="Transform to apply to the images.")

    # Resizing arguments
    parser.add_argument("--img_size", type=int, default=256, help="Image size.")
    parser.add_argument("--spacing", type=int, default=16, help="Micron per pixel spacing of the images. Only used if keep_scale is True.")
    parser.add_argument("--keep_scale", type=bool, default=True, help="Whether to resize the images such that each image has the same scale or not.")
    parser.add_argument("--biopsy_size_limit", type=int, default=4096, help="Maximum size of the biopsy in pixels.")

    # Encoder arguments
    # parser.add_argument("--encoder_model", type=str, default="retccl", help="Encoder model to use.")
    parser.add_argument("--encoder_model", type=str, default="resnet18", help="Encoder model to use.")
    parser.add_argument("--latents", type=bool, default=False, help="Whether to use latents or not.")
    parser.add_argument("--rotation_invariant", type=bool, default=True, help="Whether to average the rotations or not. Only used if latents_path is given.")

    # Other arguments
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs to perform per fold.")
    parser.add_argument("--note", type=str, default="", help="Note to add to the run.")
    args = parser.parse_args()

    if args.latents:
        assert args.encoder_model in ["resnet18", "resnet50", "retccl"], "Encoder model must be one of 'resnet18', 'resnet50' or 'retccl'."
        args.latents_path = f"latents_s{args.img_size}_{args.encoder_model}.pt"
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
    args.fill_bg = 0 if "_black" in args.transform else 0.5

    if args.model == "end-to-end":
        Model = ResNetModel
    elif args.model == "end-to-end_double-binary":
        Model = ResNetModelDoubleBinary
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Check if the class distribution factors are balanced
    if args.model == "end-to-end_double-binary" and args.see_opposite_class_data == "normal" \
    and (args.class_distr_factors[0] + args.class_distr_factors[2]  \
      != args.class_distr_factors[1] + args.class_distr_factors[3]) \
     or (args.class_distr_factors[0] + args.class_distr_factors[1]  \
      != args.class_distr_factors[2] + args.class_distr_factors[3]):
        # The only way to balance this is to show as many wildtype and doubleclone as overexpression and nullmutation
        # Since we have no doubleclone, we can't balance this without setting wildtype to 0
        print("Warning: class distribution factors are not balanced for the double binary model.")
        print("This may lead the model to learn to predict the majority class.")
    elif args.model == "end-to-end_double-binary" and args.see_opposite_class_data != "normal" \
    and (args.class_distr_factors[0] != args.class_distr_factors[1] + args.class_distr_factors[3] \
      or args.class_distr_factors[0] != args.class_distr_factors[2] + args.class_distr_factors[3]):
        # The only way to balance this is to show as many wildtype as overexpression and nullmutation, and as many doubleclone as overexpression and nullmutation
        print("Warning: class distribution factors are not balanced for the double binary model.")
        print("Since the opposite class data is being handled differently, (n wildtype) should be balanced with (n mutated + doubleclone).")
        print("This may lead the model to learn to predict the majority class.")


    # Load the datasets
    Dataset = BiopsyDataset if not args.keep_scale else BiopsyKeepScaleDataset
    test_dataset = Dataset(root_dir=DATA_DIR, class_names=P53_CLASS_NAMES,
        size=args.img_size, labels_filename="test", transform=test_transform, latents_path=args.latents_path, 
        size_limit=args.biopsy_size_limit, fill_bg=args.fill_bg, spacing=args.spacing)
    print("Test dataset size: ", len(test_dataset))
    train_dataset = Dataset(root_dir=DATA_DIR, class_names=P53_CLASS_NAMES,
        size=args.img_size, labels_filename="train", transform=train_transform, latents_path=args.latents_path,
        size_limit=args.biopsy_size_limit, fill_bg=args.fill_bg, spacing=args.spacing,
        data_limit=None)
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
    if args.model == "end-to-end_double-binary":
        model_str += f"_{args.see_opposite_class_data}"
    if args.latents_path:
        model_str += "_latent"
    if "extra" in args.transform:
        model_str += "_aug"
    size_str = str(args.img_size) if not args.keep_scale else str(args.spacing)
    if args.keep_scale:
        model_str += "_ks"
    name = f"s{size_str}_e{args.epochs}_{model_str}"
    args.group_name = name


    # Set torch.set_float32_matmul_precision('medium' | 'high') on cluster to make use of Tensor Cores
    if args.img_size < 2048 or args.keep_scale:
        torch.set_float32_matmul_precision('high')
    else:
        torch.set_float32_matmul_precision('medium')

    # Loop over the folds
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        fold = fold + 1 # Start from 1
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        train_subset.num_classes = len(P53_CLASS_NAMES)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)

        train_loader = get_balanced_dataloader(train_subset, batch_size=batch_size, weights_factor=weights_factor, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # train_set_distribution = torch.zeros(len(P53_CLASS_NAMES))
        # for _, label in train_subset:
        #     train_set_distribution[label] += 1
        # print("distribution of biopsies in train set: ", train_set_distribution)
        # val_set_distribution = torch.zeros(len(P53_CLASS_NAMES))
        # for _, label in val_subset:
        #     val_set_distribution[label] += 1
        # print("distribution of biopsies in val set: ", val_set_distribution)

        # continue

        for run in range(args.num_runs):
            print(f"\nFOLD {fold} RUN {run+1}\n")
            # Add the fold number to the config
            args.fold = fold

            # Initialize wandb
            wandb.init(project='resnet_resize', config=args, group=name)
            wandb_logger = pl.loggers.WandbLogger(project='resnet_resize', config=args, group=name)
            # Give name to this specific run
            wandb_logger.experiment.name = name + f"_f{fold}"
            wandb_logger.experiment.save()

            # Initialize the model
            model = Model(num_classes=len(P53_CLASS_NAMES), lr=args.lr, weight_decay=args.weight_decay,
                                lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma,
                                see_opposite_class_data=args.see_opposite_class_data, 
                                latents_path=args.latents_path, rot_inv=args.rotation_invariant,
                                encoder_model=args.encoder_model)
            model.to(device)

            # Define a checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                monitor='accuracy/val',
                dirpath='models/',
                filename='acc{accuracy/val:.2f}_epoch{epoch:02d}_'+name,
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
                                                     encoder_model=args.encoder_model,
                                                     num_classes=len(P53_CLASS_NAMES), 
                                                     latents_path=args.latents_path, rot_inv=args.rotation_invariant)

            # Test the model
            trainer.test(model, test_loader)
            wandb.finish()
