import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from torch import optim
from torch.optim import lr_scheduler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

import argparse
import os
from sklearn.model_selection import KFold

from load_data import BagDataset, get_balanced_dataloader, DATA_DIR, P53_CLASS_NAMES, \
    convert_presence_to_status, convert_status_to_presence, convert_presence_probs_to_status_probs
from clam_model.clam_utils import initialize_weights, calculate_error, Attn_Net, Attn_Net_Gated


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))



class CLAM_MB(pl.LightningModule):
    def __init__(self, gate = True, size_arg = "small", dropout = True, k_sample=8, num_classes=4,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, encoding_size=2048,
        lr=2e-4, weight_decay=1e-5, lr_step_size=30, lr_gamma=0.1,
        **kwargs
        ):
        super(CLAM_MB, self).__init__()
        self.size_dict = {"small": [encoding_size, 512, 256], "big": [encoding_size, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = num_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = num_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(num_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(num_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.num_classes = num_classes
        self.encoding_size = encoding_size
        self.subtyping = subtyping
        initialize_weights(self)

        self.loss_fn = nn.CrossEntropyLoss()
        self.bag_weight = 0.7
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        print("MODEL: CLAM_MB")
        print(f"MODEL ARGS: gate={gate}, size_arg={size_arg}, dropout={dropout}, k_sample={k_sample}, num_classes={num_classes}, subtyping={subtyping}" + \
              f"encoding_size={encoding_size}, lr={lr}, weight_decay={weight_decay}, lr_step_size={lr_step_size}, lr_gamma={lr_gamma}")

        self.outputs = {'train': [], 'val': [], 'test': []}

    def to(self, device):
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        # Ensure that k_sample is less than the number of instances
        k_sample = min(self.k_sample, A.shape[1])
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k_sample, device)
        n_targets = self.create_negative_targets(k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        # Ensure that k_sample is less than the number of instances
        k_sample = min(self.k_sample, A.shape[1])
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        h = h.squeeze(0).squeeze(1) # Nxftrs (squeeze out the batch and rotation dimensions)
        device = h.device
        A, h = self.attention_net(h)  # shape: NxK, Nxsize1, K=num_classes    
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.num_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) # shape: Kxsize1
        logits = torch.empty(1, self.num_classes).float().to(device)
        for c in range(self.num_classes):
            logits[0, c] = self.classifiers[c](M[c]) # shape 1
        Y_hat = torch.topk(logits, 1, dim = 1)[1] # shape 1
        Y_prob = F.softmax(logits, dim = 1) # shape K
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    def step(self, batch, batch_idx, mode='train'):
        data, label = batch
        logits, Y_prob, Y_hat, _, instance_dict = self(data, label=label, instance_eval=True)

        loss = self.loss_fn(logits, label)
        instance_loss = instance_dict['instance_loss']
        total_loss = self.bag_weight * loss + (1-self.bag_weight) * instance_loss

        self.outputs[mode].append({
            'loss': loss, 
            'labels': label,
            'preds': Y_hat.squeeze(0).cpu(),
            'probs': Y_prob.squeeze().cpu(),
            'instance_loss': instance_loss,
            'instance_preds': instance_dict['inst_preds'],
            'instance_labels': instance_dict['inst_labels'],
            'error': calculate_error(Y_hat, label)
        })
        return total_loss
    
    def on_epoch_end(self, mode):
        outputs = self.outputs[mode]
        self.outputs[mode] = []
        # Log loss and error
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        instance_loss = torch.stack([x['instance_loss'] for x in outputs]).mean()
        error = np.mean([x['error'] for x in outputs])
        self.log(f'loss/{mode}', loss)
        self.log(f'instance_loss/{mode}', instance_loss)
        self.log(f'error/{mode}', error)

        # Log accuracy
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        accuracy_per_class, _ = self.get_accuracy_per_class(preds, labels)
        avg_accuracy = accuracy_per_class.mean().item()
        self.log(f'accuracy/{mode}', avg_accuracy)

        # Log the confusion matrix into wandb
        if mode in ['val', 'test']:
            wandb.log({f"{mode}_confusion_matrix": wandb.plot.confusion_matrix(
                preds=preds.cpu().numpy(), 
                y_true=labels.cpu().numpy(),
                class_names=P53_CLASS_NAMES, title=f"{mode.capitalize()} confusion matrix")})

        # Log clustering accuracy
        inst_preds = torch.tensor(np.concatenate([x['instance_preds'] for x in outputs]))
        inst_labels = torch.tensor(np.concatenate([x['instance_labels'] for x in outputs]))
        inst_accuracy_per_class, _ = self.get_accuracy_per_class(inst_preds, inst_labels, n_classes=2)
        avg_inst_accuracy = inst_accuracy_per_class.mean().item()
        self.log(f'clustering_accuracy/{mode}', avg_inst_accuracy)

        # Log the confusion matrix into wandb
        if mode in ['val', 'test']:
            if len(inst_preds) > 0:
                wandb.log({f"{mode}_inst_confusion_matrix": wandb.plot.confusion_matrix(
                    preds=inst_preds.cpu().numpy(), 
                    y_true=inst_labels.cpu().numpy(),
                    class_names=["Normal", "Abnormal"], title=f"{mode.capitalize()} inst confusion matrix")})

        # Log AUC
        if self.num_classes == 2:
            probs = torch.stack([x['probs'] for x in outputs]).detach()
            labels = torch.cat([x['labels'] for x in outputs])
            auc = roc_auc_score(labels.cpu().numpy(), probs[:, 1].cpu().numpy())
        else:
            aucs = []
            binary_labels = label_binarize(labels.cpu().numpy(), classes=[i for i in range(self.num_classes)])
            probs = torch.stack([x['probs'] for x in outputs]).detach()
            for class_idx in range(self.num_classes):
                if class_idx in labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            auc = np.nanmean(np.array(aucs))
        self.log(f'auc/{mode}', auc)

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

    def get_accuracy_per_class(self, preds, labels, n_classes=4):
        confusion_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int64, device=self.device)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        accuracy_per_class = torch.diag(confusion_matrix) / confusion_matrix.sum(1)
        accuracy_per_class[torch.isnan(accuracy_per_class)] = 0
        return accuracy_per_class, confusion_matrix



class CLAM_db(CLAM_MB):
    def __init__(self, see_opposite_class_data="no_loss", size_arg="small", 
            gate=True, dropout=True,
            **kwargs):
        kwargs["num_classes"] = 4
        super(CLAM_db, self).__init__(**kwargs)
        self.db = 2

        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = self.db)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = self.db)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(self.db)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(self.db)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        initialize_weights(self)
        
        self.loss_fn = nn.BCELoss()
        self.see_opposite_class_data = see_opposite_class_data
        
        print("SUBMODEL: CLAM_db")
        print(f"SUBMODEL ARGS: see_opposite_class_data={see_opposite_class_data}")

    def step(self, batch, batch_idx, mode='train'):
        data, label = batch
        logits, Y_prob_db, Y_hat, _, instance_dict = self(data, label=label, instance_eval=True)
        Y_prob = convert_presence_probs_to_status_probs(Y_prob_db)

        db_label = convert_status_to_presence(label)
        loss = 0
        if label != 2 or self.see_opposite_class_data == "normal":
            loss += self.loss_fn(Y_prob_db[:, 0], db_label[:, 0])
        if label != 1 or self.see_opposite_class_data == "normal":
            loss += self.loss_fn(Y_prob_db[:, 1], db_label[:, 1])
        instance_loss = instance_dict['instance_loss']
        total_loss = self.bag_weight * loss + (1-self.bag_weight) * instance_loss

        self.outputs[mode].append({
            'loss': loss, 
            'labels': label,
            'preds': Y_hat.squeeze(0).cpu(),
            'probs': Y_prob.squeeze().cpu(),
            'instance_loss': instance_loss,
            'instance_preds': instance_dict['inst_preds'],
            'instance_labels': instance_dict['inst_labels'],
            'error': calculate_error(Y_hat, label)
        })
        return total_loss

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        h = h.squeeze(0).squeeze(1) # Nxftrs (squeeze out the batch and rotation dimensions)
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = convert_status_to_presence(label).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
            if total_inst_loss == 0:
                total_inst_loss = torch.tensor(0.0).to(device)

        M = torch.mm(A, h) # Matrix representing the attended features
        logits = torch.empty(1, self.db).float().to(device)
        for c in range(self.db):
            logits[0, c] = self.classifiers[c](M[c])
        Y_prob_db = F.sigmoid(logits) # Convert to probabilities
        Y_hat = convert_presence_to_status(Y_prob_db).unsqueeze(0)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob_db, Y_hat, A_raw, results_dict



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a model on the biopsies dataset.")
    # General arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients every n batches.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay.")
    parser.add_argument("--lr_step_size", type=int, default=30, help="Step size for the learning rate scheduler.")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Gamma for the learning rate scheduler.")

    # Specific arguments
    parser.add_argument("--model", type=str, default="CLAM_MB", help="Model to use.")
    # parser.add_argument("--model", type=str, default="CLAM_db", help="Model to use.")
    parser.add_argument("--encoding_size", type=int, default=512)
    parser.add_argument("--class_distr_factors", nargs="+", type=int, default=[1,1,1,0], help="Class distribution factors.")
    parser.add_argument("--see_opposite_class_data", type=str, default="no_loss", help="How to handle the opposite class data in the double binary model.")
    # parser.add_argument("--see_opposite_class_data", type=str, default="normal", help="How to handle the opposite class data in the double binary model.")
    # parser.add_argument("--latents_path", type=str, default="bag_latents_gs256_retccl__backup.pt", help="Path to the latents file.")
    parser.add_argument("--latents_path", type=str, default="bag_latents_gs64_resnet18_tuned.pt", help="Path to the latents file.")
    parser.add_argument("--subtyping", type=bool, default=False, help="Whether to use subtyping in the CLAM model.")
    parser.add_argument("--mix_bags", type=int, default=0, help="How many mixed bags in the CLAM model.")
    # parser.add_argument("--mix_bags", type=int, default=1e6, help="How many mixed bags in the CLAM model.")

    # Other arguments
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs to perform per fold.")
    parser.add_argument("--note", type=str, default="", help="Note to add to the run.")
    args = parser.parse_args()


    # Custom parsing
    args.class_distr_factors = np.array(args.class_distr_factors)
    if args.mix_bags:
        args.class_distr_factors = np.array([1, 1, 1, 1]) # Mix bags means all classes are represented

    weights_factor = args.class_distr_factors
    batch_size = args.batch_size

    if args.model == "CLAM_MB":
        Model = CLAM_MB
    elif args.model == "CLAM_db":
        Model = CLAM_db
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
    Dataset = BagDataset
    test_dataset = Dataset(root_dir=DATA_DIR, class_names=P53_CLASS_NAMES,
        labels_filename="test", latents_path=args.latents_path, grid_spacing=256,
        mix_bags=False, data_limit=None)
    print("Test dataset size: ", len(test_dataset))
    train_dataset = Dataset(root_dir=DATA_DIR, class_names=P53_CLASS_NAMES,
        labels_filename="train", latents_path=args.latents_path, grid_spacing=256,
        mix_bags=args.mix_bags, data_limit=None)
    print("Train dataset size: ", len(train_dataset))


    # Initialize k-fold cross-validation (same seed each time for reproducibility)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Create the test loader
    num_workers = 0
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    # Set id for the multi-fold run
    id_nr = np.random.randint(1e6)
    args.id_nr = id_nr
    
    model_str = args.model
    if "db" in args.model:
        model_str += f"_{args.see_opposite_class_data}"
    if args.subtyping:
        model_str += "_sub"
    if args.mix_bags:
        model_str += f"_m"
    if args.encoding_size != 2048:
        model_str += f"_enc{args.encoding_size}"
    name = f"e{args.epochs}_{model_str}"
    args.group_name = name

    project_name = "CLAM"

    # Loop over the folds
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        fold = fold + 1 # Start from 1
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        train_subset.num_classes = len(P53_CLASS_NAMES)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)
        val_subset.num_classes = len(P53_CLASS_NAMES)

        train_loader = get_balanced_dataloader(train_subset, batch_size=batch_size, weights_factor=weights_factor, num_workers=num_workers)
        if not args.mix_bags:
            val_loader = torch.utils.data.DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            val_loader = get_balanced_dataloader(val_subset, batch_size=batch_size, weights_factor=weights_factor, num_workers=num_workers)

        for run in range(args.num_runs):
            print(f"\nFOLD {fold} RUN {run+1}\n")
            # Add the fold number to the config
            args.fold = fold

            # Initialize wandb
            wandb.init(project=project_name, config=args, group=name)
            wandb_logger = pl.loggers.WandbLogger(project=project_name, config=args, group=name)
            # Give name to this specific run
            wandb_logger.experiment.name = name + f"_f{fold}"
            wandb_logger.experiment.save()

            # Initialize the model
            model = Model(num_classes=len(P53_CLASS_NAMES), lr=args.lr, weight_decay=args.weight_decay,
                                lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma, encoding_size=args.encoding_size,
                                see_opposite_class_data=args.see_opposite_class_data, subtyping=args.subtyping)
            model.to(device)

            # Define a checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                monitor='accuracy/val',
                dirpath='models/',
                filename='acc{accuracy/val:.2f}_epoch{epoch:02d}_'+f'{model_str}',
                auto_insert_metric_name=False,
                save_top_k=1,
                mode='max',
            )

            # Train the model
            trainer = pl.Trainer(max_epochs=args.epochs,
                                limit_train_batches=None if not args.mix_bags else 1178,
                                limit_val_batches=None if not args.mix_bags else 294,
                                logger=wandb_logger,
                                log_every_n_steps=5,
                                accumulate_grad_batches=args.accumulate_grad_batches,
                                callbacks=[checkpoint_callback],
                                accelerator="auto" if device.type == "cuda" else "cpu")
            trainer.fit(model, train_loader, val_loader)

            # Load the best model
            model = Model.load_from_checkpoint(checkpoint_callback.best_model_path, encoding_size=args.encoding_size,
                                                     num_classes=len(P53_CLASS_NAMES), subtyping=args.subtyping)

            # Test the model
            trainer.test(model, test_loader)
            wandb.finish()

