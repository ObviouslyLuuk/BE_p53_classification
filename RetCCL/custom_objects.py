"""Custom torchvision.models.resnet.* objects."""

from typing import Optional, Any

import torch
from torchvision.models.resnet import Weights, WeightsEnum, ResNet, BasicBlock
from torchvision.models.resnet import resnet18 as _resnet18
from torchvision.models.resnet import ResNet18_Weights
from torchvision import transforms

from . import _retccl as retccl


_all_ = (
  'HistoExtendedResnet18_Weights',
  'ResNet18_Weights',
  'resnet18',
  'retccl_resnet50',
  'HistoRetCCLResnet50_Weights'
)


class HistoExtendedResnet18_Weights(WeightsEnum):
  """Weights adapted from: https://github.com/ozanciga/self-supervised-histopathology.

  Original mpp varying from 0.25 to 0.5 and tile size 224.
  """
  SSHWeights = Weights(
    url='https://storage.googleapis.com/cold.s3.ellogon.ai/resnet18-histo-ssl.pth',
    transforms=transforms.Compose([
      transforms.PILToTensor(),
      transforms.ConvertImageDtype(torch.float),
      transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ]),
    meta={}
  )


def resnet18(*, weights: Optional[ResNet18_Weights | HistoExtendedResnet18_Weights] = None,
       progress: bool = True, **kwargs: Any) -> ResNet:
  if isinstance(weights, ResNet18_Weights) or weights is None:
    return _resnet18(weights=weights, progress=progress, *kwargs)

  model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
  model.load_state_dict(weights.get_state_dict(progress=progress))
  return model



class HistoRetCCLResnet50_Weights(WeightsEnum):
  """Weights adapted from: https://github.com/Xiyue-Wang/RetCCL.

  Original input size is 256 at 1mpp.
  """
  RetCCLWeights = Weights(
    url='https://storage.googleapis.com/cold.s3.ellogon.ai/resnet50-histo-retccl.pth',
    transforms=transforms.Compose([
      transforms.PILToTensor(),
      transforms.ConvertImageDtype(torch.float),
      transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ]),
    meta={}
  )


def retccl_resnet50(*, weights: Optional[HistoRetCCLResnet50_Weights] = None,
       progress: bool = True, **kwargs: Any) -> retccl.RetCCLResNet:
  model = retccl.RetCCLResNet(
    retccl.Bottleneck,
    [3, 4, 6, 3],
    num_classes=128,
    mlp=False,
    two_branch=False,
    normlinear=True,
    **kwargs
  )
  if weights is not None:
    model.load_state_dict(weights.get_state_dict(progress=progress))
  return model


