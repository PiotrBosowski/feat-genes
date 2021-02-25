import torch
import torchvision
import settings


# 1   ResNet34Spawner
# 2   ResNet50Spawner
# 3   ResNet101Spawner
# 4   ResNet152Spawner
# 5   WideResNet50Spawner
# 6   WideResNet101Spawner
# 7   AlexNetSpawner
# 8   SqueezeNet1_1Spawner
# 9   DenseNet121Spawner
# 10  DenseNet161Spawner
# 11  DenseNet169Spawner
# 12  DenseNet201Spawner
# 13  GoogleNetSpawner
# 14  Inception_v3Spawner
# 15  MnasNet1_0Spawner
# 16  MobileNet_v2Spawner
# 17  ShuffleNet_v2_x1_0Spawner
# 18  VGG11Spawner
# 19  VGG11BNSpawner
# 20  VGG13Spawner
# 21  VGG13BNSpawner
# 22  VGG16Spawner
# 23  VGG16BNSpawner
# 24  VGG19Spawner
# 25  VGG19BNSpawner


class ResNet50Spawner:
    def __init__(self):
        self.name = "ResNet-50"

    def __call__(self):
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048,
                                   out_features=len(settings.all_labels))
        return model


class ResNet101Spawner:
    def __init__(self):
        self.name = "ResNet-101"

    def __call__(self):
        model = torchvision.models.resnet101(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048,
                                   out_features=len(settings.all_labels))
        return model


class ResNet152Spawner:
    def __init__(self):
        self.name = "ResNet-152"

    def __call__(self):
        model = torchvision.models.resnet152(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048,
                                   out_features=len(settings.all_labels))
        return model


class WideResNet50Spawner:
    def __init__(self):
        self.name = "Wide_ResNet-50"

    def __call__(self):
        model = torchvision.models.wide_resnet50_2(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048,
                                   out_features=len(settings.all_labels))
        return model


class WideResNet101Spawner:
    def __init__(self):
        self.name = "Wide_ResNet-101"

    def __call__(self):
        model = torchvision.models.wide_resnet101_2(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048,
                                   out_features=len(settings.all_labels))
        return model


class AlexNetSpawner:
    def __init__(self):
        self.name = "AlexNet"

    def __call__(self):
        model = torchvision.models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096,
                                              out_features=len(
                                                  settings.all_labels))
        return model


class DenseNet121Spawner:
    def __init__(self):
        self.name = "DenseNet121"

    def __call__(self):
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(in_features=1024,
                                           out_features=len(
                                               settings.all_labels))
        return model


class DenseNet161Spawner:
    def __init__(self):
        self.name = "DenseNet161"

    def __call__(self):
        model = torchvision.models.densenet161(pretrained=True)
        model.classifier = torch.nn.Linear(in_features=2208,
                                           out_features=len(
                                               settings.all_labels))
        return model


class DenseNet169Spawner:
    def __init__(self):
        self.name = "DenseNet169"

    def __call__(self):
        model = torchvision.models.densenet169(pretrained=True)
        model.classifier = torch.nn.Linear(in_features=1664,
                                           out_features=len(
                                               settings.all_labels))
        return model


class DenseNet201Spawner:
    def __init__(self):
        self.name = "DenseNet201"

    def __call__(self):
        model = torchvision.models.densenet201(pretrained=True)
        model.classifier = torch.nn.Linear(in_features=1920,
                                           out_features=len(
                                               settings.all_labels))
        return model


class GoogleNetSpawner:
    def __init__(self):
        self.name = "GoogleNet"

    def __call__(self):
        model = torchvision.models.googlenet(pretrained=True)
        model.fc = torch.nn.Linear(in_features=1024,
                                   out_features=len(settings.all_labels))
        return model


class Inception_v3Spawner:
    def __init__(self):
        self.name = "Inception_v3"

    def __call__(self):
        model = torchvision.models.inception_v3(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048,
                                   out_features=len(settings.all_labels))
        return model


class MNASNet1_0Spawner:
    def __init__(self):
        self.name = "MNASNet1_0"

    def __call__(self):
        model = torchvision.models.mnasnet1_0(pretrained=True)
        model.classifier = torch.nn.Linear(in_features=1280,
                                           out_features=len(
                                               settings.all_labels))
        return model


class MobileNet_v2Spawner:
    def __init__(self):
        self.name = "MobileNet_v2"

    def __call__(self):
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier = torch.nn.Linear(in_features=1280,
                                           out_features=len(
                                               settings.all_labels))
        return model


class ShuffleNet_v2_x1_0Spawner:
    def __init__(self):
        self.name = "ShuffleNet_v2_x1_0"

    def __call__(self):
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = torch.nn.Linear(in_features=1024,
                                   out_features=len(settings.all_labels))
        return model


class VGG11Spawner:
    def __init__(self):
        self.name = "VGG11"

    def __call__(self):
        model = torchvision.models.vgg11(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096,
                                              out_features=len(
                                                  settings.all_labels))
        return model


class VGG11BNSpawner:
    def __init__(self):
        self.name = "VGG11BN"  # batch normalized

    def __call__(self):
        model = torchvision.models.vgg11_bn(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096,
                                              out_features=len(
                                                  settings.all_labels))
        return model


class VGG13Spawner:
    def __init__(self):
        self.name = "VGG13"

    def __call__(self):
        model = torchvision.models.vgg13(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096,
                                              out_features=len(
                                                  settings.all_labels))
        return model


class VGG13BNSpawner:
    def __init__(self):
        self.name = "VGG13BN"  # batch normalized

    def __call__(self):
        model = torchvision.models.vgg13_bn(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096,
                                              out_features=len(
                                                  settings.all_labels))
        return model


class VGG16Spawner:
    def __init__(self):
        self.name = "VGG16"

    def __call__(self):
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096,
                                              out_features=len(
                                                  settings.all_labels))
        return model


class VGG16BNSpawner:
    def __init__(self):
        self.name = "VGG16BN"  # batch normalized

    def __call__(self):
        model = torchvision.models.vgg16_bn(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096,
                                              out_features=len(
                                                  settings.all_labels))
        return model


class VGG19Spawner:
    def __init__(self):
        self.name = "VGG19"

    def __call__(self):
        model = torchvision.models.vgg19(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096,
                                              out_features=len(
                                                  settings.all_labels))
        return model


class VGG19BNSpawner:
    def __init__(self):
        self.name = "VGG19BN"

    def __call__(self):
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096,
                                              out_features=len(
                                                  settings.all_labels))
        return model


class ResNeXt101_32x8dSpawner:
    def __init__(self):
        self.name = "ResNeXt-101_32x8d"

    def __call__(self):
        model = torchvision.models.resnext101_32x8d(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048,
                                   out_features=len(settings.all_labels))
        return model


class ResNeXt50_32x4dSpawner:
    def __init__(self):
        self.name = "ResNeXt-50_32x4d"

    def __call__(self):
        model = torchvision.models.resnext101_32x8d(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048,
                                   out_features=len(settings.all_labels))
        return model
