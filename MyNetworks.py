

import torch.nn as nn
import torch
import torch.nn.functional as F
from CoordConv import CoordConv2d
from deformConv import DeformableConv2d


class STNetBase(nn.Module):
    """
    This is the base model that uses Spatial Transformer Networks (STN).
    Literature: https://arxiv.org/abs/1506.02025
    The implementation is taken from: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    Last accessed: 19.10.2021

    This model is trained and tested with MNIST dataset.

    Input: 28x28x1 image.
    Output: 10 classes

    """

    def __init__(self, use_stn=True):
        super(STNetBase, self).__init__()
        self.use_stn=use_stn,
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # print(self.fc_loc[2].bias.data)

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.shape[1] * xs.shape[2] * xs.shape[3])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        if self.use_stn:
            x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
class STNetCoordConv(nn.Module):
    """
    This is the base model that uses Spatial Transformer Networks (STN).
    The network also uses CoordConv layers.
    STN literature: https://arxiv.org/abs/1506.02025
    CoordConv literature: https://arxiv.org/abs/1807.03247

    The base implementation is taken from: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    The CoordConv layer implementation is taken from https://github.com/walsvid/CoordConv
    Last accessed: 19.10.2021

    The model trained and tested with MNIST dataset.

    Input: 28x28x1 image.
    Output: 10 classes
    """

    def __init__(self, use_stn=True):
        super(STNetCoordConv, self).__init__()
        self.use_stn=use_stn
        self.conv1 = CoordConv2d(1, 10, kernel_size=5, with_r=False)
        self.conv2 = CoordConv2d(10, 20, kernel_size=5, with_r=False)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            CoordConv2d(1, 8, kernel_size=7, with_r=False),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            CoordConv2d(8, 10, kernel_size=5, with_r=False),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # print(self.fc_loc[2].bias.data)

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        if self.use_stn:
            x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class STNetBaseColorSVHN(nn.Module):
    """
    This is the base model that uses Spatial Transformer Networks (STN).
    Literature: https://arxiv.org/abs/1506.02025
    The base implementation is taken from: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    Last accessed: 19.10.2021

    Input: 32x32x3 image.
    Output: 10 classes

    This model is trained and tested with SVHN dataset <http://ufldl.stanford.edu/housenumbers/>
    """

    def __init__(self, use_stn=True):
        super(STNetBaseColorSVHN, self).__init__()
        self.use_stn=use_stn
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.shape[1] * xs.shape[2] * xs.shape[3])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        if self.use_stn:
            x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
class STNetBaseCoordConvColorSVHN(nn.Module):
    """
    This is the base model that uses Spatial Transformer Networks (STN).
    The network also uses CoordConv layers.
    STN literature: https://arxiv.org/abs/1506.02025
    CoordConv literature: https://arxiv.org/abs/1807.03247

    The base implementation is taken from: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    The CoordConv layer implementation is taken from https://github.com/walsvid/CoordConv
    Last accessed: 19.10.2021

    The model is trained and tested with STL10 dataset <https://cs.stanford.edu/~acoates/stl10/>.

    Input: 96x96x3 image.
    Output: 10 classes
    """


    def __init__(self, use_stn=True):
        super(STNetBaseCoordConvColorSVHN, self).__init__()
        self.use_stn=use_stn

        self.conv1 = CoordConv2d(3, 10, kernel_size=5, with_r=False)
        self.conv2 = CoordConv2d(10, 20, kernel_size=5, with_r=False)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(8820, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 20 * 20, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        # print(xs.shape)
        xs = xs.view(-1, xs.shape[1] * xs.shape[2] * xs.shape[3])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        if self.use_stn:
            x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class STNetBaseColorSTL(nn.Module):
    """
    This is the base model that uses Spatial Transformer Networks (STN).
    Literature: https://arxiv.org/abs/1506.02025
    The base implementation is taken from: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    Last accessed: 19.10.2021

    The model is trained and tested with STL10 dataset <https://cs.stanford.edu/~acoates/stl10/>.

    Input: 96x96x3 image.
    Output: 10 classes
    """


    def __init__(self, use_stn=True):
        super(STNetBaseColorSTL, self).__init__()
        self.use_stn = use_stn
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(8820, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 20 * 20, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        # print(xs.shape)
        xs = xs.view(-1, xs.shape[1] * xs.shape[2] * xs.shape[3])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        if self.use_stn:
            x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
class STNetBaseCoordConvColorSTL(nn.Module):
    """
    This is the base model that uses Spatial Transformer Networks (STN).
    The network also uses CoordConv layers.
    STN literature: https://arxiv.org/abs/1506.02025
    CoordConv literature: https://arxiv.org/abs/1807.03247

    The base implementation is taken from: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    The CoordConv layer implementation is taken from https://github.com/walsvid/CoordConv
    Last accessed: 19.10.2021

    The model is trained and tested with STL10 dataset <https://cs.stanford.edu/~acoates/stl10/>.

    Input: 96x96x3 image.
    Output: 10 classes
    """


    def __init__(self, use_stn=True):
        super(STNetBaseCoordConvColorSTL, self).__init__()
        self.use_stn=use_stn

        self.conv1 = CoordConv2d(3, 10, kernel_size=5, with_r=False)
        self.conv2 = CoordConv2d(10, 20, kernel_size=5, with_r=False)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(8820, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 20 * 20, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        # print(xs.shape)
        xs = xs.view(-1, xs.shape[1] * xs.shape[2] * xs.shape[3])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        if self.use_stn:
            x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)





class STNetBaseDeformConv(nn.Module):
    """
    This is the base model that uses Spatial Transformer Networks (STN).
    The network also uses Deformable ConvNets (v2) layers.
    STN literature: https://arxiv.org/abs/1506.02025
    Deformable ConvNets (v2) literature: https://arxiv.org/abs/1811.11168

    The base implementation is taken from: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    The CoordConv layer implementation is taken from https://github.com/walsvid/CoordConv
    Last accessed: 19.10.2021

    The model trained and tested with MNIST dataset.

    Input: 28x28x1 image.
    Output: 10 classes
    """

    def __init__(self, use_stn=False):
        super(STNetBaseDeformConv, self).__init__()
        self.use_stn = use_stn
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop1 = nn.Dropout2d()
        self.conv3 = DeformableConv2d(20, 30, kernel_size=5, stride=1, padding=1, bias=True)
        self.conv4 = DeformableConv2d(30, 40, kernel_size=5, stride=1, padding=1, bias=True)
        self.drop2 = nn.Dropout2d()
        self.fc1 = nn.Linear(120, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 20 * 20, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        # print(xs.shape)
        xs = xs.view(-1, xs.shape[1] * xs.shape[2] * xs.shape[3])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        if self.use_stn:
            x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.drop1(self.conv2(x)), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.drop2(self.conv4(x)))

        # print(x.shape)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)




# from transformers import ViTModel, ViTConfig  # , ViTFeatureExtractor
# class STNetBaseViT(nn.Module):
#     """
#     This is the base model that uses Visual Transformer (ViT) and Spatial transformer localization-network
#
#     ViT literature: https://arxiv.org/abs/2010.11929
#     STN literature: https://arxiv.org/abs/1506.02025
#
#     The PyTorch implementation of ViT is provided by Hugging Face AI Community. <https://huggingface.co/>
#
#     The base implementation is taken from: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
#     Last accessed: 19.10.2021s
#
#     The model is trained and tested with MNIST dataset.
#
#     Input: 28x28x1 image.
#     Output: 10 classes
#     """
#
#     def __init__(self, use_stn=False):
#         super(STNetBaseViT, self).__init__()
#         self.use_stn = use_stn
#         self.configuration = ViTConfig(
#             hidden_size=512,
#             num_hidden_layers=1,
#             num_attention_heads=4,
#             intermediate_size=320,
#             hidden_act='relu',
#             hidden_dropout_prob=0.01,
#             attention_probs_dropout_prob=0.01,
#             layer_norm_eps=1e-12,
#             image_size=28,
#             patch_size=7,
#             num_channels=1
#         )
#         self.vit = ViTModel(self.configuration)
#         self.config = self.vit.config
#         self.config.hidden_states=False
#         self.config.output_attentions = False
#
#         self.fc = nn.Linear(512, 10)
#
#
#         # Spatial transformer localization-network
#         self.localization = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#
#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 3 * 3, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )
#
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#         # print(self.fc_loc[2].bias.data)
#
#     # Spatial transformer network forward function
#     def stn(self, x):
#         xs = self.localization(x)
#         xs = xs.view(-1, 10 * 3 * 3)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)
#
#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
#
#         return x
#
#     def forward(self, x):
#         # transform the input
#         if self.use_stn:
#             x = self.stn(x)
#         x = self.vit(x).pooler_output
#         # Perform the usual forward pass
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)
