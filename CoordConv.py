import torch
import torch.nn as nn
import torch.nn.modules.conv as conv


class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)
            zz_channel = zz_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
                zz_channel = zz_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out

class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out

# class AddCoordinates(object):
#
#     r"""Coordinate Adder Module as defined in 'An Intriguing Failing of
#     Convolutional Neural Networks and the CoordConv Solution'
#     (https://arxiv.org/pdf/1807.03247.pdf).
#     This module concatenates coordinate information (`x`, `y`, and `r`) with
#     given input tensor.
#     `x` and `y` coordinates are scaled to `[-1, 1]` range where origin is the
#     center. `r` is the Euclidean distance from the center and is scaled to
#     `[0, 1]`.
#     Args:
#         with_r (bool, optional): If `True`, adds radius (`r`) coordinate
#             information to input image. Default: `False`
#     Shape:
#         - Input: `(N, C_{in}, H_{in}, W_{in})`
#         - Output: `(N, (C_{in} + 2) or (C_{in} + 3), H_{in}, W_{in})`
#     Examples:
#         # >>> coord_adder = AddCoordinates(True)
#         # >>> input = torch.randn(8, 3, 64, 64)
#         # >>> output = coord_adder(input)
#         # >>> coord_adder = AddCoordinates(True)
#         # >>> input = torch.randn(8, 3, 64, 64).cuda()
#         # >>> output = coord_adder(input)
#         # >>> device = torch.device("cuda:0")
#         # >>> coord_adder = AddCoordinates(True)
#         # >>> input = torch.randn(8, 3, 64, 64).to(device)
#         # >>> output = coord_adder(input)
#     """
#
#     def __init__(self, with_r=False):
#         self.with_r = with_r
#
#     def __call__(self, image):
#         batch_size, _, image_height, image_width = image.size()
#
#         y_coords = 2.0 * torch.arange(image_height).unsqueeze(
#             1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
#         x_coords = 2.0 * torch.arange(image_width).unsqueeze(
#             0).expand(image_height, image_width) / (image_width - 1.0) - 1.0
#
#         coords = torch.stack((y_coords, x_coords), dim=0)
#
#         if self.with_r:
#             rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
#             rs = rs / torch.max(rs)
#             rs = torch.unsqueeze(rs, dim=0)
#             coords = torch.cat((coords, rs), dim=0)
#
#         coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
#
#         image = torch.cat((coords.to(image.device), image), dim=1)
#
#         return image


# class CoordConv2d(nn.Module):
#
#     r"""2D Convolution Module Using Extra Coordinate Information as defined
#     in 'An Intriguing Failing of Convolutional Neural Networks and the
#     CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).
#     Args:
#         Same as `torch.nn.Conv2d` with two additional arguments
#         with_r (bool, optional): If `True`, adds radius (`r`) coordinate
#             information to input image. Default: `False`
#     Shape:
#         - Input: `(N, C_{in}, H_{in}, W_{in})`
#         - Output: `(N, C_{out}, H_{out}, W_{out})`
#     Examples:
#         # >>> coord_conv = CoordConv(3, 16, 3, with_r=True)
#         # >>> input = torch.randn(8, 3, 64, 64)
#         # >>> output = coord_conv(input)
#         # >>> coord_conv = CoordConv(3, 16, 3, with_r=True).cuda()
#         # >>> input = torch.randn(8, 3, 64, 64).cuda()
#         # >>> output = coord_conv(input)
#         # >>> device = torch.device("cuda:0")
#         # >>> coord_conv = CoordConv(3, 16, 3, with_r=True).to(device)
#         # >>> input = torch.randn(8, 3, 64, 64).to(device)
#         # >>> output = coord_conv(input)
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1, bias=True,
#                  with_r=False):
#         super(CoordConv2d, self).__init__()
#
#         in_channels += 2
#         if with_r:
#             in_channels += 1
#
#         self.conv_layer = nn.Conv2d(in_channels, out_channels,
#                                     kernel_size, stride=stride,
#                                     padding=padding, dilation=dilation,
#                                     groups=groups, bias=bias)
#
#         self.coord_adder = AddCoordinates(with_r)
#
#     def forward(self, x):
#         x = self.coord_adder(x)
#         x = self.conv_layer(x)
#
#         return x
