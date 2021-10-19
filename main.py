
import stn_mnist, stn_stl10, stn_svhn, stn_deformConv_mnist#, vit_mnist
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    """
    This file executes all the networks.
    """

    EPOCH=2
    VERBOSE=True
    #
    # print("\nSTN base mnist with stn without coordConv")
    # stn_mnist.stn_base(train_epoch=EPOCH, use_stn=True, use_coord_conv=False, verbose=VERBOSE, flag="with_stn_without_coord")
    # print("--"*20)
    # print("\nSTN base mnist with stn with coordConv")
    # stn_mnist.stn_base(train_epoch=EPOCH, use_stn=True, use_coord_conv=True, verbose=VERBOSE, flag="with_stn_with_coord")
    #
    # print("--"*20)
    # print("\nSTN base svhn with stn without coordConv")
    # stn_svhn.stn_color_svhn(train_epoch=EPOCH, use_stn=True, use_coord_conv=False, verbose=VERBOSE, flag="with_stn_without_coord")
    # print("--"*20)
    # print("\nSTN base svhn with stn with coordConv")
    # stn_svhn.stn_color_svhn(train_epoch=EPOCH, use_stn=True, use_coord_conv=True, verbose=VERBOSE, flag="with_stn_with_coord")

    # print("--"*20)
    # print("\nSTN base stl10 with stn with coordConv")
    # stn_stl10.stn_color_stl(train_epoch=EPOCH, use_stn=True, use_coord_conv=False, verbose=VERBOSE, flag="with_stn_without_coord")
    # print("--"*20)
    # print("\nSTN base stl10 with stn with coordConv")
    # stn_stl10.stn_color_stl(train_epoch=EPOCH, use_stn=True, use_coord_conv=False, verbose=VERBOSE, flag="with_stn_with_coord")
    #
    # print("--"*20)
    print("\nSTN base mnist with stn with coord")
    stn_deformConv_mnist.stn_deform_conv(train_epoch=EPOCH, use_stn=True, use_coord_conv=False, verbose=VERBOSE, flag="with_stn")

    print("--"*20)
    print("\nSTN base mnist with stn with coord")
    vit_mnist.vit_base(train_epoch=EPOCH, use_stn=True, use_coord_conv=False, verbose=VERBOSE, flag="with_stn")

    # pass