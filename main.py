

if __name__ == "__main__":
  """
  This file executes all the networks.
  """
  import stn_mnist, stn_stl10, stn_svhn, stn_deformConv_mnist, vit_mnist
  import warnings
  warnings.filterwarnings('ignore')
  import os

  EPOCH=10
  VERBOSE=True

  
  print("\Base network with STN but without coordConv, trained on MNIST (28x28 digit images)")
  stn_mnist.stn_base(train_epoch=EPOCH, use_stn=True, use_coord_conv=False, verbose=VERBOSE, flag="with_stn_without_coord")
  print("--"*20)
  print("\Base network with STN and coordConv, trained on MNIST")
  stn_mnist.stn_base(train_epoch=EPOCH, use_stn=True, use_coord_conv=True, verbose=VERBOSE, flag="with_stn_with_coord")
  
  print("--"*20)
  print("\Base network with STN but without coordConv, trained on SVHN (32x32x3 color digit images)")
  stn_svhn.stn_color_svhn(train_epoch=EPOCH, use_stn=True, use_coord_conv=False, verbose=VERBOSE, flag="with_stn_without_coord")
  print("--"*20)
  print("\Base network with STN and coordConv, trained on SVHN (32x32x3 color digit images)")
  stn_svhn.stn_color_svhn(train_epoch=EPOCH, use_stn=True, use_coord_conv=True, verbose=VERBOSE, flag="with_stn_with_coord")

  print("--"*20)
  print("\Base network with STN but without coordConv, trained on SLT10 (96x96x3 color natural images)")
  stn_stl10.stn_color_stl(train_epoch=EPOCH+10, use_stn=True, use_coord_conv=False, verbose=VERBOSE, flag="with_stn_without_coord")
  print("--"*20)
  print("\Base network with STN and coordConv, trained on SLT10 (96x96x3 color natural images)")
  stn_stl10.stn_color_stl(train_epoch=EPOCH+10, use_stn=True, use_coord_conv=False, verbose=VERBOSE, flag="with_stn_with_coord")
  
  print("--"*20)
  print("\Base network with STN and deformable CovNets (v2), trained on MNIST (28x28 digit images)")
  stn_deformConv_mnist.stn_deform_conv(train_epoch=EPOCH, use_stn=True, use_coord_conv=False, verbose=VERBOSE, flag="with_stn")

  print("--"*20)
  print("\nSTN base mnist with stn with coord")
  vit_mnist.vit_base(train_epoch=EPOCH, use_stn=True, use_coord_conv=False, verbose=VERBOSE, flag="with_stn")

  root_path = os.path.abspath(os.path.dirname(__file__))
  data_path = os.path.join(root_path, "data/")
  fig_path = os.path.join(root_path, "figures/")
  model_path = os.path.join(root_path, "savedModels/")

  print("The results with related figures and trained models are stored in the following directories: \n{}\n{}".format(fig_path, model_path))
