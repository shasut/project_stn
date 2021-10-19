from __future__ import print_function

def stn_base(train_epoch=15, use_stn=True, use_coord_conv=False, verbose=False, flag=None):
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision

    import matplotlib
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab! (while using headless gpu cluser server)
    import matplotlib.pyplot as plt
    import numpy as np
    import datetime

    import os
    # print(os.getcwd())
    import time

    from my_utils import createDirectoryIfNotExits, plot_loss, roc_multiclass, plot_confusion_matrix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    def convert_image_np(inp):
        """Convert a Tensor to numpy image."""
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp
    def visualize_stn():
        with torch.no_grad():
            # Get a batch of training data
            data = next(iter(test_loader))[0].to(device)

            input_tensor = data.cpu()
            transformed_input_tensor = model.stn(data).cpu()

            in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))

            out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor))

            # Plot the results side-by-side
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Dataset Images')

            axarr[1].imshow(out_grid)
            axarr[1].set_title('Transformed Images')

            f.savefig(fig_path + MODEL_NAME + "_" + flag + "_" + "grid.png", bbox_inches='tight', transparent=False)

    root_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(root_path, "data/")
    fig_path = os.path.join(root_path, "figures/")
    model_path = os.path.join(root_path, "savedModels/")
    createDirectoryIfNotExits(data_path)
    createDirectoryIfNotExits(fig_path)
    createDirectoryIfNotExits(model_path)

    TRAIN_EP = train_epoch
    LRT = 1e-2
    BATCH_SIZE = 100

    from my_data_loaders import my_data_loader_mnist
    train_loader, test_loader = my_data_loader_mnist(data_path, BATCH_SIZE)

    from MyNetworks import STNetBase, STNetCoordConv

    if use_coord_conv==False:
        model = STNetBase(use_stn=use_stn).to(device)
        MODEL_NAME = "STNBase"
        # print("Model: STN Base Model")
    else:
        model = STNetCoordConv(use_stn=use_stn).to(device)
        MODEL_NAME = "STNetCoordConv"
        # print("Model: STN with CoordConv Base Model")


    # from torchsummary import summary
    # summary(model, (1, 28, 28))

    optimizer = optim.SGD(model.parameters(), lr=LRT)


    train_losses=[]
    val_losses=[]
    it = time.time()
    for epoch in range(TRAIN_EP):
        model.train()

        running_loss = 0
        t = time.time()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device, dtype=torch.long)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss/len(train_loader))

        with torch.no_grad():
            model.eval()
            running_loss = 0
            correct = 0
            for data_v, target_v in test_loader:
                data_v, target_v = data_v.to(device), target_v.to(device)
                output_v = model(data_v)
                loss_v = F.nll_loss(output_v, target_v)
                running_loss += loss_v.item()

                pred = output_v.max(1, keepdim=True)[1]
                correct += pred.eq(target_v.view_as(pred)).sum().item()

        val_acc = 100. * correct / len(test_loader.dataset)

        val_losses.append(running_loss/len(test_loader))

        t=(time.time()-t)/60
        np.save(data_path + "train_loss.npy", np.array(train_losses))
        np.save(data_path + "val_loss.npy", np.array(val_losses))

        if verbose:
            my_print='Epoch = {}   ||   train_Loss = {:6.4f}   ||   val_Loss = {:6.4f}   ||  val_ACC = {:6.4f}   ||   Duration = {:4.2f} min'.format(len(train_losses), train_losses[-1], val_losses[-1], val_acc, t )
            print("{:80}".format(my_print))

    t = round(((time.time() - it) / 60),2)
    print("total training time is {} min, for total {} epochs".format(t, TRAIN_EP))


    PATH = model_path + MODEL_NAME + "_" + flag + "_" +  "mnist.pth"
    torch.save({
        'model_archi': MODEL_NAME,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_loss': val_losses,
        'training_loss': train_losses,
    }, PATH)
    plot_loss(train_losses, val_losses, saveLocation=fig_path, plot_title= MODEL_NAME + "_" + flag)

    # Visualize the STN transformation on some input batch
    if use_stn:
        visualize_stn()

    roc_multiclass(model, device, test_loader, num_classes=10, saveLocation=fig_path, plot_title=MODEL_NAME+ "_" + flag)
    plot_confusion_matrix(model, device, test_loader, saveLocation=fig_path, plot_title=MODEL_NAME+ "_" + flag)
