#disable gradient calculation
@torch.no_grad()
def compute_selected_data(model, loss, C_images, C_targets, S):
    C, BS, N_CHANNELS, HEIGHT, WIDTH = C_images.shape

    # create a list of predictions 'pred' by applying the model to the augmented batches contained in C_images
    preds = [model(images) for images in C_images]

    # create a list of losses by applying the loss function to the predictions and labels C_targets
    # convert the list to a loss tensor 'loss_tensor' through the function torch.stack
    loss_tensor = torch.stack([loss(pred, C_targets[i]) for i,pred in enumerate(preds)])

    # select the S indices 'S_idxs' of the loss_tensor with the highest value. You may use the function torch.topk
    S_idxs = loss_tensor.topk(S, dim=0).indices

    # select the S images 'S_images' from C_images with the highest losses
    S_images = C_images[S_idxs]
    # convert the tensor 'S_images' so that it passes from shape [S, BS, N_CHANNELS, HEIGHT, WIDTH] to shape
    # [S*BS, N_CHANNELS, HEIGHT, WIDTH]. You may use the function torch.view
    S_images = S_images.view(-1, N_CHANNELS, HEIGHT, WIDTH)

    # select the S labels 'S_targets' from C_targets corresponding to the highest losses
    S_targets = C_targets[S_idxs]
    # convert the tensor 'S_targets' so that it passes from shape [S, BS] to shape
    # [S*BS]. You may use the function torch.view
    S_targets = S_targets.view(-1)

    return S_images, S_targets
