def compute_composed_data(transform_list,L, C, xb,yb):
    BS,N_CHANNELS,HEIGHT,WIDTH = xb.shape

    C_images = torch.zeros(C, BS, N_CHANNELS, HEIGHT, WIDTH, device=device)
    C_targets = torch.zeros(C, BS, device=device, dtype=torch.long)

    for c in range(C):
        # create a list of L linear transforms randomly sampled from the transform_list
        sampled_tfms = list(np.random.choice(transform_list, L, replace=False))
        # create a composition of transforms from the list sampled above. Use nn.Sequential instead of transforms.Compose in order to script the transformations
        transform = nn.Sequential(*sampled_tfms)
        # apply the composition to the original images xb
        xbt,ybt = transform(xb),yb
        # update tensors C_images and C_targets
        C_images[c],C_targets[c] = xbt,ybt

    return C_images, C_targets
