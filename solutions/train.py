def train(model,criterion,optimizer, earlystopping=True,max_epochs=30,patience=2, augment=False):
    train_history = []
    valid_history = []
    accuracy_history = []
    estop = EarlyStopping(patience=patience)
    for epoch in range(max_epochs):
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if augment:
                # generate transform list
                p = np.random.random() # probability of each transformation occurring
                transforms = transform_list(MAGN,p)
                # get the inputs; data is a list of [inputs, labels]
                xb,yb = data
                xb = xb.to(device)
                yb = yb.to(device)
                # generate the tensors 'C_images' and 'C_targets'
                C_images, C_targets = compute_composed_data(transforms, L, C, xb,yb)
                # generated the augmented data = [inputs,labels]
                inputs, labels = compute_selected_data(model, criterion, C_images, C_targets, S)
            else:
                # get the inputs; data is a list of [inputs, labels]
                inputs,labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        valid_loss, accuracy = validation(model,criterion)
        train_history.append(train_loss)
        valid_history.append(valid_loss)
        accuracy_history.append(accuracy)
        print('Epoch %02d: train loss %0.5f, validation loss %0.5f, accuracy %3.1f ' % (epoch, train_loss, valid_loss, accuracy))
        estop.step(valid_loss)
        if earlystopping and estop.early_stop:
            break
    return train_history, valid_history, accuracy_history
