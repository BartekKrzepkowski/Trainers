n_epochs = 101
lr = 5e-3
latent_dim = 50

wae = WAEMMD(latent_dim).to(device)

optimizer_wae = torch.optim.Adam(wae.parameters(), lr=lr)
criterion_1 = torch.nn.MSELoss().to(device)

sigma = 2 * latent_dim * 0.01
C = 1.0

for epoch in range(n_epochs):
    epoch_losses_train = []
    for step, (x, _) in enumerate(train_loader):
        x = x.to(device)
        encoded, decoded = wae(x)
        rec_loss = criterion_1(decoded, x)
        latent_loss = wae.mmd_loss(encoded, sigma)
        loss = rec_loss + C * latent_loss
        optimizer_wae.zero_grad()
        loss.backward()
        optimizer_wae.step()

        epoch_losses_train += [loss.item()]

    epoch_losses_test = []
    for step, (x, _) in enumerate(test_loader):
        x = x.to(device)
        encoded, decoded = wae(x)
        rec_loss = criterion_1(decoded, x)
        latent_loss = wae.mmd_loss(encoded, sigma)
        loss = rec_loss + C * latent_loss
        epoch_losses_test.append(loss.item())

    print(
        f'Epoch: {epoch}  |  train loss: {np.mean(epoch_losses_train):.4f}  |  test loss: {np.mean(epoch_losses_test):.4f}')
