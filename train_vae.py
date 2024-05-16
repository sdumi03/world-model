import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image

from os import makedirs
from os.path import join, exists

from models import vae
from utils import misc, learning, loaders


def vae_loss_function(recon_x, x, mu, logsigma):
    BCE = torch.nn.functional.mse_loss(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


def test(model, test_loader, device):
    model.eval()
    test_loader.dataset.load_next_buffer()
    test_loss = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.float().to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += vae_loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print(f"===> Average test loss: {test_loss:.4f}")
    return test_loss


def train(model, train_loader, optimizer, device):
    model.train()
    train_loader.dataset.load_next_buffer()
    train_loss = 0

    for data in train_loader:
        data = data.float().to(device)
        print(data[0][0])

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"===> Average train loss: {train_loss / len(train_loader.dataset):.4f}")


def main(args):
    # Fix numeric divergence due to bug in Cudnn
    # torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((misc.RED_SIZE, misc.RED_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((misc.RED_SIZE, misc.RED_SIZE)),
        transforms.ToTensor()
    ])

    rollouts_path = join('datasets', args.env)
    assert exists(rollouts_path), 'Rollouts does not exists'

    dataset_train = loaders.RolloutObservationDataset(
        rollouts_path, transform_train, args.dimension, train=True
    )
    dataset_test = loaders.RolloutObservationDataset(
        rollouts_path, transform_test, args.dimension, train=False
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    model = vae.MODEL(misc.IMAGE_CHANNELS, misc.LATENT_SIZE, args.dimension).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = learning.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = learning.EarlyStopping('min', patience=30)

    vae_dir = join('trained', args.env, 'vae')
    makedirs(vae_dir, exist_ok=True)
    makedirs(join(vae_dir, 'samples'), exist_ok=True)

    best_file = join(vae_dir, 'best.tar')

    if not args.noreload and exists(best_file):
        state = torch.load(best_file)
        print(
            f"Reloading model at epoch {state['epoch']}, with test error {state['precision']}"
        )
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        earlystopping.load_state_dict(state['earlystopping'])

    if not args.nosamples:
        samples_every = [ix for ix in range(10, args.epochs + 1, 10)]

    current_best = None
    for epoch in range(1, args.epochs + 1):
        print()

        # Training
        train(model, train_loader, optimizer, device)
        test_loss = test(model, test_loader, device)
        scheduler.step(test_loss)
        earlystopping.step(test_loss)

        # Checkpointing
        if not current_best or test_loss < current_best:
            current_best = test_loss

            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'precision': test_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'earlystopping': earlystopping.state_dict()
            }, best_file)

        if not args.nosamples and epoch in samples_every:
            with torch.no_grad():
                sample = torch.randn(misc.RED_SIZE, misc.LATENT_SIZE).to(device)
                sample = model.decoder(sample).cpu()
                for i in range(len(sample)):
                    save_image(
                        sample[i].view(misc.IMAGE_CHANNELS, misc.RED_SIZE, misc.RED_SIZE),
                        join(vae_dir, 'samples', f"sample_{epoch}_{i}.png")
                    )

        if earlystopping.stop:
            print(f"End of Training because of early stopping at epoch {epoch}")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', type=str, help='Enviroment for training'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32, help='Batch size for training'
    )
    parser.add_argument(
        '--epochs', type=int, default=1000, help='Number of epochs for training'
    )
    parser.add_argument(
        '--noreload', action='store_true', help='Best model is not reloaded if specified'
    )
    parser.add_argument(
        '--nosamples', action='store_true', help='Does not save samples during training if specified'
    )
    parser.add_argument(
        '--dimension', type=str, default='1d', help='Dimension of the VAE model (1d or 2d)'
    )
    args = parser.parse_args()

    main(args)