import argparse

import torch
from torchvision import transforms
from torchvision.utils import save_image

from models import vae # vae.MODEL

# from utils.misc import save_checkpoint
# from utils.misc import LSIZE, RED_SIZE
from utils import misc

# from utils.learning import EarlyStopping
# from utils.learning import ReduceLROnPlateau
from utils import learning

# from utils.loaders import RolloutObservationDataset
from utils import loaders

from os import makedirs
from os.path import join, exists


def vae_loss_function(recon_x, x, mu, logsigma):
    BCE = torch.nn.functional.mse_loss(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


def test(model, dataset_test, test_loader):
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += vae_loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")
    return test_loss


def train(epoch, model, dataset_train, optimizer):
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]"
            )
            print(
                f"Loss: {loss.item() / len(data):.6f}"
            )

    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")


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

    rollouts_path = join('datasets', args.generator)
    assert exists(rollouts_path), 'Rollouts does not exists...'

    dataset_train = loaders.RolloutObservationDataset(
        rollouts_path, transform_train, train=True
    )
    dataset_test = loaders.RolloutObservationDataset(
        rollouts_path, transform_test, train=False
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    model = vae.MODEL(misc.IMAGE_CHANNELS, misc.LATENT_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30)

    vae_dir = join('trained', args.generator, 'vae')
    makedirs(vae_dir, exist_ok=True)
    makedirs(join(vae_dir, 'samples'), exist_ok=True)

    best_filename = join(vae_dir, 'best.tar')

    if not args.noreload and exists(best_filename):
        state = torch.load(best_filename)
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
        train(epoch)
        test_loss = test()
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
            }, best_filename)
        
        if not args.nosamples and epoch in samples_every:
            with torch.no_grad():
                sample = torch.randn(RED_SIZE, LSIZE).to(device)
                sample = model.decoder(sample).cpu()
                save_image(
                    sample.view(64, 3, RED_SIZE, RED_SIZE),
                    join(vae_dir, 'samples', f"sample_{epoch}.png")
                )

        if earlystopping.stop:
            print(f"End of Training because of early stopping at epoch {epoch}")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generator', type=str, help='Generator of rollouts'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32, help='Batch size for training'
    )
    parser.add_argument(
        '--epochs', type=int, default=1000, help='Number of epochs to train'
    )
    # parser.add_argument(
    #     '--logdir', type=str, help='Directory where results are logged'
    # )
    parser.add_argument(
        '--noreload', action='store_true', help='Best model is not reloaded if specified'
    )
    parser.add_argument(
        '--nosamples', action='store_true', help='Does not save samples during training if specified'
    )
    args = parser.parse_args()

    main(args)