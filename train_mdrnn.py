import argparse
import numpy as np
import torch
from torchvision import transforms

from os import makedirs
from os.path import join, exists

from models import vae, mdrnn
from utils import misc, learning, loaders


def to_latent(vae_model, state, next_state, dimension):
    """ Transform observations to latent space.

    :args state: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_state: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_state, latent_next_state)
        - latent_state: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - latent_next_state: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        if dimension == '1d':
            state, next_state = [
                x.view(-1, x.size(-2), x.size(-1))
                for x in (state, next_state)
            ]

        if dimension == '2d':
            img_size = state.shape[-1]
            state, next_state = [
                torch.nn.functional.upsample(
                    x.view(-1, 3, img_size, img_size),
                    size=misc.RED_SIZE,
                    mode='bilinear',
                    align_corners=True
                )
                for x in (state, next_state)
            ]

        (state_mu, state_logsigma), (next_state_mu, next_state_logsigma) = [
            vae_model(x)[1:]
            for x in (state, next_state)
        ]

        if dimension == '1d':
            latent_state, latent_next_state = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu))
                for x_mu, x_logsigma in
                [(state_mu, state_logsigma), (next_state_mu, next_state_logsigma)]
            ]

        if dimension == '2d':
            state_mu = state_mu.unsqueeze(1)
            state_logsigma = state_logsigma.unsqueeze(1)
            next_state_mu = next_state_mu.unsqueeze(1)
            next_state_logsigma = next_state_logsigma.unsqueeze(1)
            bs = state_logsigma.shape[0]

            latent_state, latent_next_state = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(bs, misc.SEQ_LEN, misc.LATENT_SIZE)
                for x_mu, x_logsigma in
                [(state_mu, state_logsigma), (next_state_mu, next_state_logsigma)]
            ]

    return latent_state, latent_next_state


def get_loss(mdrnn_model, latent_state, action, reward, done, latent_next_state, include_reward, dimension):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_state, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(done, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_state: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_state: (BSIZE, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    if dimension == '1d':
        action = action.view(-1)
        reward = reward.view(-1)
        done = done.view(-1)

    if dimension == '2d':
        latent_state, action, reward, done, latent_next_state = \
            [
                arr.transpose(1, 0)
                for arr in [latent_state, action, reward, done, latent_next_state]
            ]

    mus, sigmas, logpi, rewards, dones = mdrnn_model(action, latent_state)
    gmm = mdrnn.gmm_loss(latent_next_state, mus, sigmas, logpi)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(dones, done)

    if include_reward:
        mse = torch.nn.functional.mse_loss(rewards, reward)
        scale = misc.LATENT_SIZE + 2
    else:
        mse = 0
        scale = misc.LATENT_SIZE + 1

    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


def test(mdrnn_model, vae_model, test_loader, device, batch_size, include_reward, dimension):
    mdrnn_model.eval()
    test_loader.dataset.load_next_buffer()
    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    for ix, data in enumerate(test_loader):
        state, action, reward, done, next_state = [arr.to(device) for arr in data]

        latent_state, latent_next_state = to_latent(vae_model, state, next_state, dimension)

        with torch.no_grad():
            losses = get_loss(
                mdrnn_model, latent_state, action, reward,
                done, latent_next_state, include_reward, dimension
            )

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else losses['mse']

    test_loss = cum_loss * batch_size / len(test_loader.dataset)
    print(f"===> Average test loss: {test_loss:.4f}")
    return test_loss


def train(mdrnn_model, vae_model, train_loader, optimizer, device, batch_size, include_reward, dimension):
    mdrnn_model.train()
    train_loader.dataset.load_next_buffer()
    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    for ix, data in enumerate(train_loader):
        state, action, reward, done, next_state = [arr.to(device) for arr in data]

        latent_state, latent_next_state = to_latent(vae_model, state, next_state, dimension)

        losses = get_loss(
            mdrnn_model, latent_state, action, reward,
            done, latent_next_state, include_reward, dimension
        )

        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else losses['mse']

    print(f"===> Average train loss: {cum_loss * batch_size / len(train_loader.dataset):.4f}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Lambda(
        lambda x: np.transpose(x, (0, 3, 1, 2)) / 255
    )

    rollouts_path = join('datasets', args.env)
    assert exists(rollouts_path), 'Rollouts does not exists'

    dataset_train = loaders.RolloutSequenceDataset(
        rollouts_path, misc.SEQ_LEN, transform, args.dimension, buffer_size=30
    )
    dataset_test = loaders.RolloutSequenceDataset(
        rollouts_path, misc.SEQ_LEN, transform, args.dimension, buffer_size=10, train=False
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    vae_file = join('trained', args.env, 'vae', 'best.tar')
    assert exists(vae_file), 'VAE is not trained'

    vae_state = torch.load(vae_file)
    print(
        f"Loading VAE at epoch {vae_state['epoch']} with test error {vae_state['precision']}"
    )

    vae_model = vae.MODEL(misc.IMAGE_CHANNELS * self.SEQ_LEN, misc.LATENT_SIZE, args.dimension).to(device)
    vae_model.load_state_dict(vae_state['state_dict'])

    mdrnn_model = mdrnn.MODEL(misc.LATENT_SIZE, misc.ACTION_SIZE, misc.R_SIZE, 5, args.dimension).to(device)
    optimizer = torch.optim.RMSprop(mdrnn_model.parameters(), lr=1e-3, alpha=.9)
    scheduler = learning.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = learning.EarlyStopping('min', patience=30)

    rnn_dir = join('trained', args.env, 'mdrnn')
    makedirs(rnn_dir, exist_ok=True)

    rnn_file = join(rnn_dir, 'best.tar')

    if not args.noreload and exists(rnn_file):
        rnn_state = torch.load(rnn_file)
        print(
            f"Reloading MDRNN at epoch {rnn_state['epoch']}, with test error {rnn_state['precision']}"
        )
        mdrnn_model.load_state_dict(rnn_state['state_dict'])
        optimizer.load_state_dict(rnn_state['optimizer'])
        scheduler.load_state_dict(vae_state['scheduler'])
        earlystopping.load_state_dict(vae_state['earlystopping'])

    current_best = None
    for epoch in range(1, args.epochs + 1):
        print()
        print('Epoch:', epoch)

        # Training
        train(mdrnn_model, vae_model, train_loader, optimizer, device, args.batch_size, args.include_reward, args.dimension)
        test_loss = test(mdrnn_model, vae_model, test_loader, device, args.batch_size, args.include_reward, args.dimension)
        scheduler.step(test_loss)
        earlystopping.step(test_loss)

        # Checkpointing
        if not current_best or test_loss < current_best:
            current_best = test_loss

            torch.save({
                'epoch': epoch,
                'state_dict': mdrnn_model.state_dict(),
                'precision': test_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'earlystopping': earlystopping.state_dict()
            }, rnn_file)

        if earlystopping.stop:
            print(f"End of Training because of early stopping at epoch {epoch}")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', type=str, help='Enviroment for training'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16, help='Batch size for training'
    )
    parser.add_argument(
        '--epochs', type=int, default=1000, help='Number of epochs for training'
    )
    parser.add_argument(
        '--noreload', action='store_true', help='Best model is not reloaded if specified'
    )
    parser.add_argument(
        '--include-reward', action='store_true', help='Add a reward modelisation term to the loss'
    )
    parser.add_argument(
        '--dimension', type=str, default='1d', help='Dimension of the MDRNN model (1d or 2d)'
    )
    args = parser.parse_args()

    main(args)