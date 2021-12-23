import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data.data_loading import get_data
from gan.cGAN import save_models, print_training_progress, get_device, generator_loss, \
    discriminator_loss, weights_init, generate_noise, load_model
from gan.gan_models import LimbLengthGenerator, LimbLengthDiscriminator
from pose.pose_utils import GOOD_POSES_PATH, NOISE_DIMENSION, CLIMBGAN_PATH


def parse_args():
    parser = argparse.ArgumentParser(description='cLimbGAN correction.')
    parser.add_argument("-data", type=str,
                        required=False, default=GOOD_POSES_PATH,
                        help="Path to the good poses that we want to train the cLimbGAN on.")
    parser.add_argument("-path", type=str,
                        required=False, default=CLIMBGAN_PATH,
                        help="Path to load and save the model from.")
    parser.add_argument("-scratch", type=bool, default=False,
                        help="Train the GAN from scratch or resume training a previous model.")
    parser.add_argument("-version", type=int,
                        required=False, default=780,
                        help="If you don't train from scratch, specify the model version to be resumed for training.")
    parser.add_argument("-epochs", type=int,
                        required=False, default=1000,
                        help="How many epochs to train the model for.")

    return parser.parse_args()


def main(args):
    # Model parameters
    batch_size = 64
    learning_rate = 0.0002
    split_ratio = 1
    device = get_device()

    # Set fixed random number seed
    torch.manual_seed(42)

    train_loader, _, train_coordinate_dataset, _ = get_data(batch_size, split_ratio, path=args.data)
    generator = LimbLengthGenerator().to(device)
    discriminator = LimbLengthDiscriminator().to(device)
    G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    if args.scratch:
        args.version = 1
        generator.apply(weights_init)
        discriminator.apply(weights_init)
    else:
        generator, discriminator, G_optimizer, D_optimizer = \
            load_model(generator, discriminator, G_optimizer, D_optimizer, args.version, args.path)

    for epoch in range(args.version, args.version + args.epochs + 1):

        D_loss_list, G_loss_list = [], []
        G_loss, D_total_loss = None, None

        for index, (real_coordinates, labels, limb_lengths) in enumerate(train_loader):
            D_optimizer.zero_grad()
            real_coordinates = real_coordinates.to(device).float()
            labels = labels.to(device)
            labels = labels.unsqueeze(1).long()
            # Calculate limb lengths
            real_target = Variable(torch.ones(real_coordinates.size(0), 1).to(device))
            fake_target = Variable(torch.zeros(real_coordinates.size(0), 1).to(device))

            D_real_loss = discriminator_loss(discriminator((real_coordinates, labels, limb_lengths)), real_target)

            noise_vector = generate_noise(real_coordinates.size(0), NOISE_DIMENSION, device=device)

            generated_coordinates = generator((noise_vector, labels, limb_lengths))

            output = discriminator((generated_coordinates.detach(), labels, limb_lengths))
            D_fake_loss = discriminator_loss(output, fake_target)

            D_total_loss = (D_real_loss + D_fake_loss) / 2
            D_loss_list.append(D_total_loss)

            D_total_loss.backward()
            D_optimizer.step()

            # Train generator with real labels
            G_optimizer.zero_grad()
            G_loss = generator_loss(discriminator((generated_coordinates, labels, limb_lengths)), real_target)
            G_loss_list.append(G_loss)

            G_loss.backward()
            G_optimizer.step()

        print_training_progress(epoch, G_loss, D_total_loss)
        if epoch % 52 == 0:
            save_models(generator, discriminator, G_optimizer, D_optimizer, epoch, args.path)

    save_models(generator, discriminator, G_optimizer, D_optimizer, args.epochs, args.path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
