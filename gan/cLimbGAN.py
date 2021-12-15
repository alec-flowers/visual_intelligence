import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data.data import get_data
from gan.cGAN import save_models, print_training_progress, get_device, generator_loss, \
    discriminator_loss, weights_init, generate_noise, load_model
from gan.gan_model import LimbLengthGenerator, LimbLengthDiscriminator
from pose.utils import GOOD_POSES_PATH, NOISE_DIMENSION, CLIMBGAN_PATH

adversarial_loss = nn.BCELoss()
TRAIN_ON_GPU = False


if __name__ == "__main__":
    # Model parameters
    batch_size = 64
    NUM_EPOCHS = 1000
    learning_rate = 0.0002
    shuffle = True
    split_ratio = 1
    save_generated_images = False
    TRAIN_ON_GPU = True
    PRINT_STATS_AFTER_BATCH = 39

    train_from_scratch = False
    continue_training = True
    version = 520
    device = get_device()

    num_examples_to_generate = 10
    LATENT_DIM = 50
    N_CLASSES = 3
    start = None

    # Set fixed random number seed
    torch.manual_seed(42)

    train_loader, _, train_coordinate_dataset, _ = get_data(batch_size, split_ratio, path=GOOD_POSES_PATH)
    generator = LimbLengthGenerator().to(device)
    discriminator = LimbLengthDiscriminator().to(device)
    G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    if train_from_scratch:
        version = 1
        generator.apply(weights_init)
        discriminator.apply(weights_init)
    elif continue_training:
        generator, discriminator, G_optimizer, D_optimizer = \
            load_model(generator, discriminator, G_optimizer, D_optimizer, version, CLIMBGAN_PATH)

    for epoch in range(version, NUM_EPOCHS + 1):

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
            save_models(generator, discriminator, G_optimizer, D_optimizer, epoch, CLIMBGAN_PATH)

    save_models(generator, discriminator, G_optimizer, D_optimizer, NUM_EPOCHS, CLIMBGAN_PATH)
    print(f'Finished!')
