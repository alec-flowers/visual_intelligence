import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data import get_data
from model import Generator, Discriminator
from utils import GOOD_POSES_PATH

adversarial_loss = nn.BCELoss()
TRAIN_ON_GPU = False


def save_models(generator, discriminator, G_optimizer, D_optimizer, epoch):
    """ Save models at specific point in time. """
    torch.save(generator.state_dict(), "./saved_model/cGAN" + f'/generator_{epoch}.pth')
    torch.save(discriminator.state_dict(), "./saved_model/cGAN" + f'/discriminator_{epoch}.pth')
    torch.save(G_optimizer.state_dict(), "./saved_model/cGAN" + f'/G_optimizer_{epoch}.pth')
    torch.save(D_optimizer.state_dict(), "./saved_model/cGAN" + f'/D_optimizer_{epoch}.pth')


def print_training_progress(epoch, generator_loss, discriminator_loss):
    """ Print training progress. """
    print('Losses after epoch %4d: generator %.3f, discriminator %.3f' %
          (epoch, generator_loss.item(), discriminator_loss.item()))


def get_device():
    """ Retrieve device based on settings and availability. """
    return torch.device("cuda:0" if torch.cuda.is_available() and TRAIN_ON_GPU else "cpu")


def generator_loss(fake_output, label):
    gen_loss = adversarial_loss(fake_output, label)
    return gen_loss


def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss


# custom weights initialization called on gen and disc model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def generate_noise(number_of_images=1, noise_dimension=100, device=None):
    """ Generate noise for number_of_images images, with a specific noise_dimension """
    return torch.randn(number_of_images, noise_dimension, device=device)


def load_model(generator, discriminator, G_optimizer, D_optimizer, version):
    generator.load_state_dict(torch.load("./saved_model/cGAN" + f'/generator_{version}.pth'))
    discriminator.load_state_dict(torch.load("./saved_model/cGAN" + f'/discriminator_{version}.pth'))
    G_optimizer.load_state_dict(torch.load("./saved_model/cGAN" + f'/G_optimizer_{version}.pth'))
    D_optimizer.load_state_dict(torch.load("./saved_model/cGAN" + f'/D_optimizer_{version}.pth'))
    return generator, discriminator, G_optimizer, D_optimizer


if __name__ == "__main__":
    # Model parameters
    batch_size = 64
    NUM_EPOCHS = 1000
    learning_rate = 0.0002
    shuffle = True
    split_ratio = 1
    save_generated_images = False
    TRAIN_ON_GPU = False
    PRINT_STATS_AFTER_BATCH = 39

    train_from_scratch = False
    continue_training = True
    version = 572
    device = get_device()

    num_examples_to_generate = 10
    LATENT_DIM = 100
    N_CLASSES = 3
    start = None

    # Set fixed random number seed
    torch.manual_seed(42)

    train_loader, _, train_coordinate_dataset, _ = get_data(batch_size, split_ratio, path=GOOD_POSES_PATH)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    if train_from_scratch:
        start = 1
        generator.apply(weights_init)
        discriminator.apply(weights_init)
    elif continue_training:
        start = version
        generator, discriminator, G_optimizer, D_optimizer = \
            load_model(generator, discriminator, G_optimizer, D_optimizer, version)

    for epoch in range(start, NUM_EPOCHS + 1):

        D_loss_list, G_loss_list = [], []
        G_loss, D_total_loss = None, None

        for index, (real_images, labels) in enumerate(train_loader):
            D_optimizer.zero_grad()
            real_images = real_images.to(device).float()
            labels = labels.to(device)
            labels = labels.unsqueeze(1).long()

            real_target = Variable(torch.ones(real_images.size(0), 1).to(device))
            fake_target = Variable(torch.zeros(real_images.size(0), 1).to(device))

            D_real_loss = discriminator_loss(discriminator((real_images, labels)), real_target)
            # D_real_loss.backward()

            noise_vector = generate_noise(real_images.size(0), LATENT_DIM, device=device)

            generated_image = generator((noise_vector, labels))

            output = discriminator((generated_image.detach(), labels))
            D_fake_loss = discriminator_loss(output, fake_target)

            # train with fake
            # D_fake_loss.backward()

            D_total_loss = (D_real_loss + D_fake_loss) / 2
            D_loss_list.append(D_total_loss)

            D_total_loss.backward()
            D_optimizer.step()

            # Train generator with real labels
            G_optimizer.zero_grad()
            G_loss = generator_loss(discriminator((generated_image, labels)), real_target)
            G_loss_list.append(G_loss)

            G_loss.backward()
            G_optimizer.step()

        print_training_progress(epoch, G_loss, D_total_loss)
        if epoch % 52 == 0:
            save_models(generator, discriminator, G_optimizer, D_optimizer, epoch)

    save_models(generator, discriminator, G_optimizer, D_optimizer, NUM_EPOCHS)
    print(f'Finished!')
