import argparse
import os

import torch
import torch.nn as nn
import time
from datetime import datetime
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.gen = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.gen(z)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.dis = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.dis(img)
        return out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):
    # Print all configs to confirm parameter settings
    print_flags()

    # Define loss function
    criterion_G = nn.BCELoss()
    criterion_D = nn.BCELoss()
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            t1 = time.time()

            imgs.to(device)
            img_nr = imgs.shape[0]
            real_target = torch.FloatTensor(img_nr, 1).uniform_(0.7, 1.2).to(device)
            fake_target = torch.FloatTensor(img_nr, 1).uniform_(0., 0.3).to(device)
            
            noise = torch.randn(img_nr, args.latent_dim).to(device)
            gen_imgs = generator(noise).to(device)

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            flatten_imgs = imgs.view(img_nr, -1).to(device)
            pred_real = discriminator(flatten_imgs)
            loss_D_real = criterion_D(pred_real, real_target)
            pred_fake = discriminator(gen_imgs)
            loss_D_fake = criterion_D(pred_fake, fake_target)
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            pred = discriminator(gen_imgs)
            loss_G = criterion_G(pred, real_target)
            loss_G.backward()
            optimizer_G.step()

            t2 = time.time()

            # Print statements
            examples_per_second = img_nr/float(t2-t1)
            if i % 10 == 0:
                print("[{}] Train Epoch {}, Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                    "D_Loss = {:.2f}, G_Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), epoch, i,
                        len(dataloader), img_nr, examples_per_second,
                        loss_D, loss_G

                ))

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)
                save_image(gen_imgs.view(img_nr, 1, 28, 28)[:25],
                           './images/gan/{}.png'.format(batches_done),
                           nrow=5, normalize=True)


def print_flags():
  """
  Prints all entries in args variable.
  """
  for key, value in vars(args).items():
    print(key + ' : ' + str(value))

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    assert args.mode in ('train', 'eval')

    # Create output image directory
    os.makedirs('images/gan', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),(0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    if args.mode == 'train':
        train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device)

        # You can save your generator here to re-use it to generate images for your
        # report, e.g.:
        torch.save(generator.state_dict(), "./models/gan_mnist_generator.pt")

    else:
        step = 8
        generator.load_state_dict(torch.load('./models/gan_mnist_generator.pt'))
        noise = torch.randn(2, args.latent_dim).to(device)
        start = noise[0].unsqueeze(0)
        end = noise[1].unsqueeze(0)
        diff = end - start
        noise = start
        for i in range(1, step):
            tmp = start + (i / step) * diff
            noise = torch.cat((noise, tmp))
        noise = torch.cat((noise, end))
        gen_imgs = generator(noise)
        save_image(gen_imgs.view(gen_imgs.shape[0], 1, 28, 28),
                    './images/gan/interpolation.png',
                    nrow=gen_imgs.shape[0], normalize=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or eval')
    args = parser.parse_args()

    main()
