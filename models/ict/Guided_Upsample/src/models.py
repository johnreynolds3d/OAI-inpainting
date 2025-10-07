import os

import torch
from torch import nn, optim

from .loss import AdversarialLoss, PerceptualLoss, StyleLoss
from .networks import Discriminator, Discriminator2, InpaintGenerator_5


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super().__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + "_gen.pth")
        self.dis_weights_path = os.path.join(config.PATH, name + "_dis.pth")

    def _strip_dataparallel_prefix(self, state_dict):
        """
        Remove 'module.' prefix from state dict keys.

        When a model is saved with DataParallel (multi-GPU training), PyTorch adds
        a 'module.' prefix to all parameter names. When loading on a single GPU,
        these prefixes cause key mismatch errors.

        This function detects if the state dict has DataParallel prefixes and
        strips them if present, ensuring compatibility between single and multi-GPU
        trained checkpoints.

        Args:
            state_dict: The loaded state dictionary from checkpoint

        Returns:
            state_dict: Fixed state dictionary with stripped prefixes if needed
        """
        # Check if any key has the 'module.' prefix
        has_module_prefix = any(key.startswith("module.") for key in state_dict)

        if has_module_prefix:
            print(
                "   ⚠️  Detected DataParallel checkpoint - stripping 'module.' prefixes..."
            )
            # Create new state dict with stripped keys
            new_state_dict = {}
            for key, value in state_dict.items():
                # Remove 'module.' prefix if present
                new_key = key[7:] if key.startswith("module.") else key
                new_state_dict[new_key] = value
            print(f"   ✅ Converted {len(state_dict)} keys for single-GPU loading")
            return new_state_dict

        return state_dict

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print(f"Loading {self.name} generator...")

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(
                    self.gen_weights_path, map_location=lambda storage, loc: storage
                )

            # Strip DataParallel prefixes if present (for multi-GPU trained models)
            generator_state = self._strip_dataparallel_prefix(data["generator"])
            self.generator.load_state_dict(generator_state)
            self.iteration = data["iteration"]

        # load discriminator only when training
        if (self.config.MODE == 1 or self.config.score) and os.path.exists(
            self.dis_weights_path
        ):
            print(f"Loading {self.name} discriminator...")

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(
                    self.dis_weights_path, map_location=lambda storage, loc: storage
                )

            # Strip DataParallel prefixes if present (for multi-GPU trained models)
            discriminator_state = self._strip_dataparallel_prefix(data["discriminator"])
            self.discriminator.load_state_dict(discriminator_state)

    def save(self):
        print(f"\nsaving {self.name}...\n")
        torch.save(
            {"iteration": self.iteration, "generator": self.generator.state_dict()},
            self.gen_weights_path,
        )

        torch.save(
            {"discriminator": self.discriminator.state_dict()}, self.dis_weights_path
        )


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super().__init__("InpaintingModel", config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]

        if config.Generator == 4:
            print("*******remove IN*******")
            generator = InpaintGenerator_5()
        else:
            # Default generator for other configurations
            generator = InpaintGenerator_5()

        if config.Discriminator == 0:
            discriminator = Discriminator(
                in_channels=3, use_sigmoid=config.GAN_LOSS != "hinge"
            )
        else:
            discriminator = Discriminator2(
                in_channels=3, use_sigmoid=config.GAN_LOSS != "hinge"
            )

        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module("generator", generator)
        self.add_module("discriminator", discriminator)

        self.add_module("l1_loss", l1_loss)
        self.add_module("perceptual_loss", perceptual_loss)
        self.add_module("style_loss", style_loss)
        self.add_module("adversarial_loss", adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2),
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2),
        )

    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        dis_loss.backward()
        self.dis_optimizer.step()

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = (
            self.adversarial_loss(gen_fake, True, False)
            * self.config.INPAINT_ADV_LOSS_WEIGHT
        )
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = (
            self.l1_loss(outputs, images)
            * self.config.L1_LOSS_WEIGHT
            / torch.mean(masks)
        )
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        gen_loss.backward()
        self.gen_optimizer.step()

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked, edges), dim=1)

        if self.config.Generator in {0, 2, 4}:
            outputs = self.generator(inputs)
        else:
            outputs = self.generator(inputs, masks)

        if self.config.score:
            gen_fake, _ = self.discriminator(outputs)
            gen_fake = gen_fake.view(8, -1)
            print(torch.mean(gen_fake, dim=1))

        return outputs
