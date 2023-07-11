import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SlotAttention(nn.Module):
    def __init__(self, num_iter, num_slots, input_size, slot_size, mlp_hidden_size, epsilon=1e-8, simple=False,
                 project_inputs=False, gain=1, temperature_factor=1):
        super().__init__()
        self.temperature_factor = temperature_factor
        self.num_iter = num_iter
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.input_size = input_size

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, 1, self.slot_size)))
        self.slots_log_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, 1, self.slot_size)))

        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(input_size, slot_size, bias=False)
        self.project_v = nn.Linear(input_size, slot_size, bias=False)

        nn.init.xavier_uniform_(self.project_q.weight, gain=gain)
        nn.init.xavier_uniform_(self.project_k.weight, gain=gain)
        nn.init.xavier_uniform_(self.project_v.weight, gain=gain)

        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_hidden_size, self.slot_size)
        )

        self.simple = simple
        if self.simple:
            assert slot_size == input_size
            self.norm_mlp = nn.Identity()
            del self.gru;
            self.gru = lambda x, h, alpha=0.5: h * alpha + x * (1 - alpha)
            del self.mlp;
            self.mlp = torch.zeros_like
        self.project_x = nn.Linear(input_size, input_size) if project_inputs else nn.Identity()

    def forward(self, inputs: 'BTC', num_iter=0, slots: 'BSC' = None) -> '(BSC, BST, BST)':
        inputs = self.project_x(inputs)

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        if slots is None:
            slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(len(inputs), self.num_slots,
                                                                                  self.slot_size,
                                                                                  device=self.slots_mu.device)

        for _ in range(num_iter or self.num_iter):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)
            q *= self.slot_size ** -0.5

            attn_logits = torch.bmm(q, k.transpose(-1, -2))

            attn_pixelwise = F.softmax(attn_logits / self.temperature_factor, dim=1)
            attn_slotwise = F.normalize(attn_pixelwise + self.epsilon, p=1, dim=-1)

            updates = torch.bmm(attn_slotwise, v)

            slots = self.gru(updates.flatten(end_dim=1), slots_prev.flatten(end_dim=1)).reshape_as(slots)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_logits, attn_slotwise


def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)


"""Adds soft positional embedding with learnable projection."""


class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, padding=2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        #print(x.shape)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2).to(device)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        #         x = F.pad(x, (4,4,4,4)) # no longer needed
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:, :, :self.resolution[0], :self.resolution[1]]
        x = x.permute(0, 2, 3, 1)
        return x


"""Slot Attention-based auto-encoder for object discovery."""


class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution=(128,128), num_slots=8, num_iterations=5, hid_dim=14):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters=self.num_iterations,
            eps=1e-8,
            hidden_dim=128)

    def forward(self, image):
        #print(image.shape)
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        #print(slots.shape)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))

        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3, 1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        return recon_combined, recons, masks, slots


if __name__=="__main__":
    model=SlotAttentionAutoEncoder().cuda()
    input=torch.randn((2,3,128,128)).cuda()
    #summary(model,(3,256,256))
    recon_combined, recons, masks, slots=model(input)
    print(recon_combined.shape)