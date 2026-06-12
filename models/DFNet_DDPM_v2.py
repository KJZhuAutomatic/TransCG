"""
Depth Filler Network.

Author: Hongjie Fang.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .dense import DenseBlock
from .duc import DenseUpsamplingConvolution
import math

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    返回所传递的值列表vals中的特定索引，同时考虑到批处理维度。
    """
    batch_size = t.shape[0]
    # out = vals.gather(-1, t.cpu())
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    接收一个图像和一个时间步长作为输入，并 返回它的噪声版本
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    #均值+方差
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

@torch.no_grad()#防止内存爆炸
def sample_timestep_(x, t, pred, betas, alphas_cumprod, alphas_cumprod_prev, posterior_variance, noise_off_th=1):
    """
    调用模型来预测图像中的噪声，并返回
    去噪后的图像。
    如果我们还没有进入最后一步，则对该图像施加噪声。
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    alphas_t = 1 - betas_t
    alphas_cumprod_t = get_index_from_list(
        alphas_cumprod, t, x.shape
    )
    alphas_cumprod_prev_t = get_index_from_list(
        alphas_cumprod_prev, t, x.shape
    )
    mean = (torch.sqrt(alphas_t) * (1 - alphas_cumprod_prev_t) * x + torch.sqrt(alphas_cumprod_prev_t) * betas_t * pred) / (1 - alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t[0] < noise_off_th:
        return mean
    else:
        noise = torch.randn_like(x)
        return mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()#防止内存爆炸
def sample_timestep(x, t, pred, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance):
    """
    调用模型来预测图像中的噪声，并返回
    去噪后的图像。
    如果我们还没有进入最后一步，则对该图像施加噪声。
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # 调用模型（当前图像--噪声预测）。
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * pred / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    return model_mean

    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()#防止内存爆炸
def ddim_sample_timestep(x, t, t_next, pred_noise, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod ):
    """
    调用模型来预测图像中的噪声，并返回
    去噪后的图像。
    如果我们还没有进入最后一步，则对该图像施加噪声。
    """

    alphas_cumprod_t = get_index_from_list(alphas_cumprod, t, x.shape)
    alphas_cumprod_t_next = get_index_from_list(alphas_cumprod, t_next, x.shape)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )

    # x_0
    x_0 = (x - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t

    sigma = ((1 - alphas_cumprod_t / alphas_cumprod_t_next) * (1 - alphas_cumprod_t_next) / (1 - alphas_cumprod_t)).sqrt()
    c = (1 - alphas_cumprod_t_next - sigma ** 2).sqrt()
    model_mean = x_0 * alphas_cumprod_t_next.sqrt() + c * pred_noise

    return model_mean

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class DFNet_DDPM_v2(nn.Module):
    """
    Depth Filler Network (DFNet).
    """
    def __init__(self, in_channels = 4, hidden_channels = 64, L = 5, k = 12, use_DUC = True, **kwargs):
        super(DFNet_DDPM_v2, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.L = L
        self.k = k
        self.use_DUC = use_DUC
        # First
        self.first = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense1: skip
        self.dense1s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense1s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense1s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense1: normal
        self.dense1_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense1 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense1_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense2: skip
        self.dense2s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense2s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense2s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense2: normal
        self.dense2_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense2 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense2_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense3: skip
        self.dense3s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense3s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense3s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense3: normal
        self.dense3_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense3 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense3_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense4
        self.dense4_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense4 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense4_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # DUC upsample 1
        self.updense1_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense1 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.updense1_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor = 2)

        # DUC upsample 2
        self.updense2_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense2 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.updense2_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor = 2)
        # DUC upsample 3
        self.updense3_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense3 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.updense3_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor = 2)
        # DUC upsample 4
        self.updense4_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense4 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.updense4_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor = 2)
        # Final
        self.final = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, 1, kernel_size = 1, stride = 1)
        )

        # diffusion parameter
        time_dim = 4
        self.time_embedding = SinusoidalPositionEmbeddings(dim=time_dim)
        timesteps = 1000
        sampling_timesteps = 200
        self.num_per_sample = 1
        self.objective = 'pred_x0'
        # betas = cosine_beta_schedule(timesteps)
        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = sampling_timesteps
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        # self.self_condition = False
        # self.scale = cfg.MODEL.DiffusionDet.SNR_SCALE
        # self.box_renewal = True
        # self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))


    def _make_upconv(self, in_channels, out_channels, upscale_factor = 2):
        if self.use_DUC:
            return DenseUpsamplingConvolution(in_channels, out_channels, upscale_factor = upscale_factor)
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size = upscale_factor, stride = upscale_factor, padding = 0, output_padding = 0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def forward(self, depth_gt, depth_render):
        batch_size, h, w = depth_render.shape
        device = depth_render.device
        dtype = depth_render.dtype
        if self.training:
            t = torch.randint(self.num_timesteps, size=(batch_size,)).to(device)
            noisy_depth = forward_diffusion_sample(depth_gt, t, device, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod)
            time_embed = self.time_embedding(t)
            time_embed = time_embed[:, :, None, None].repeat(1, 1, h, w)
            noisy_depth = noisy_depth.view(batch_size, 1, h, w)
            noise_input = torch.cat((noisy_depth, time_embed), dim = 1)
            pred_depth = self.forward_network(noise_input, depth_render)
        else:
            # assert depth_gt is not False
            total_timesteps, sampling_timesteps, eta = self.num_timesteps , self.sampling_timesteps, self.ddim_sampling_eta
            times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            noisy_depth = torch.randn(size=(batch_size, 1, h, w), device=device, dtype=dtype)
            for n, (time, time_next) in enumerate(time_pairs):
                t = torch.as_tensor(time, device=device, dtype=torch.int64).repeat(batch_size)
                time_embed = self.time_embedding(t)
                time_embed = time_embed[:, :, None, None].repeat(1, 1, h, w)
                noise_input = torch.cat((noisy_depth, time_embed), dim = 1)
                pred_depth = self.forward_network(noise_input, depth_render)
                pred_depth = pred_depth.view(batch_size, 1, h, w)
                pred_noise = self.predict_noise_from_start(noisy_depth, t, pred_depth)

                if time_next < 0:
                    noisy_depth = pred_depth
                else:
                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[time_next]

                    sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()

                    # noise = torch.randn_like(pred_poses_x0)

                    noisy_depth = pred_depth * alpha_next.sqrt() + c * pred_noise # + sigma * noise

        return pred_depth.view(batch_size, h, w)

    def forward_network(self, noisy_input, condi_depth):
        # 720 x 1280 (rgb, depth) -> 360 x 640 (h)
        n, h, w = condi_depth.shape
        condi_depth = condi_depth.view(n, 1, h, w)
        # noisy_depth = noisy_depth.view(n, 1, h, w)
        h = self.first(torch.cat((noisy_input, condi_depth), dim = 1))

        # dense1: 360 x 640 (h, depth1) -> 180 x 320 (h, depth2)
        depth1 = F.interpolate(condi_depth, scale_factor = 0.5, mode = "nearest")
        # dense1: skip
        h_d1s = self.dense1s_conv1(h)
        h_d1s = self.dense1s(torch.cat((h_d1s, depth1), dim = 1))
        h_d1s = self.dense1s_conv2(h_d1s)
        # dense1: normal
        h = self.dense1_conv1(h)
        h = self.dense1(torch.cat((h, depth1), dim = 1))
        h = self.dense1_conv2(h)

        # dense2: 180 x 320 (h, depth2) -> 90 x 160 (h, depth3)
        depth2 = F.interpolate(depth1, scale_factor = 0.5, mode = "nearest")
        # dense2: skip
        h_d2s = self.dense2s_conv1(h)
        h_d2s = self.dense2s(torch.cat((h_d2s, depth2), dim = 1))
        h_d2s = self.dense2s_conv2(h_d2s)
        # dense2: normal
        h = self.dense2_conv1(h)
        h = self.dense2(torch.cat((h, depth2), dim = 1))
        h = self.dense2_conv2(h)

        # dense3: 90 x 160 (h, depth3) -> 45 x 80 (h, depth4)
        depth3 = F.interpolate(depth2, scale_factor = 0.5, mode = "nearest")
        # dense3: skip
        h_d3s = self.dense3s_conv1(h)
        h_d3s = self.dense3s(torch.cat((h_d3s, depth3), dim = 1))
        h_d3s = self.dense3s_conv2(h_d3s)

        # dense3: normal
        h = self.dense3_conv1(h)
        h = self.dense3(torch.cat((h, depth3), dim = 1))
        h = self.dense3_conv2(h)

        # dense4: 45 x 80
        depth4 = F.interpolate(depth3, scale_factor = 0.5, mode = "nearest")
        h = self.dense4_conv1(h)
        h = self.dense4(torch.cat((h, depth4), dim = 1))
        h = self.dense4_conv2(h)

        # updense1: 45 x 80 -> 90 x 160
        h = self.updense1_conv(h)
        h = self.updense1(torch.cat((h, depth4), dim = 1))
        h = self.updense1_duc(h)

        # updense2: 90 x 160 -> 180 x 320
        h = torch.cat((h, h_d3s), dim = 1)
        h = self.updense2_conv(h)
        h = self.updense2(torch.cat((h, depth3), dim = 1))
        h = self.updense2_duc(h)

        # updense3: 180 x 320 -> 360 x 640
        h = torch.cat((h, h_d2s), dim = 1)
        h = self.updense3_conv(h)
        h = self.updense3(torch.cat((h, depth2), dim = 1))
        h = self.updense3_duc(h)

        # updense4: 360 x 640 -> 720 x 1280
        h = torch.cat((h, h_d1s), dim = 1)
        h = self.updense4_conv(h)
        h = self.updense4(torch.cat((h, depth1), dim = 1))
        h = self.updense4_duc(h)

        # final
        h = self.final(h)

        return rearrange(h, 'n 1 h w -> n h w')
