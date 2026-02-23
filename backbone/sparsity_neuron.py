import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import surrogate

import tool

class TensorNormalization(nn.Module):
    def __init__(self, mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std

    def forward(self, X):
        return normalizex(X, self.mean, self.std)


def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x



class APLayer(nn.Module):
    def __init__(self,kernel_size):
        super(APLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.AvgPool2d(kernel_size),
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class ZO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, delta):
        out = (input > 0).float()
        ctx.save_for_backward(input, torch.tensor([delta], device=input.device))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, delta_tensor = ctx.saved_tensors
        delta = delta_tensor.item()
        grad_input = grad_output.clone()

        sample_size = 5
        abs_z = torch.abs(torch.randn((sample_size,) + input.size(),
                                      device=input.device, dtype=input.dtype))
        mask = torch.abs(input[None, :, :]) < abs_z * delta
        grad_input = grad_input * torch.mean(mask * abs_z, dim=0) / (2 * delta)
        return grad_input, None


class LIFSpike(nn.Module):
    def __init__(self, tau=0.5, delta=0.5, init_thresh=1.0, adaptive_ratio=0.5):
        super().__init__()

        self.tau = tau
        self.delta = delta
        self.init_thresh = init_thresh
        self.adaptive_ratio = adaptive_ratio

        self.thresh = None
        self.base_rate = None
        self.adaptive_mask = None

        self.total_sessions = 0
        self.rate_history = []
        self.thresh_history = []
        self.avg_adaptive_sessions = []
        self.avg_nonadaptive_sessions = []

    def forward(self, x):

        session = tool.session
        args = tool.args
        self.total_sessions = args.sessions


        while len(self.avg_adaptive_sessions) <= args.sessions:
            self.avg_adaptive_sessions.append([])
            self.avg_nonadaptive_sessions.append([])


        B, T, C, H, W = x.size()

        if T != args.time_step and args.dataset not in ['dvs128gesture']:
            T, B, C, H, W = x.size()

        mem = torch.zeros_like(x[:, 0])
        spikes = []


        if self.thresh is None:
            self.thresh = nn.Parameter(
                torch.full((1, C, 1, 1), self.init_thresh, device=x.device)
            )

            num_adaptive = int(C * self.adaptive_ratio)
            mask = torch.zeros(C, dtype=torch.bool)
            perm = torch.randperm(C)
            mask[perm[:num_adaptive]] = True
            self.adaptive_mask = mask.to(x.device)


        for t in range(T):

            mem = mem * self.tau + x[:, t]

            if args.sg == 'zoo':
                spike = ZO.apply(mem - self.thresh, self.delta)
            elif args.sg == 'zif':
                spike = ZIF.apply(mem - self.thresh, self.delta)
            elif args.sg == 'atan':
                spike = surrogate.ATan()(mem - self.thresh)
            elif args.sg == 'sigmoid':
                spike = surrogate.Sigmoid()(mem - self.thresh)
            elif args.sg == 'piecewise_exp':
                spike = surrogate.PiecewiseExp()(mem - self.thresh)
            elif args.sg == 'piecewise_leaky_relu':
                spike = surrogate.PiecewiseLeakyReLU()(mem - self.thresh)
            elif args.sg == 'qp_seudo_spike':
                spike = surrogate.QPseudoSpike()(mem - self.thresh)
            else:
                raise ValueError(f"Unknown surrogate function: {args.sg}")

            mem = mem * (1 - spike)
            spikes.append(spike)

        out = torch.stack(spikes, dim=1)


        with torch.no_grad():

            current_rate = out.float().mean(dim=(0, 1, 3, 4))  # [C]

            if session == 0:
                self.base_rate = current_rate.detach().clone()

            elif session > 0 and self.base_rate is not None:

                decay_factor = math.exp(-session / args.tau_decay)

                aligned_thresh = self.thresh.data.view(-1)

                # non-adaptive
                aligned_thresh[~self.adaptive_mask] += (
                    decay_factor * args.beta *
                    (current_rate[~self.adaptive_mask] -
                     self.base_rate[~self.adaptive_mask])
                )

                # adaptive
                aligned_thresh[self.adaptive_mask] += (
                    decay_factor * args.theta *
                    (current_rate[self.adaptive_mask] -
                     self.base_rate[self.adaptive_mask])
                )

                self.thresh.data = aligned_thresh.view_as(self.thresh)

        return out



    def plot_avg_spike_rate_per_time(self, out, args):

        B, T, C, H, W = out.shape
        avg_spike_per_time = out.float().mean(dim=[2, 3, 4])  # shape: [B, T]

        width = max(12, T * 0.3)
        height = 6 if B <= 8 else 8
        plt.figure(figsize=(width, height))

        for b in range(B):
            y = avg_spike_per_time[b].detach().cpu().numpy()
            if y.max() > 1e-4:
                plt.plot(y, label=f'Sample {b}')

        plt.xlabel("Time Step")
        plt.ylabel("Average Spike Rate (all channels)")
        plt.title("Average Spike Rate over Time (per Sample)")
        plt.grid(True)


        if B <= 10:
            plt.legend(loc='upper right')
        else:
            plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')


        os.makedirs(args.save_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(args.save_path, f"{timestamp}_apaptive_spike_rate.png")

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Spike rate figure saved to {filename}")

    def plot_adaptive_vs_nonadaptive_spike_rate(self, out, adaptive_mask, args=None, session=0):

        B, T, C, H, W = out.shape
        out_float = out.float()
        if adaptive_mask.sum() == 0 or (~adaptive_mask).sum() == 0:
            print(f"[Warning] Adaptive or Non-Adaptive neuron count is zero. Skip plotting.")
            return

        # 1. 平均脉冲率，按 batch + space 平均，保留时间和通道维度: [T, C]
        mean_over_batch_space = out_float.mean(dim=(0, 3, 4))  # [T, C]

        # 2. 分别获取 adaptive / non-adaptive 的每时刻平均放电率
        adaptive_mask = adaptive_mask.to(out.device)
        adaptive_rate = mean_over_batch_space[:, adaptive_mask]  # [T, C_adaptive]
        nonadaptive_rate = mean_over_batch_space[:, ~adaptive_mask]  # [T, C_nonadaptive]

        # 3. 对通道再平均，得到每个时间步的平均放电率 [T]
        avg_adaptive = adaptive_rate.mean(dim=1).detach().cpu().numpy()
        avg_nonadaptive = nonadaptive_rate.mean(dim=1).detach().cpu().numpy()

        # 添加数据
        self.avg_adaptive_sessions[session].append(avg_adaptive)
        self.avg_nonadaptive_sessions[session].append(avg_nonadaptive)

        # 只有最后一个 session 时才画图
        if session != self.total_sessions - 1:
            return

        # 4. 画图
        plt.figure(figsize=(10, 5))
        plt.plot(avg_adaptive, label="Adaptive Neurons", color='tab:orange')
        plt.plot(avg_nonadaptive, label="Non-Adaptive Neurons", color='tab:blue')

        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel("Average Spike Rate", fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # 5. 保存图像
        os.makedirs(args.save_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(args.save_path, f"{timestamp}_spike_rate_{session}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[Saved] Spike rate comparison saved to {filename}")


    def plot_all_session_averages(self):

        num_sessions = len(self.avg_adaptive_sessions)
        color_map = plt.get_cmap('tab10')

        plt.figure(figsize=(12, 6))

        legend_handles = []
        legend_labels = []

        for session_id in range(num_sessions):
            a_data = self.avg_adaptive_sessions[session_id]
            n_data = self.avg_nonadaptive_sessions[session_id]

            if len(a_data) == 0 or len(n_data) == 0:
                continue

            # [N, T] → [T]
            a_mean = np.mean(np.stack(a_data), axis=0)
            n_mean = np.mean(np.stack(n_data), axis=0)
            T = len(a_mean)

            color = color_map(session_id % 10)

            # adaptive
            line_a, = plt.plot(range(T), a_mean, linestyle='-', color=color, linewidth=2.0,
                               label=f'Session {session_id} (adaptive)' if session_id < 5 else None)
            # non-adaptive
            line_n, = plt.plot(range(T), n_mean, linestyle='--', color=color, linewidth=2.0,
                               label=f'Session {session_id} (non-adaptive)' if session_id < 5 else None)

            if session_id < 5:
                legend_handles.extend([line_a, line_n])
                legend_labels.extend([line_a.get_label(), line_n.get_label()])

        plt.xlabel('Time Step T', fontsize=16, fontweight='bold')
        plt.ylabel('Average Firing Rate', fontsize=16, fontweight='bold')
        plt.title('Adaptive vs Non-Adaptive Neuron Firing Rate per Session',
                  fontsize=18, fontweight='bold')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        if legend_handles:
            plt.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=13,
                title='Legend',
                title_fontsize=14
            )

        save_dir = "plots"
        full_path = os.path.join(self.args.save_path, save_dir)
        os.makedirs(full_path, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(full_path, f"{timestamp}_firing_rate_all_sessions.png")

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to {save_path}")

    def save_stats(self):
        import numpy as np
        np.save(os.path.join(self.save_dir, "firing_rate.npy"), np.stack(self.rate_history))
        np.save(os.path.join(self.save_dir, "thresh.npy"), np.stack(self.thresh_history))

    def plot_stats(self):
        import numpy as np
        rate = np.load(os.path.join(self.save_dir, "firing_rate.npy"))  # [session, C]
        thresh = np.load(os.path.join(self.save_dir, "thresh.npy"))  # [session, C]

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(rate)
        plt.title("Firing Rate per Channel")
        plt.xlabel("Session")
        plt.ylabel("Firing Rate")

        plt.subplot(1, 2, 2)
        plt.plot(thresh)
        plt.title("Threshold per Channel")
        plt.xlabel("Session")
        plt.ylabel("Threshold")
        plt.tight_layout()
        plt.savefig(os.path.join("logs", "lif_dynamics.png"))
        plt.close()



def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x



class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y

