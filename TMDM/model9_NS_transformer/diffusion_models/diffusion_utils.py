import math
import torch
import numpy as np


def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)

    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)

    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2

    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)

    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start

    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008

        betas = torch.tensor(
            [
                min(
                    1
                    - (
                        math.cos(
                            ((i + 1) / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    )
                    / (
                        math.cos(
                            (i / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    ),
                    max_beta,
                )
                for i in range(num_timesteps)
            ]
        )

        if schedule == "cosine_reverse":
            betas = betas.flip(0)

    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [
                start
                + 0.5
                * (end - start)
                * (1 - math.cos(t / (num_timesteps - 1) * math.pi))
                for t in range(num_timesteps)
            ]
        )

    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")

    return betas


def extract(input_tensor, t, x):
    """
    Extract values from a 1-D tensor according to timestep t, then reshape
    for broadcasting to x.

    input_tensor: [T]
    t:            [B]
    x:            target tensor, e.g. [B, L, C]
    """
    shape = x.shape

    t = t.to(input_tensor.device).long()

    out = torch.gather(input_tensor, 0, t)

    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def q_sample_residual(
    r0,
    r_prior,
    alphas_bar_sqrt,
    one_minus_alphas_bar_sqrt,
    t,
    noise=None,
):
    """
    q(r_t | r_0, r_prior)

    r_t =
        sqrt(alpha_bar_t) * r0
        + (1 - sqrt(alpha_bar_t)) * r_prior
        + sqrt(1 - alpha_bar_t) * noise

    In V1:
        r_prior = 0
    """
    if noise is None:
        noise = torch.randn_like(r0).to(r0.device)

    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, r0)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, r0)

    r_t = (
        sqrt_alpha_bar_t * r0
        + (1 - sqrt_alpha_bar_t) * r_prior
        + sqrt_one_minus_alpha_bar_t * noise
    )

    return r_t


def _get_raw_model(model):
    """
    Support both normal model and DataParallel model.
    """
    return model.module if hasattr(model, "module") else model


def _get_sample_temperature(model):
    """
    Read sampling temperature from model.args.
    Default = 1.0, which recovers the original sampling behavior.
    """
    raw_model = _get_raw_model(model)
    return float(getattr(raw_model.args, "sample_temperature", 1.0))


def _make_sampling_timestep(t, batch_size, device):
    """
    During sampling, t is usually an int.
    Patch Transformer denoiser needs t shape [B], not [1].
    """
    if isinstance(t, int):
        return torch.full(
            (batch_size,),
            t,
            device=device,
            dtype=torch.long,
        )

    t_tensor = t.to(device).long()

    if t_tensor.dim() == 0:
        t_tensor = t_tensor.repeat(batch_size)

    if t_tensor.numel() == 1:
        t_tensor = t_tensor.repeat(batch_size)

    return t_tensor


def p_sample_residual(
    model,
    x,
    x_mark,
    y_base,
    r_t,
    r_prior,
    t,
    alphas,
    one_minus_alphas_bar_sqrt,
):
    """
    One reverse step for residual diffusion.

    V1 compatible model call:
        eps_theta = model(x, x_mark, y_base, r_t, r_prior, t_tensor)

    The model predicts eps_theta.

    sample_temperature:
        temperature > 1.0 makes residual samples more diverse.
        temperature < 1.0 makes residual samples more concentrated.
    """
    raw_model = _get_raw_model(model)
    device = next(raw_model.parameters()).device
    temperature = _get_sample_temperature(model)

    # Temperature-scaled reverse noise.
    z = temperature * torch.randn_like(r_t).to(device)

    t_tensor = _make_sampling_timestep(
        t=t,
        batch_size=r_t.shape[0],
        device=device,
    )

    alpha_t = extract(alphas, t_tensor, r_t)

    sqrt_one_minus_alpha_bar_t = extract(
        one_minus_alphas_bar_sqrt,
        t_tensor,
        r_t,
    )

    t_prev = torch.clamp(t_tensor - 1, min=0)

    sqrt_one_minus_alpha_bar_t_m_1 = extract(
        one_minus_alphas_bar_sqrt,
        t_prev,
        r_t,
    )

    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()

    denom = sqrt_one_minus_alpha_bar_t.square() + 1e-8

    gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / denom

    gamma_1 = (
        sqrt_one_minus_alpha_bar_t_m_1.square()
        * alpha_t.sqrt()
        / denom
    )

    gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (
        alpha_t.sqrt() + sqrt_alpha_bar_t_m_1
    ) / denom

    eps_theta = model(
        x,
        x_mark,
        y_base,
        r_t,
        r_prior,
        t_tensor,
    ).to(device).detach()

    r0_reparam = 1 / (sqrt_alpha_bar_t + 1e-8) * (
        r_t
        - (1 - sqrt_alpha_bar_t) * r_prior
        - eps_theta * sqrt_one_minus_alpha_bar_t
    )

    r_t_m_1_hat = gamma_0 * r0_reparam + gamma_1 * r_t + gamma_2 * r_prior

    beta_t_hat = (
        sqrt_one_minus_alpha_bar_t_m_1.square()
        / denom
        * (1 - alpha_t)
    )

    r_t_m_1 = r_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z

    return r_t_m_1


def p_sample_residual_t_1to0(
    model,
    x,
    x_mark,
    y_base,
    r_t,
    r_prior,
    one_minus_alphas_bar_sqrt,
):
    """
    Final reverse step from t=1 to t=0.

    V1 compatible:
        model(x, x_mark, y_base, r_t, r_prior, t_tensor)

    Note:
        No extra sampling noise is added at the final t=0 reconstruction step.
    """
    raw_model = _get_raw_model(model)
    device = next(raw_model.parameters()).device

    t_tensor = torch.zeros(
        r_t.shape[0],
        device=device,
        dtype=torch.long,
    )

    sqrt_one_minus_alpha_bar_t = extract(
        one_minus_alphas_bar_sqrt,
        t_tensor,
        r_t,
    )

    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()

    eps_theta = model(
        x,
        x_mark,
        y_base,
        r_t,
        r_prior,
        t_tensor,
    ).to(device).detach()

    r0_reparam = 1 / (sqrt_alpha_bar_t + 1e-8) * (
        r_t
        - (1 - sqrt_alpha_bar_t) * r_prior
        - eps_theta * sqrt_one_minus_alpha_bar_t
    )

    return r0_reparam.to(device)


def p_sample_loop_residual(
    model,
    x,
    x_mark,
    y_base,
    r_prior,
    n_steps,
    alphas,
    one_minus_alphas_bar_sqrt,
    return_sequence=False,
):
    """
    Sample from p(r_0 | x, y_base).

    V1:
        r_prior = 0
        Residual Patch Transformer
        noise loss

    sample_temperature:
        Applied to:
            1. initial residual noise r_T
            2. reverse step noise z

    Memory-saving behavior:
        return_sequence=False returns [r0] only.
        This avoids storing all reverse states during testing.

    Existing caller remains valid:
        r_tile_seq = p_sample_loop_residual(...)
        gen_r = r_tile_seq[-1]
    """
    raw_model = _get_raw_model(model)
    device = next(raw_model.parameters()).device
    temperature = _get_sample_temperature(model)

    # Temperature-scaled initial noise.
    z = temperature * torch.randn_like(r_prior).to(device)

    # r_T ~ N(r_prior, temperature^2 I)
    cur_r = z + r_prior

    if return_sequence:
        r_seq = [cur_r]
    else:
        r_seq = None

    for t in reversed(range(1, n_steps)):
        cur_r = p_sample_residual(
            model=model,
            x=x,
            x_mark=x_mark,
            y_base=y_base,
            r_t=cur_r,
            r_prior=r_prior,
            t=t,
            alphas=alphas,
            one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt,
        )

        if return_sequence:
            r_seq.append(cur_r)

    r0 = p_sample_residual_t_1to0(
        model=model,
        x=x,
        x_mark=x_mark,
        y_base=y_base,
        r_t=cur_r,
        r_prior=r_prior,
        one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt,
    )

    if return_sequence:
        r_seq.append(r0)
        return r_seq

    return [r0]


def kld(y1, y2, grid=(-20, 20), num_grid=400):
    y1, y2 = y1.numpy().flatten(), y2.numpy().flatten()

    p_y1, _ = np.histogram(
        y1,
        bins=num_grid,
        range=[grid[0], grid[1]],
        density=True,
    )
    p_y1 += 1e-7

    p_y2, _ = np.histogram(
        y2,
        bins=num_grid,
        range=[grid[0], grid[1]],
        density=True,
    )
    p_y2 += 1e-7

    return (p_y1 * np.log(p_y1 / p_y2)).sum()
