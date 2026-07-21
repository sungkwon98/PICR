import torch
import torch.nn as nn


class MLPBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        device: str,
        architecture_config: dict = None,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        
        base_shape = architecture_config["base_shape"]
        layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in base_shape:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.layers = nn.Sequential(*layers).to(self.device)
        self.layers.train()
        
    def forward(self, x_state_batch, x_action_batch):
        x = torch.cat([x_state_batch, x_action_batch], dim=-1).flatten(1, 2)
        x = self.layers(x)
        return x
    
    def reset(self):
        pass

    def reset_partial(self, batch_indices):
        pass


class MLPStateHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        device: str,
        architecture_config: dict = None,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.device = device
        self.state_mean_shape = architecture_config["state_mean_shape"]
        self.state_logstd_shape = architecture_config["state_logstd_shape"]

        state_mean_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.state_mean_shape:
            state_mean_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            state_mean_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        state_mean_layers.append(nn.Linear(self.state_mean_shape[-1], state_dim))
        self.state_mean_layers = nn.Sequential(*state_mean_layers).to(self.device)
        self.state_mean_layers.train()

        if self.state_logstd_shape is not None:
            self.output_std = True
            state_logstd_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in self.state_logstd_shape:
                state_logstd_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                state_logstd_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            state_logstd_layers.append(nn.Linear(self.state_logstd_shape[-1], state_dim))
            self.state_logstd_layers = nn.Sequential(*state_logstd_layers).to(self.device)
            self.state_logstd_layers.train()
        else:
            self.output_std = False

        if self.output_std:
            self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=self.device) * -5.0)
            self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=self.device) * 0.0)

    def forward(self, x, x_state_batch):
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
            x_state_batch = x_state_batch.flatten(0, 1).unsqueeze(1)
        else:
            sequence_len = 0
        state_mean = self.state_mean_layers(x) + x_state_batch[:, -1]
        state_logstd = self.state_logstd_layers(x) if self.output_std else -torch.inf * torch.ones(x.shape[0], self.state_dim, device=self.device)
        if self.output_std:
            self.state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
            state_logstd = self.state_max_logstd - nn.functional.softplus(self.state_max_logstd - state_logstd)
            state_logstd = self.state_min_logstd + nn.functional.softplus(state_logstd - self.state_min_logstd)
        if sequence_len > 0:
            state_mean = state_mean.view(-1, sequence_len, self.state_dim)
            state_logstd = state_logstd.view(-1, sequence_len, self.state_dim)
        return state_mean, torch.exp(state_logstd)

    def reset(self):
        pass
    
    def reset_partial(self, batch_indices):
        pass



class RNNBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        device: str,
        architecture_config: dict = None,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        
        rnn_type = architecture_config["rnn_type"]
        rnn_num_layers = architecture_config["rnn_num_layers"]
        rnn_hidden_size = architecture_config["rnn_hidden_size"]
        self.memory = Memory(input_dim, device, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        
    def forward(self, x_state_batch, x_action_batch):
        x = torch.cat([x_state_batch, x_action_batch], dim=-1)
        x = self.memory(x)
        return x
    
    def reset(self):
        self.memory.reset()
        
    def reset_partial(self, batch_indices):
        self.memory.reset_partial(batch_indices)


class Memory(nn.Module):
    def __init__(self, input_dim: int, device: str, type: str, num_layers: int, hidden_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=self.input_dim, hidden_size=hidden_size, num_layers=num_layers, device=self.device, batch_first=True)
        self.hidden_states = None

    def forward(self, x):
        x, self.hidden_states = self.rnn(x, self.hidden_states)
        return x[:, -1]
    
    def reset(self):
        self.hidden_states = None

    def reset_partial(self, batch_indices):
        if self.hidden_states is not None:
            self.hidden_states[:, batch_indices] = 0.0

import torch
import torch.nn as nn

class RWMEnsemble(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        ensemble_size: int = 1,
        history_horizon: int = 1,
        architecture_config: dict = None,
        prediction_indices=None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.ensemble_size = ensemble_size
        self.history_horizon = history_horizon
        self.architecture_config = architecture_config
        if prediction_indices is None:
            self.prediction_indices = None
            self.output_dim = int(state_dim)
        else:
            indices = torch.as_tensor(prediction_indices, dtype=torch.long, device=self.device)
            if indices.ndim != 1 or indices.numel() == 0:
                raise ValueError("prediction_indices must be a non-empty 1D index list.")
            if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(state_dim):
                raise ValueError(f"prediction_indices must be within [0, {state_dim}).")
            self.register_buffer("prediction_indices", indices, persistent=False)
            self.output_dim = int(indices.numel())
        self._init_networks()

    def _init_networks(self):
        self.state_base = self._create_base()
        self.state_heads = nn.ModuleList([
            MLPStateHead(
                self.base_output_dim,
                self.output_dim,
                self.device,
                self.architecture_config
            ).to(self.device) for _ in range(self.ensemble_size)
        ])
    def _create_base(self):
        if self.architecture_config["type"] == "mlp":
            input_dim = self.history_horizon * (self.state_dim + self.action_dim)
            self.base_output_dim = self.architecture_config["base_shape"][-1]
            self.prediction_type = "single"
            return MLPBase(
                input_dim=input_dim,
                device=self.device,
                architecture_config=self.architecture_config,
            )
        elif self.architecture_config["type"] == "rnn":
            input_dim = self.state_dim + self.action_dim
            self.base_output_dim = self.architecture_config["rnn_hidden_size"]
            self.prediction_type = "single"
            return RNNBase(
                input_dim=input_dim,
                device=self.device,
                architecture_config=self.architecture_config
            )
        else:
            raise ValueError("Invalid architecture type.")

    def forward(self, x_state_batch, x_action_batch, model_ids=None):
        state_means, state_stds = [], []
        state_base_output = self.state_base(x_state_batch, x_action_batch)
        residual_state_batch = self._prediction_state(x_state_batch)
        
        for head in self.state_heads:
            state_mean, state_std = head(state_base_output, residual_state_batch)
            if self.prediction_type == "sequence":
                state_mean = state_mean[:, -1]
                state_std = state_std[:, -1]
            state_means.append(state_mean.unsqueeze(0))
            state_stds.append(state_std.unsqueeze(0))

        state_means = torch.cat(state_means, dim=0)
        state_stds = torch.cat(state_stds, dim=0)
        
        if model_ids is None:
            output_state_means = state_means.mean(dim=0)
        else:
            output_state_means = torch.gather(state_means, 0, model_ids.repeat(1, 1, self.output_dim)).squeeze(0)
        
        aleatoric_uncertainty = state_stds.mean(dim=0).sum(dim=1)
        epistemic_uncertainty = state_means.std(dim=0).sum(dim=1) if self.ensemble_size > 1 else torch.zeros(output_state_means.shape[0], device=self.device)
        return output_state_means, aleatoric_uncertainty, epistemic_uncertainty

    def _prediction_state(self, state_batch):
        if self.prediction_indices is None:
            return state_batch
        return state_batch.index_select(-1, self.prediction_indices)

    def _prediction_target(self, state_target):
        if self.prediction_indices is None:
            return state_target
        return state_target.index_select(-1, self.prediction_indices)

    def _compose_full_state(self, predicted_state, base_state):
        if self.prediction_indices is None:
            return predicted_state
        full_state = base_state.clone()
        full_state.index_copy_(-1, self.prediction_indices, predicted_state)
        return full_state

    def compute_loss(self, state_batch, action_batch, bootstrap=False, state_loss_mask=None):
        state_losses = []
        sequence_losses = []
        bound_losses = []
        kl_losses = []
        
        for i in range(self.ensemble_size):
            self.reset()
            if bootstrap:
                ids = torch.randint(0, state_batch.shape[0], (state_batch.shape[0],), device=self.device)
            else:
                ids = torch.arange(0, state_batch.shape[0], device=self.device)
            
            state_loss, sequence_loss, bound_loss, kl_loss = self.compute_state_loss(
                self.state_heads[i],
                state_batch[ids],
                action_batch[ids],
                state_loss_mask=state_loss_mask,
            )
            
            state_losses.append(state_loss.unsqueeze(0))
            sequence_losses.append(sequence_loss.unsqueeze(0))
            bound_losses.append(bound_loss.unsqueeze(0))
            kl_losses.append(kl_loss.unsqueeze(0))
        
        self.reset()
        state_loss = torch.mean(torch.cat(state_losses, dim=0), dim=0)
        sequence_loss = torch.mean(torch.cat(sequence_losses, dim=0), dim=0)
        bound_loss = torch.mean(torch.cat(bound_losses, dim=0), dim=0)
        kl_loss = torch.mean(torch.cat(kl_losses, dim=0), dim=0)
        return state_loss, sequence_loss, bound_loss, kl_loss

    def compute_state_loss(self, head, state_batch, action_batch, state_loss_mask=None):
        forecast_horizon = state_batch.shape[1] - self.history_horizon
        x_state_batch = state_batch[:, :self.history_horizon]
        state_losses = []
        sequence_losses = []
        bound_losses = []
        kl_losses = []
        
        for i in range(forecast_horizon):
            if self.prediction_type == "single":
                state_target_full = state_batch[:, self.history_horizon + i]
                state_target = self._prediction_target(state_target_full)
            elif self.prediction_type == "sequence":
                state_target = state_batch[:, i + 1:self.history_horizon + i + 1]
                if self.prediction_indices is not None:
                    state_target = self._prediction_target(state_target)
            else:
                raise ValueError("Invalid state prediction type.")
            
            if self.architecture_config["type"] in ["rnn", "rssm"] and i > 0:
                x_action_batch = action_batch[:, self.history_horizon + i:self.history_horizon + i + 1]
                if self.prediction_type == "sequence":
                    state_target = state_target[:, [-1]]
            else:
                x_action_batch = action_batch[:, i + 1:self.history_horizon + i + 1]
            
            state_mean_pred, state_std_pred = head.forward(
                self.state_base.forward(x_state_batch, x_action_batch),
                self._prediction_state(x_state_batch),
            )
            state_loss, sequence_loss = self.compute_regression_loss(
                state_mean_pred,
                state_std_pred,
                state_target,
                state_loss_mask=state_loss_mask,
            )
            bound_loss = self.compute_bound_loss(head) if head.output_std else torch.tensor(0.0, device=self.device)
            kl_loss = self.state_base.kl_loss if self.architecture_config["type"] == "rssm" else torch.tensor(0.0, device=self.device)
            
            state_losses.append(state_loss.unsqueeze(0))
            sequence_losses.append(sequence_loss.unsqueeze(0))
            bound_losses.append(bound_loss.unsqueeze(0))
            kl_losses.append(kl_loss.unsqueeze(0))
            
            if self.prediction_type == "sequence":
                state_mean_pred = state_mean_pred[:, -1]
                state_std_pred = state_std_pred[:, -1]

            sampled_pred = (
                torch.randn_like(state_mean_pred, device=self.device) * state_std_pred + state_mean_pred
                if head.output_std
                else state_mean_pred
            )
            if self.prediction_indices is not None:
                if self.prediction_type != "single":
                    raise ValueError("prediction_indices currently supports prediction_type='single' only.")
                next_state = self._compose_full_state(sampled_pred, state_target_full)
            else:
                next_state = sampled_pred
            
            if self.architecture_config["type"] in ["rnn", "rssm"]:
                x_state_batch = next_state.unsqueeze(1)
            else:
                x_state_batch = torch.cat(
                    [
                        x_state_batch[:, 1:].clone(),
                        next_state.unsqueeze(1),
                    ],
                    dim=1
                )
        
        state_loss = torch.mean(torch.cat(state_losses, dim=0), dim=0)
        sequence_loss = torch.mean(torch.cat(sequence_losses, dim=0), dim=0)
        bound_loss = torch.mean(torch.cat(bound_losses, dim=0), dim=0)
        kl_loss = torch.mean(torch.cat(kl_losses, dim=0), dim=0)
        return state_loss, sequence_loss, bound_loss, kl_loss

    def compute_regression_loss(self, state_mean_pred, state_std_pred, state_target, loss_type="mse", state_loss_mask=None):
        mask = self._loss_mask(state_loss_mask, state_mean_pred)
        if loss_type == "mse":
            if self.prediction_type == "sequence":
                state_mean_pred_seq, state_mean_pred = state_mean_pred[:, :-1], state_mean_pred[:, -1]
                state_std_pred_seq, state_std_pred = state_std_pred[:, :-1], state_std_pred[:, -1]
                state_target_seq, state_target = state_target[:, :-1], state_target[:, -1]
                state_pred_seq = torch.randn_like(state_mean_pred_seq, device=self.device) * state_std_pred_seq + state_mean_pred_seq
                state_pred_seq = state_pred_seq.flatten(0, 1)
                state_target_seq = state_target_seq.flatten(0, 1)
                sequence_loss = self._masked_square_loss(state_pred_seq, state_target_seq, mask)
            else:
                sequence_loss = torch.tensor(0.0, device=self.device)
            state_pred = torch.randn_like(state_mean_pred, device=self.device) * state_std_pred + state_mean_pred
            state_loss = self._masked_square_loss(state_pred, state_target, mask)
            return state_loss, sequence_loss
        elif loss_type == "gaussian_nll":
            if self.prediction_type == "sequence":
                state_mean_pred_seq, state_mean_pred = state_mean_pred[:, :-1], state_mean_pred[:, -1]
                state_std_pred_seq, state_std_pred = state_std_pred[:, :-1], state_std_pred[:, -1]
                state_target_seq, state_target = state_target[:, :-1], state_target[:, -1]
                state_mean_pred_seq = state_mean_pred_seq.flatten(0, 1)
                state_std_pred_seq = state_std_pred_seq.flatten(0, 1)
                state_target_seq = state_target_seq.flatten(0, 1)
                sequence_loss = self._masked_gaussian_nll_loss(state_mean_pred_seq, state_target_seq, state_std_pred_seq, mask)
            else:
                sequence_loss = torch.tensor(0.0, device=self.device)
            state_loss = self._masked_gaussian_nll_loss(state_mean_pred, state_target, state_std_pred, mask)
            return state_loss, sequence_loss
        else:
            raise ValueError("Invalid loss type.")

    def _loss_mask(self, state_loss_mask, reference):
        if state_loss_mask is None:
            return None
        mask = torch.as_tensor(state_loss_mask, device=reference.device, dtype=reference.dtype)
        if self.prediction_indices is not None and mask.shape[-1] == self.state_dim:
            mask = mask.index_select(-1, self.prediction_indices)
        if mask.shape[-1] != self.output_dim:
            raise ValueError(f"Expected state_loss_mask dim {self.output_dim}, got {mask.shape[-1]}.")
        return mask.reshape(1, self.output_dim)

    def _masked_square_loss(self, pred, target, mask):
        diff = pred - target
        if mask is not None:
            diff = diff * mask
        return torch.sum(torch.square(diff), dim=-1).mean(dim=0)

    def _masked_gaussian_nll_loss(self, mean, target, std, mask):
        loss = nn.functional.gaussian_nll_loss(mean, target, std ** 2, reduction="none")
        if mask is not None:
            loss = loss * mask
        return torch.sum(loss, dim=-1).mean(dim=0)
        
    def compute_bound_loss(self, head):
        return torch.mean(head.state_max_logstd) - torch.mean(head.state_min_logstd)

    def reset(self):
        self.state_base.reset()
        for head in self.state_heads:
            head.reset()


    def reset_partial(self, batch_indices):
        self.state_base.reset_partial(batch_indices)
        for head in self.state_heads:
            head.reset_partial(batch_indices)
