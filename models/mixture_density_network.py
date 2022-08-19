import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np


class MDN(nn.Module):
    """

    """
    def __init__(self, cfg, device):
        super(MDN, self).__init__()
        self.device = device
        dec_cfg = cfg['model']['decoder']['params']
        self.num_gaussians = dec_cfg['num_gaussians']
        self.output_dims = dec_cfg['output_dims']
        enc_feat_len = cfg['model']['encoder']['params']['enc_feat_len']

        self.pi = nn.Linear(in_features=enc_feat_len, out_features=self.num_gaussians)
        self.mu = nn.Linear(in_features=enc_feat_len, out_features=3*self.num_gaussians)
        self.u_diag = nn.Linear(in_features=enc_feat_len, out_features=3*self.num_gaussians)
        self.u_upper = nn.Linear(in_features=enc_feat_len, out_features=3*self.num_gaussians)

    def forward(self, X):
        """
        X: (batch_size, enc_feat_len) feature vector
        """
        # Apply softmax to pi to ensure weights sum to 1
        pi = self.pi(X) # (B, num_gaussians)
        pi = F.softmax(pi - pi.max(), dim=-1)

        # Reshape mu to (B, num_gaussians, 3)
        mu = self.mu(X).view(X.shape[0], self.num_gaussians, -1)

        # Apply modified ELU to ensure diagonals are positive
        u_diag = nn.ELU()(self.u_diag(X)) + 1 + 1e-15 # (B, 3*num_gaussians)
        u_diag = u_diag.view(X.shape[0], self.num_gaussians, -1) # (B, num_gaussians, 3)
        u_upper = self.u_upper(X).view(X.shape[0], self.num_gaussians, -1) # (B, num_gaussians, 3)

        # Build upper-triangular Cholesky factor from predictions
        U = torch.diag_embed(u_diag) # (B, num_gaussians, 3, 3)

        # Returns the indices of the upper triangular elements in the matrix
        upper_idx = torch.triu_indices(3, 3, offset=1)

        # Use indices to set upper-triangular elements
        U[:, :, upper_idx[0], upper_idx[1]] = u_upper

        return pi, mu, U

    @staticmethod
    def nll_loss(pi, mu, U, y):
        """
        pi: (B, num_gaussians)
        mu: (B, num_gaussians, 3)
        U: (B, num_gaussians, 3, 3)
        y: (B, 3)

        return: (B,)
        """
        u_diag_sum = U.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        y_diff_vec = (y.unsqueeze(-1).expand(-1, -1, 3) - mu).unsqueeze(-1)
        vec_norm = torch.linalg.vector_norm(torch.matmul(U, y_diff_vec).squeeze(-1), dim=-1)
        return -(pi * (u_diag_sum - 0.5 * vec_norm**2).exp()).sum(-1).log()

    @staticmethod
    def sample(pi, mu, U):
        """
        pi: (B, num_gaussians)
        mu: (B, num_gaussians, 3)
        U: (B, num_gaussians, 3, 3)

        samples: (B, 3)
        """
        # First sample the mixture component index based on the weights
        comp_idx = torch.multinomial(pi, num_samples=1) # (B, 1)

        # Expand idx to (B, 1, 3) and sample each mu vec
        comp_idx_mu = comp_idx.unsqueeze(-1).expand(-1, -1, 3)
        comp_mu = mu.gather(1, comp_idx_mu).squeeze(1)
        # print(comp_mu.shape) (B, 3)

        # Expand idx to (B, 1, 3, 3) and sample each U matrix
        comp_idx_U = comp_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3)
        comp_U = U.gather(1, comp_idx_U).squeeze(1)
        #print(comp_U.shape) # (B, 3, 3)

        # Sample from each normal distribution in the batch
        inverse_covariance = torch.matmul(comp_U.transpose(1, 2), comp_U)
        m = MultivariateNormal(comp_mu, precision_matrix=inverse_covariance)
        samples = m.sample()

        return samples

if __name__=="__main__":
    # Testing
    import yaml
    cfg_path = "/home/martin/projects/end-to-end-driving/configs/regnet_seq_model_mdn.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print(cfg)

    mdn = MDN(cfg, "cpu")
    batch_size = 32
    feat_len = 256
    X = torch.randn((batch_size, feat_len))
    pi, mu, U = mdn(X)
    print(pi.shape)
    print(mu.shape)
    print(U.shape)
    y = torch.randn((batch_size, 3))
    loss = MDN.nll_loss(pi, mu, U, y)
    sample = MDN.sample(pi, mu, U)
    print(sample.shape)


"""
From MDN tutorial
"""
# ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
# LOG2PI = math.log(2 * math.pi)

# class MixtureDensityHead(nn.Module):
#     def __init__(self, config: DictConfig, **kwargs):
#         self.hparams = config
#         super().__init__()
#         self._build_network()

#     def _build_network(self):
#         self.pi = nn.Linear(self.hparams.input_dim, self.hparams.num_gaussian)
#         nn.init.normal_(self.pi.weight)
#         self.sigma = nn.Linear(
#             self.hparams.input_dim,
#             self.hparams.num_gaussian,
#             bias=self.hparams.sigma_bias_flag,
#         )
#         self.mu = nn.Linear(self.hparams.input_dim, self.hparams.num_gaussian)
#         nn.init.normal_(self.mu.weight)
#         if self.hparams.mu_bias_init is not None:
#             for i, bias in enumerate(self.hparams.mu_bias_init):
#                 nn.init.constant_(self.mu.bias[i], bias)

#     def forward(self, x):
#         pi = self.pi(x)
#         sigma = self.sigma(x)
#         # Applying modified ELU activation
#         sigma = nn.ELU()(sigma) + 1 + 1e-15
#         mu = self.mu(x)
#         return pi, sigma, mu

#     def gaussian_probability(self, sigma, mu, target, log=False):
#         """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
#         Arguments:
#             sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
#                 size, G is the number of Gaussians, and O is the number of
#                 dimensions per Gaussian.
#             mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
#                 number of Gaussians, and O is the number of dimensions per Gaussian.
#             target (BxI): A batch of target. B is the batch size and I is the number of
#                 input dimensions.
#         Returns:
#             probabilities (BxG): The probability of each point in the probability
#                 of the distribution in the corresponding sigma/mu index.
#         """
#         target = target.expand_as(sigma)
#         if log:
#             ret = (
#                 -torch.log(sigma)
#                 - 0.5 * LOG2PI
#                 - 0.5 * torch.pow((target - mu) / sigma, 2)
#             )
#         else:
#             ret = (ONEOVERSQRT2PI / sigma) * torch.exp(
#                 -0.5 * ((target - mu) / sigma) ** 2
#             )
#         return ret  # torch.prod(ret, 2)

#     def log_prob(self, pi, sigma, mu, y):
#         log_component_prob = self.gaussian_probability(sigma, mu, y, log=True)
#         log_mix_prob = torch.log(
#             nn.functional.gumbel_softmax(pi, tau=1, dim=-1) + 1e-15
#         )
#         return torch.logsumexp(log_component_prob + log_mix_prob, dim=-1)

#     def sample(self, pi, sigma, mu):
#         """Draw samples from a MoG."""
#         categorical = Categorical(pi)
#         pis = categorical.sample().unsqueeze(1)
#         sample = Variable(sigma.data.new(sigma.size(0), 1).normal_())
#         # Gathering from the n Gaussian Distribution based on sampled indices
#         sample = sample * sigma.gather(1, pis) + mu.gather(1, pis)
#         return sample

#     def generate_samples(self, pi, sigma, mu, n_samples=None):
#         if n_samples is None:
#             n_samples = self.hparams.n_samples
#         samples = []
#         softmax_pi = nn.functional.gumbel_softmax(pi, tau=1, dim=-1)
#         assert (
#             softmax_pi < 0
#         ).sum().item() == 0, "pi parameter should not have negative"
#         for _ in range(n_samples):
#             samples.append(self.sample(softmax_pi, sigma, mu))
#         samples = torch.cat(samples, dim=1)
#         return samples

#     def generate_point_predictions(self, pi, sigma, mu, n_samples=None):
#         # Sample using n_samples and take average
#         samples = self.generate_samples(pi, sigma, mu, n_samples)
#         if self.hparams.central_tendency == "mean":
#             y_hat = torch.mean(samples, dim=-1)
#         elif self.hparams.central_tendency == "median":
#             y_hat = torch.median(samples, dim=-1).values
#         return y_hat
