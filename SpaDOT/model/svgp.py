import torch
import torch.nn as nn
import torch.nn.functional as F

class SVGP(nn.Module):
    def __init__(self, model_config, inducing_points, N_train, jitter=1e-2):
        '''
        SVGP model for variational inference in Gaussian processes.
        Parameters
        ----------
        model_config : dict
            Configuration dictionary containing model parameters.
        inducing_points : np.ndarray
            Inducing points for the SVGP model.
        N_train : int
            Number of training points.
        jitter : float
            Jitter term to add to the diagonal of covariance matrices for numerical stability.
        '''
        super(SVGP, self).__init__()
        self.N_train = N_train
        self.jitter = jitter
        # Inducing points
        self.inducing_index_points = torch.tensor(inducing_points, 
                                                  dtype=model_config['dtype']).to(model_config['device'])
        # Kernel
        self.kernel = Kernel(kernel_type=model_config['kernel_type'], 
                             scale=model_config['kernel_scale'], 
                             dtype=model_config['dtype'],
                             device=model_config['device'])


    def _add_diagonal_jitter(self, matrix, jitter):
        """
        Add jitter to the diagonal of a matrix.
        """
        if matrix.dim() == 2:
            Eye = torch.eye(matrix.size(0), device=matrix.device)
        else:
            Eye = torch.eye(matrix.size(-1), device=matrix.device).expand(matrix.shape)
        return matrix + jitter * Eye

    def kernel_matrix(self, x, y, diag_only=False):
        kernel_mat = self.kernel(x, y)
        return torch.diagonal(kernel_mat) if diag_only else kernel_mat

    def variational_loss(self, x, y, noise, mu_hat, A_hat):
        b, m = x.shape[0], self.inducing_index_points.shape[0]
        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)
        K_mm_inv = torch.linalg.inv(self._add_diagonal_jitter(K_mm, self.jitter))

        K_nn = self.kernel_matrix(x, x, diag_only=True)
        K_nm = self.kernel_matrix(x, self.inducing_index_points)
        K_mn = torch.transpose(K_nm, 0, 1)

        mean_vector = torch.matmul(K_nm, torch.matmul(K_mm_inv, mu_hat))
        KL_term = self._compute_kl_term(K_mm, K_mm_inv, mu_hat, A_hat, m)
        L_3_sum_term = self._compute_l3_term(K_nn, K_nm, K_mn, noise, y, mean_vector, K_mm_inv, A_hat, b)

        return L_3_sum_term, KL_term

    def approximate_posterior_params(self, index_points_test, index_points_train, y, noise):
        b = index_points_train.shape[0]
        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)
        K_mm_inv = torch.linalg.inv(self._add_diagonal_jitter(K_mm, self.jitter))

        K_xx = self.kernel_matrix(index_points_test, index_points_test, diag_only=True) 
        K_xm = self.kernel_matrix(index_points_test, self.inducing_index_points)
        K_mx = torch.transpose(K_xm, 0, 1)  # (m, x)

        K_nm = self.kernel_matrix(index_points_train, self.inducing_index_points)
        K_mn = torch.transpose(K_nm, 0, 1) 

        sigma_l = K_mm + (self.N_train / b) * torch.matmul(K_mn, K_nm / noise[:,None])
        sigma_l_inv = torch.linalg.inv(self._add_diagonal_jitter(sigma_l, self.jitter))
        mean_vector = (self.N_train / b) * torch.matmul(K_xm, torch.matmul(sigma_l_inv, torch.matmul(K_mn, y/noise)))

        K_xm_Sigma_l_K_mx = torch.matmul(K_xm, torch.matmul(sigma_l_inv, K_mx))
        B = K_xx + torch.diagonal(-torch.matmul(K_xm, torch.matmul(K_mm_inv, K_mx)) + K_xm_Sigma_l_K_mx)

        mu_hat = (self.N_train / b) * torch.matmul(torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mn)), y / noise)
        A_hat = torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mm))

        return mean_vector, B, mu_hat, A_hat

    def _compute_kl_term(self, K_mm, K_mm_inv, mu_hat, A_hat, m):
        K_mm_chol = torch.linalg.cholesky(self._add_diagonal_jitter(K_mm, self.jitter))
        S_chol = torch.linalg.cholesky(self._add_diagonal_jitter(A_hat, self.jitter))
        K_mm_log_det = 2 * torch.sum(torch.log(torch.diagonal(K_mm_chol)))
        S_log_det = 2 * torch.sum(torch.log(torch.diagonal(S_chol)))

        return 0.5 * (K_mm_log_det - S_log_det - m +
                      torch.trace(torch.matmul(K_mm_inv, A_hat)) +
                      torch.sum(mu_hat * torch.matmul(K_mm_inv, mu_hat)))

    def _compute_l3_term(self, K_nn, K_nm, K_mn, noise, y, mean_vector, K_mm_inv, A_hat, b):
        precision = 1 / noise
        K_tilde_terms = precision * (K_nn - torch.diagonal(torch.matmul(K_nm, torch.matmul(K_mm_inv, K_mn))))
        lambda_mat = torch.matmul(K_nm.unsqueeze(2), torch.transpose(K_nm.unsqueeze(2), 1, 2))
        lambda_mat = torch.matmul(K_mm_inv, torch.matmul(lambda_mat, K_mm_inv))
        trace_terms = precision * torch.einsum('bii->b', torch.matmul(A_hat, lambda_mat))
        return -0.5 * (torch.sum(K_tilde_terms) + torch.sum(trace_terms) +
                        torch.sum(torch.log(noise)) + b * torch.log(2 * torch.tensor(torch.pi)) +
                        torch.sum(precision * (y - mean_vector) ** 2))


'''
Kernel same as in SpatialPCA
'''
class Kernel(nn.Module):
    def __init__(self, kernel_type='Gaussian', scale=0.1, dtype=torch.float64, device='cpu'):
        super(Kernel, self).__init__()
        self.kernel_type = kernel_type
        self.scale = torch.tensor([scale], dtype=dtype, device=device)

    def forward(self, x, y):
        d = torch.cdist(x, y, p=2) # Euclidean distance
        # kernel type
        if self.kernel_type == 'Gaussian':
            res = torch.exp(-1*torch.square(d)/self.scale)
        if self.kernel_type == 'Cauchy':
            res = 1 / (1 + 1*torch.square(d)/self.scale)
        if self.kernel_type == 'Quadratic':
            res = 1 - 1 * torch.square(d) / (1 * torch.square(d) + self.scale)
        return res