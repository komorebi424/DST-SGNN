import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
import scipy.linalg as la
from model.RevIN_L import RevIN


class SGSC(nn.Module):
    def __init__(self, pre_length, embed_size, seq_length,
                 feature_size, hidden_size, patch_len, d_model, hard_thresholding_fraction=1, hidden_size_factor=1,
                 sparsity_threshold=0.01):
        super(SGSC, self).__init__()
        self.dimension_factor = 5
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length

        self.seq_length = seq_length
        self.patch_len = patch_len
        self.d_model = d_model
        self.stride = 24
        self.patch_num = (self.seq_length - self.patch_len) // self.stride + 1
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale1 = 0.02
        self.embeddings = nn.Parameter(torch.randn(self.patch_len, self.embed_size))
        self.token_fc = nn.Linear(self.patch_len, self.embed_size)

        self.B = self.d_model

        self.b1 = nn.Parameter(self.scale1 * torch.randn(2, self.B))

        self.b2 = nn.Parameter(self.scale1 * torch.randn(2, self.B))

        self.fc12 = nn.Linear(self.patch_num * self.embed_size, 768).double()

        self.fc21 = nn.Linear(self.frequency_size, self.B)
        self.fc22 = nn.Linear(self.frequency_size, self.B)

        self.fc23 = nn.Linear(self.frequency_size, self.B)
        self.fc24 = nn.Linear(self.frequency_size, self.B)



        self.relu = nn.LeakyReLU(inplace=False)

        self.c_in = self.patch_num
        self.revin_layer = RevIN(self.c_in, affine=True, subtract_last=False)

        self.embeddings_10 = nn.Parameter(torch.randn(self.patch_num, 12).double())
        self.gcfc = nn.Linear(self.d_model, self.embed_size)
        self.to('cuda:0')

    def token_embedding(self, U):
        K = self.embeddings
        result1 = torch.matmul(U, K)

        return result1

    def LDGOSM(self, x):
        E = F.relu(x)

        I = torch.eye(E.size(2), device=E.device)
        E_T_E = torch.matmul(E.transpose(1, 2), E)
        A = I + E_T_E


        X_T_X = torch.matmul(x.transpose(1, 2), x)
        eigenvalues, eigenvectors = torch.linalg.eigh(X_T_X)

        L = self.relu(eigenvalues)
        L_inv_sqrt = torch.rsqrt(L + 1e-1) * self.scale1

        L_inv_sqrt_diag = torch.stack([torch.diag(l) for l in L_inv_sqrt])

        D = eigenvectors

        M_list = []
        for i in range(D.size(0)):
            M_i = torch.matmul(torch.matmul(D[i], L_inv_sqrt_diag[i]), D[i].transpose(0, 1))
            M_list.append(M_i)

        M = torch.stack(M_list)

        X_T_X_M_list = []
        for i in range(M.size(0)):
            X_T_X_M_list.append(torch.matmul(torch.matmul(M[i].transpose(0, 1), X_T_X[i]), M[i]))

        M_X_M = torch.stack(X_T_X_M_list)
        E_T_X_list = []
        for i in range(E.size(0)):
            E_T_X_list.append(torch.matmul(E[i].transpose(0, 1), x[i]))

        E_T_X = torch.stack(E_T_X_list)

        E_E_T_X_list = []
        for i in range(E_T_X.size(0)):
            E_E_T_X_list.append(torch.matmul(E[i], E_T_X[i]))

        E_E_T_X = torch.stack(E_E_T_X_list)
        X_T_E_E_T_X_list = []
        for i in range(E_E_T_X.size(0)):

            X_T_E_E_T_X_list.append(torch.matmul(x[i].transpose(1, 0), E_E_T_X[i]))

        X_T_E_E_T_X_list = torch.stack(X_T_E_E_T_X_list)


        M_E_M_list = []
        for i in range(M.size(0)):
            M_E_M_list.append(torch.matmul(torch.matmul(M[i].transpose(1, 0), X_T_E_E_T_X_list[i]),
                                           M[i]))

        M_E_M = torch.stack(M_E_M_list)

        result = M_X_M + M_E_M
        result_regularized = torch.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)

        eigenvalues_result, eigenvectors_result = torch.linalg.eigh(result_regularized)


        W_list = []
        for i in range(M.size(0)):
            W_i = torch.matmul(M[i], eigenvectors_result[i])
            W_list.append(W_i)

        W = torch.stack(W_list) * self.scale1

        return W

    def GC(self, x):

        X1 = self.fc21(x)
        X2 = self.fc22(x)

        W = self.LDGOSM(X1)
        P_real = torch.einsum('bpi,bii->bpi', X2, W)

        oa_real = torch.einsum('bpi,bik->bpk', P_real.transpose(2, 1), X2)
        oa_real = oa_real.permute(0, 2, 1)

        prr = F.softmax(P_real, dim=-1)

        w1 = self.fc23(x[0])
        w2 = self.fc24(x[0])
        w10 = torch.einsum('bpi,ij->bpj', prr.transpose(2, 1), w1)
        w20 = torch.einsum('bpi,ij->bpj', prr.transpose(2, 1), w2)

        o1_real = ( # 8 36 36 F^(T)X点积F^(T)G
                torch.einsum('bli,bii->bli', oa_real, w10) - \
                self.b1[0]
        )
        o2_real = (
                torch.einsum('bli,bii->bli', o1_real, w20) - \
                self.b2[0]
        )

        ob_real = torch.einsum('bik,bkp->bip', P_real, o2_real)
        ob_real = self.gcfc(ob_real)


        return ob_real

    def forward(self, x):

        x = x.permute(0, 2, 1).contiguous()
        B, N, L = x.shape
        x = self.token_fc(x)

        x = x.reshape(-1, self.patch_num, self.embed_size) # 8*7 12 128
        x = x.permute(0, 2, 1)
        x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)
        x = x.reshape(B, N, self.embed_size)

        bias = x
        x1 = self.GC(x)
        x1 = x1 + bias

        x1 = x1.reshape(-1, self.patch_num, self.embed_size)  # 8*7 12 128
        x1 = x1.permute(0, 2, 1)
        x1 = self.revin_layer(x1, 'denorm')
        x1 = x1.permute(0, 2, 1)

        x = x1.reshape(B, -1, self.patch_num * self.embed_size)

        x = x.double()

        x = self.fc12(x)


        x = x.float()
        return x
