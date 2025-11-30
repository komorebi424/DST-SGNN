import pandas as pd
import torch
from torch import nn
from model.SGSC_L import SGSC
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
import transformers
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


class LADSGNN(nn.Module):
    def __init__(self, stride, pre_length, embed_size, feature_size, seq_length, hidden_size, patch_len, d_model):
        super(LADSGNN, self).__init__()
        self.max_len = 60
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.stride = 24
        self.patch_len = patch_len
        self.d_model = d_model
        self.patch_num = (self.seq_length - self.patch_len) // self.stride + 1

        self.moving_avg = 2
        self.decompsition = series_decomp(self.moving_avg)

        self.model1s = SGSC(self.pre_length, self.embed_size, self.seq_length, self.feature_size, self.hidden_size,
                            self.patch_len, self.d_model)
        self.model1t = SGSC(self.pre_length, self.embed_size, self.seq_length, self.feature_size, self.hidden_size,
                            self.patch_len, self.d_model)

        self.fc1 = nn.Linear(768, 512).double()
        self.fc2 = nn.Linear(512, self.hidden_size).double()
        self.fc3 = nn.Linear(self.hidden_size, self.pre_length).double()

        self.llm_model = 'GPT2'
        if self.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('gpt2')

            self.gpt2_config.num_hidden_layers = 6
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        self.jayus0 = nn.Linear(768, 760)
        self.jayus1 = nn.Linear(768, 8)
        self.jayus2 = nn.Linear(self.max_len, 1)
        self.top_k = 5
        self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
        self.pred_len = pre_length
        self.seq_len = seq_length
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token


        for i, (name, param) in enumerate(self.llm_model.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="lora_only",
        )
        self.llm_model = get_peft_model(self.llm_model, config)

        self.cuda(0)

    def forward(self, x):

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x_normalized = (x - mean) / (std + 1e-8)

        x_normalized = x_normalized.float()

        z1 = x_normalized
        z1 = z1.permute(0, 2, 1)
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=24)

        zz1 = z1.reshape(z1.shape[0], z1.shape[1] * z1.shape[2], z1.shape[3])
        season1, trend1 = self.decompsition(zz1)
        m1s = season1.permute(0, 2, 1)
        m1t = trend1.permute(0, 2, 1)
        F1s = self.model1s(m1s)
        F1t = self.model1t(m1t)
        F1 = F1s + F1t

        de = F1
        F1 = self.jayus0(F1)
        x_normalized = x_normalized.permute(0, 2, 1)
        x_enc = x_normalized.contiguous().reshape(x_normalized.shape[0] * x_normalized.shape[1], x_normalized.shape[2],
                                                  1)

        x_sample = x_normalized[0]

        min_values = x_sample.min(dim=1).values
        max_values = x_sample.max(dim=1).values
        trends = x_sample[:, 1:] - x_sample[:, :-1]
        trend_sums = trends.sum(dim=1)

        corr_matrix = self.calculate_correlation_matrix(x_normalized, batch_idx=0)
        top_k_related = self.get_top_k_related(corr_matrix, k=self.top_k)

        prompt_list = []
        for i in range(F1.shape[1]):
            trend = trend_sums[i].item()
            related_vars = [f"V{j}(r={corr:.2f})" for j, corr in top_k_related[i]]
            spatial_desc = f"Top related variables to V{i}: {{{', '.join(related_vars)}}}"

            prompt = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                f"Input statistics: min value {min_values[i]:.2f}, max value {max_values[i]:.2f}. "
                f"The trend of input is {'upward' if trend > 0 else 'downward'}, "
                f"Spatio-temporality: Spatial dependencies: {spatial_desc}.<|end_prompt|>"
            )
            prompt_list.append(prompt)

        prompt_input_ids = self.tokenizer(
            prompt_list, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_len
        ).input_ids.to(x_enc.device)

        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_input_ids)

        prompt_embeddings = prompt_embeddings.unsqueeze(0).repeat(F1.shape[0], 1, 1, 1)

        prompt_embeddings = self.jayus1(prompt_embeddings)
        prompt_embeddings = prompt_embeddings.permute(0, 1, 3, 2)
        prompt_embeddings = self.jayus2(prompt_embeddings)

        x_prompt = prompt_embeddings.view(F1.shape[0], -1, 8)

        x_all = torch.cat([x_prompt, F1], dim=2)
        F3 = self.llm_model(inputs_embeds=x_all).last_hidden_state

        F4 = F3 + de
        F5 = F4.double()

        F5 = self.fc1(F5)
        F5 = self.fc2(F5).double()
        F5 = self.fc3(F5)

        K1 = F5
        K1 = K1.permute(0, 2, 1)

        K1_denormalized = K1 * (std + 1e-8) + mean
        K1_denormalized = K1_denormalized.permute(0, 2, 1)

        return K1_denormalized

    def calculate_correlation_matrix(self, F1, batch_idx=0):

        batch_data = F1[batch_idx]  # shape: [7, 768]

        mean = batch_data.mean(dim=1, keepdim=True)  # [7, 1]
        std = batch_data.std(dim=1, keepdim=True)  # [7, 1]
        std[std == 0] = 1e-8

        normalized = (batch_data - mean) / std  # shape: [7, 768]

        corr_matrix = torch.matmul(normalized, normalized.T) / (normalized.shape[1] - 1)
        return corr_matrix.cpu().numpy()  # shape: [7, 7]

    def get_top_k_related(self, corr_matrix, k=3):

        top_k_related = []
        for i in range(corr_matrix.shape[0]):
            corr_values = np.abs(corr_matrix[i])

            corr_values[i] = -np.inf
            top_indices = np.argsort(-corr_values)[:k]

            top_k_related.append([
                (j, corr_matrix[i, j]) for j in top_indices
            ])

        return top_k_related


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2 + 1)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)

        x = self.avg(x)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
