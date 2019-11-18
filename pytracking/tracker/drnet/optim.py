import torch
from pytracking import optimization, TensorList, operation
import math


class FactorizedConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y: TensorList, filter_reg: torch.Tensor, projection_reg, params, sample_weights: TensorList,
                 projection_activation, response_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.params = params
        self.projection_reg = projection_reg
        self.projection_activation = projection_activation
        self.response_activation = response_activation
        # self.filter_reg = TensorList([self.filter_reg[0] for i in range(len(self.training_samples))])
        # projection_reg = TensorList([projection_reg[0] for i in range(len(self.training_samples))])
        self.diag_M = self.filter_reg.concat(projection_reg)


    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters, projection_matrices]
        :return: [data_terms, filter_regularizations, proj_mat_regularizations]
        """
        filter = x[:len(x)//2]  # w2 in paper
        P = x[len(x)//2:]       # w1 in paper

        # Do first convolution
        compressed_samples = operation.conv1x1(self.training_samples, P).apply(self.projection_activation)

        # Do second convolution
        residuals = operation.conv2d(compressed_samples, filter, mode='same').apply(self.response_activation)

        # Compute data residuals
        residuals = residuals - self.y
        # residuals[0] = torch.abs(residuals[0])
        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * filter)

        # Add regularization for projection matrix
        residuals.extend(self.projection_reg.apply(math.sqrt) * P)

        return residuals


    def ip_input(self, a: TensorList, b: TensorList):
        num = len(a) // 2       # Number of filters
        a_filter = a[:num]
        b_filter = b[:num]
        a_P = a[num:]
        b_P = b[num:]

        # Filter inner product
        # ip_out = a_filter.reshape(-1) @ b_filter.reshape(-1)
        ip_out = operation.conv2d(a_filter, b_filter).view(-1)

        # Add projection matrix part
        # ip_out += a_P.reshape(-1) @ b_P.reshape(-1)
        ip_out += operation.conv2d(a_P.view(1,-1,1,1), b_P.view(1,-1,1,1)).view(-1)

        # Have independent inner products for each filter
        return ip_out.concat(ip_out.clone())

    def M1(self, x: TensorList):
        # print(len(x),len(self.diag_M))
        return x / self.diag_M


class ConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y: TensorList, filter_reg: torch.Tensor, sample_weights: TensorList, response_activation,alpha=1,beta=1):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.response_activation = response_activation

        self.alpha = alpha
        self.beta = beta

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters]
        :return: [data_terms, filter_regularizations]
        """
        # Do convolution and compute residuals
        residuals = operation.conv2d(self.training_samples, x, mode='same').apply(self.response_activation)
        residuals = residuals - self.y
        # residuals[0] = torch.abs(residuals[0])
        # print(torch.sum(self.sample_weights[0]>0))
        # sorted_ohem_loss, idx = torch.sort(torch.sum(torch.abs(residuals[0][:torch.sum(self.sample_weights[0]>0),:,:,:]),(1,2,3)), descending=True)
        # print(idx.shape,residuals[0].shape)
        # sorted_ohem_loss, idx = torch.sort(torch.abs(residuals[0][:torch.sum(self.sample_weights[0]>0),:,:,:]),dim=0, descending=True)
        if self.beta<0:
            for i in range(len(residuals)):
                tmp_data=torch.abs(residuals[i])
                more_sample_weight = torch.max(torch.max(torch.max(tmp_data,1)[0],1)[0],1)[0]+0.01
                more_sample_weight = more_sample_weight/torch.max(more_sample_weight[:torch.sum(self.sample_weights[i]>0)])
  
                max_loss_weight = torch.exp(-self.beta*more_sample_weight)
                residuals[i]=max_loss_weight.view(-1,1,1,1)*residuals[i]
                # print(max_loss_weight)

        elif self.alpha>=0:
            # torch.abs(residuals[0][:torch.sum(self.sample_weights[0]>0),:,:,:])

            for i in range(len(residuals)):
                tmp_data=torch.abs(residuals[i])
                more_sample_weight = torch.sum(tmp_data,(1,2,3))+0.01
                more_sample_weight = more_sample_weight/torch.max(more_sample_weight[:torch.sum(self.sample_weights[i]>0)])

                max_loss=torch.max(torch.max(torch.max(tmp_data,1)[0],1)[0],1)[0]
                max_loss = max_loss.view(-1,1,1,1)+0.01
                tmp = torch.exp(self.alpha*tmp_data/max_loss)
                # print(tmp)
                # for i in range(torch.sum(self.sample_weights[0]>0)):
                max_loss_weight = torch.exp(self.beta*more_sample_weight)
                residuals[i]=max_loss_weight.view(-1,1,1,1)*residuals[i]
                # print(max_loss_weight)

                residuals[i]= tmp*residuals[i]
        else:
            for i in range(len(residuals)):
                sorted_ohem_loss, idx = torch.sort(torch.abs(residuals[i][:torch.sum(self.sample_weights[i]>0),:,:,:]),dim=0, descending=True)

                keep_num = min(sorted_ohem_loss.size()[0], 50)
                tmp_sample_weights = self.sample_weights.copy()
                
                if keep_num < sorted_ohem_loss.size()[0]:
                #     # print(1)

                    keep_idx_cuda = idx[:keep_num]
                    for m in range(keep_idx_cuda.shape[2]):
                        for n in range(keep_idx_cuda.shape[3]):
                            residuals[i][keep_idx_cuda[:,0,m,n],0,m,n] = residuals[i][keep_idx_cuda[:,0,m,n],0,m,n]*2     


        # keep_num = min(sorted_ohem_loss.size()[0], 30)
        tmp_sample_weights = self.sample_weights.copy()
        
        # if keep_num < sorted_ohem_loss.size()[0]:
        #     # print(1)

        #     keep_idx_cuda = idx[:keep_num]
        #     for i in range(keep_idx_cuda.shape[2]):
        #         for j in range(keep_idx_cuda.shape[3]):
        #             residuals[0][keep_idx_cuda[:,0,i,j],0,i,j] = residuals[0][keep_idx_cuda[:,0,i,j],0,i,j]*2
        #     # tmp_sample_weights[0][keep_idx_cuda] = 1/250
            # residuals[0][keep_idx_cuda,:,:,:]=0.


        residuals = tmp_sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * x)

        return residuals

    def ip_input(self, a: TensorList, b: TensorList):
        # return a.reshape(-1) @ b.reshape(-1)
        # return (a * b).sum()
        return operation.conv2d(a, b).view(-1)
