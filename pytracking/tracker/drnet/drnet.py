from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import time
from pytracking import dcf, fourier, TensorList, operation
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor
from pytracking.libs.optimization import GaussNewtonCG, ConjugateGradient, GradientDescentL2
from .optim import ConvProblem, FactorizedConvProblem
from pytracking.features import augmentation
import numpy as np
def xyxy2xywh(boxes, stacked=False):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    if stacked:
        return torch.stack([cx, cy, w, h], dim=1)
    else:
        return cx, cy, w, h


class DRNet(BaseTracker):

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.features.initialize()
        self.features_initialized = True


    def initialize(self, image, state, *args, **kwargs):

        # Initialize some stuff
        self.frame_num = 1
        if not hasattr(self.params, 'device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize features
        self.initialize_features()

        # Check if image is color
        self.params.features.set_is_color(image.shape[2] == 3)

        # Get feature specific params
        self.fparams = self.params.features.get_fparams('feature_params')

        self.time = 0
        tic = time.time()

        # Get position and size
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Set search area
        self.target_scale = 1.0
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        if search_area > self.params.max_image_sample_size:
            self.target_scale =  math.sqrt(search_area / self.params.max_image_sample_size)
        elif search_area < self.params.min_image_sample_size:
            self.target_scale =  math.sqrt(search_area / self.params.min_image_sample_size)
        # print("self.target_scale",self.target_scale)
        # Check if IoUNet is used
        self.use_iou_net = getattr(self.params, 'use_iou_net', True)
        self.alpha = getattr(self.params, 'alpha', 0)
        self.beta = getattr(self.params, 'beta', 0)
        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Use odd square search area and set sizes
        feat_max_stride = max(self.params.features.stride())
        if getattr(self.params, 'search_area_shape', 'square') == 'square':
            self.img_sample_sz = torch.round(torch.sqrt(torch.prod(self.base_target_sz * self.params.search_area_scale))) * torch.ones(2)
        elif self.params.search_area_shape == 'initrect':
            self.img_sample_sz = torch.round(self.base_target_sz * self.params.search_area_scale)
        else:
            raise ValueError('Unknown search area shape')
        if self.params.feature_size_odd:
            self.img_sample_sz += feat_max_stride - self.img_sample_sz % (2 * feat_max_stride)
        else:
            self.img_sample_sz += feat_max_stride - (self.img_sample_sz + feat_max_stride) % (2 * feat_max_stride)

        # Set sizes
        self.img_support_sz = self.img_sample_sz
        self.feature_sz = self.params.features.size(self.img_sample_sz)
        self.output_sz = self.params.score_upsample_factor * self.img_support_sz  # Interpolated size of the output
        self.kernel_size = self.fparams.attribute('kernel_size')[0]

        self.iou_img_sample_sz = self.img_sample_sz

        # Optimization options
        self.params.precond_learning_rate = self.fparams.attribute('learning_rate')
        if self.params.CG_forgetting_rate is None or max(self.params.precond_learning_rate) >= 1:
            self.params.direction_forget_factor = 0
        else:
            self.params.direction_forget_factor = (1 - max(self.params.precond_learning_rate))**self.params.CG_forgetting_rate

        self.output_window = None
        if getattr(self.params, 'window_output', False):
            if getattr(self.params, 'use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), self.output_sz.long()*self.params.effective_search_area / self.params.search_area_scale, centered=False).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=False).to(self.params.device)

        # Initialize some learning things
        self.init_learning()

        # Convert image
        im = numpy_to_torch(image)
        self.im = im    # For debugging only

        # Setup scale bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        x = self.generate_init_samples(im)


        ## 初始化回归分支
        self.init_dr_net()


        # Initialize projection matrix
        self.init_projection_matrix(x)

        # Transform to get the training sample
        train_x = self.preprocess_sample(x)

        # Generate label function
        init_y = self.init_label_function(train_x)

        # Init memory
        self.init_memory(train_x)

        # Init optimizer and do initial optimization
        self.init_optimization(train_x, init_y)

        self.pos_drnet = self.pos.clone()

        self.time += time.time() - tic


    def init_optimization(self, train_x, init_y):
        # Initialize filter
        filter_init_method = getattr(self.params, 'filter_init_method', 'zeros')


        self.filter = TensorList(
            [x.new_zeros(1, cdim, sz[0], sz[1]) for x, cdim, sz in zip(train_x, self.compressed_dim, self.kernel_size)])

        if filter_init_method == 'zeros':
            pass
        elif filter_init_method == 'randn':
            for f in self.filter:
                f.normal_(0, 1/f.numel())
        elif filter_init_method == 'msra':
            for f in self.filter:
                nn.init.kaiming_normal_(f, a=1)

        else:
            raise ValueError('Unknown "filter_init_method"')

        # Get parameters
        self.params.update_projection_matrix = getattr(self.params, 'update_projection_matrix', True) and self.params.use_projection_matrix
        optimizer = getattr(self.params, 'optimizer', 'GaussNewtonCG')

        # Setup factorized joint optimization
        if self.params.update_projection_matrix:
            # self.joint_problem = FactorizedConvProblem(self.init_training_samples, init_y, self.filter_reg,
            #                                            self.fparams.attribute('projection_reg'), self.params, self.init_sample_weights,
            #                                            self.projection_activation, self.response_activation)
            self.joint_problem = FactorizedConvProblem(self.init_training_samples,init_y, self.filter_reg,
                                                       self.fparams.attribute('projection_reg'), self.params, self.init_sample_weights,
                                                       self.projection_activation, self.response_activation)  

            # Variable containing both filter and projection matrix

            # joint_var = TensorList([TensorList([tmp_filter,tmp_p]) for tmp_filter,tmp_p in zip(self.filter,self.projection_matrix)])
            joint_var = self.filter.concat(self.projection_matrix)
            # print(len(joint_var),len(self.init_training_samples),len(init_y))
            # Initialize optimizer
            analyze_convergence = getattr(self.params, 'analyze_convergence', False)
            if optimizer == 'GaussNewtonCG':
                # self.joint_optimizer = GaussNewtonCG(self.joint_problem, joint_var, plotting=(self.params.debug >= 3), analyze=analyze_convergence, fig_num=(12, 13, 14)) 
                self.joint_optimizer = GaussNewtonCG(self.joint_problem, joint_var, plotting=(self.params.debug >= 3), analyze=analyze_convergence, fig_num=(12, 13, 14)) 
            elif optimizer == 'GradientDescentL2':
                self.joint_optimizer = GradientDescentL2(self.joint_problem, joint_var, self.params.optimizer_step_length, self.params.optimizer_momentum, plotting=(self.params.debug >= 3), debug=analyze_convergence, fig_num=(12, 13))

            # Do joint optimization
            if isinstance(self.params.init_CG_iter, (list, tuple)):
                self.joint_optimizer.run(self.params.init_CG_iter)
            else:
                self.joint_optimizer.run(self.params.init_CG_iter // self.params.init_GN_iter, self.params.init_GN_iter)
                # for tmp_joint_optimizer in self.joint_optimizer:
                #     tmp_joint_optimizer.run(self.params.init_CG_iter // self.params.init_GN_iter, self.params.init_GN_iter)

            if analyze_convergence:
                opt_name = 'CG' if getattr(self.params, 'CG_optimizer', True) else 'GD'
                for val_name, values in zip(['loss', 'gradient'], [self.joint_optimizer.losses, self.joint_optimizer.gradient_mags]):
                    val_str = ' '.join(['{:.8e}'.format(v.item()) for v in values])
                    file_name = '{}_{}.txt'.format(opt_name, val_name)
                    with open(file_name, 'a') as f:
                        f.write(val_str + '\n')
                raise RuntimeError('Exiting')

        # Re-project samples with the new projection matrix
        compressed_samples = self.project_sample(self.init_training_samples, self.projection_matrix)
        for train_samp, init_samp in zip(self.training_samples, compressed_samples):
            train_samp[:init_samp.shape[0],...] = init_samp

        self.hinge_mask = None

        # Initialize optimizer
        self.conv_problem = ConvProblem(self.training_samples, self.y, self.filter_reg, self.sample_weights, self.response_activation,self.alpha,self.beta)
        # self.conv_problem = [ConvProblem(TensorList([tmp_train]), TensorList([tmp_y]), self.filter_reg, TensorList([tmp_weight]), self.response_activation) for tmp_train,tmp_y,tmp_weight in zip(self.training_samples,self.y,self.sample_weights)]
        if optimizer == 'GaussNewtonCG':
            self.filter_optimizer = ConjugateGradient(self.conv_problem, self.filter, fletcher_reeves=self.params.fletcher_reeves,
                                                      direction_forget_factor=self.params.direction_forget_factor, debug=(self.params.debug>=3), fig_num=(12,13))
            # self.filter_optimizer = [ConjugateGradient(tmp_problem, TensorList([tmp_filter]), fletcher_reeves=self.params.fletcher_reeves,
                                                      # direction_forget_factor=self.params.direction_forget_factor, debug=(self.params.debug>=3), fig_num=(12,13)) for tmp_problem,tmp_filter in zip(self.conv_problem,self.filter)]
        elif optimizer == 'GradientDescentL2':
            self.filter_optimizer = GradientDescentL2(self.conv_problem, self.filter, self.params.optimizer_step_length, self.params.optimizer_momentum, debug=(self.params.debug >= 3), fig_num=12)

        # Transfer losses from previous optimization
        if self.params.update_projection_matrix:
            self.filter_optimizer.residuals = self.joint_optimizer.residuals
            self.filter_optimizer.losses = self.joint_optimizer.losses
            # for tmp_filter_optimizer,tmp_joint_optimizer in zip(self.filter_optimizer,self.joint_optimizer):
            #     tmp_filter_optimizer.residuals = tmp_joint_optimizer.residuals
            #     tmp_filter_optimizer.losses = tmp_joint_optimizer.losses

        if not self.params.update_projection_matrix:
            self.filter_optimizer.run(self.params.init_CG_iter)
            # for tmp_filter_optimizer in self.filter_optimizer:
            #     tmp_filter_optimizer.run(self.params.init_CG_iter)

        # Post optimization
        self.filter_optimizer.run(self.params.post_init_CG_iter)
        # for tmp_filter_optimizer in self.filter_optimizer:
        #     tmp_filter_optimizer.run(self.params.post_init_CG_iter)

        # Free memory
        del self.init_training_samples
        if self.params.use_projection_matrix:
            # for tmp_joint_problem,tmp_joint_optimizer in zip(self.joint_problem,self.joint_optimizer):
            del self.joint_problem, self.joint_optimizer


    def track(self, image):

        self.frame_num += 1

        # Convert image
        im = numpy_to_torch(image)
        self.im = im    # For debugging only

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = self.pos.round()
        # print('33333',self.target_scale, self.params.scale_factors)
        sample_scales = self.target_scale * self.params.scale_factors
        test_x = self.extract_processed_sample(im, self.pos, sample_scales, self.img_sample_sz)

        # Compute scores
        scores_raw = self.apply_filter(test_x)

        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw)

        # Update position and scale
        if flag != 'not_found':
            # if self.use_iou_net:
            #     update_scale_flag = getattr(self.params, 'update_scale_when_uncertain', True) or flag != 'uncertain'
            #     if getattr(self.params, 'use_classifier', True):
            #         self.update_state(sample_pos + translation_vec)
            #     self.refine_target_box(sample_pos, sample_scales[scale_ind], scale_ind, update_scale_flag)
            if self.use_iou_net:
                update_scale_flag = getattr(self.params, 'update_scale_when_uncertain', True) or flag != 'uncertain'
                if getattr(self.params, 'use_classifier', True):
                    self.update_state(sample_pos + translation_vec)
                self.predict_target_box(sample_pos, sample_scales[scale_ind], scale_ind, update_scale_flag)

            elif getattr(self.params, 'use_classifier', True):
                self.update_state(sample_pos + translation_vec, sample_scales[scale_ind])

        if self.params.debug >= 2:
            show_tensor(s[scale_ind,...], 5, title='Max score = {:.2f}'.format(torch.max(s[scale_ind,...]).item()))


        # ------- UPDATE ------- #

        # Check flags and set learning rate if hard negative
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.hard_negative_learning_rate if hard_negative else None

        if update_flag:
            # Get train sample
            train_x = TensorList([x[scale_ind:scale_ind+1, ...] for x in test_x])


            # Create label for sample
            train_y = self.get_label_function(sample_pos, sample_scales[scale_ind])

            # Update memory
            self.update_memory(train_x, train_y, learning_rate)

        # Train filter
        if hard_negative:
            self.filter_optimizer.run(self.params.hard_negative_CG_iter)
            # for tmp_filter_optimizer in self.filter_optimizer:
            #     tmp_filter_optimizer.run(self.params.hard_negative_CG_iter)


        elif (self.frame_num-1) % self.params.train_skipping == 0:
            self.filter_optimizer.run(self.params.CG_iter)
            # for tmp_filter_optimizer in self.filter_optimizer:
            #     tmp_filter_optimizer.run(self.params.CG_iter)

        # Set the pos of the tracker to iounet pos
        if self.use_iou_net and flag != 'not_found':
            self.pos = self.pos_drnet.clone()

        # Return new state
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        return new_state.tolist()


    def apply_filter(self, sample_x: TensorList):
        return TensorList([operation.conv2d(tmp_x, tmp_filter, mode='same') for tmp_x,tmp_filter in zip(sample_x,self.filter)])

    def localize_target(self, scores_raw):
        # Weighted sum (if multiple features) with interpolation in fourier domain
        weight = self.fparams.attribute('translation_weight', 1.0)
        # scores_raw[0]=scores_raw[0]*0.7
        # scores_raw[1]=scores_raw[1]*0.3
        # scores_raw = weight * scores_raw
        sf_weighted = fourier.cfft2(scores_raw) / (scores_raw.size(2) * scores_raw.size(3))

        for i, (sz, ksz) in enumerate(zip(self.feature_sz, self.kernel_size)):
            sf_weighted[i] = fourier.shift_fs(sf_weighted[i], math.pi * (1 - torch.Tensor([ksz[0]%2, ksz[1]%2]) / sz))

        scores_fs = fourier.sum_fs(sf_weighted)
        scores = fourier.sample_fs(scores_fs, self.output_sz)

        if self.output_window is not None and not getattr(self.params, 'perform_hn_without_windowing', False):
            scores *= self.output_window

        if getattr(self.params, 'advanced_localization', False):
            return self.localize_advanced(scores)

        # Get maximum
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp.float().cpu()

        # Convert to displacements in the base scale
        disp = (max_disp + self.output_sz / 2) % self.output_sz - self.output_sz / 2

        # Compute translation vector and scale change factor
        translation_vec = disp[scale_ind, ...].view(-1) * (self.img_support_sz / self.output_sz) * self.target_scale
        translation_vec *= self.params.scale_factors[scale_ind]

        # Shift the score output for visualization purposes
        if self.params.debug >= 2:
            sz = scores.shape[-2:]
            scores = torch.cat([scores[...,sz[0]//2:,:], scores[...,:sz[0]//2,:]], -2)
            scores = torch.cat([scores[...,:,sz[1]//2:], scores[...,:,:sz[1]//2]], -1)

        return translation_vec, scale_ind, scores, None

    def localize_advanced(self, scores):
        """Does the advanced localization with hard negative detection and target not found."""

        sz = scores.shape[-2:]

        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            scores_orig = scores.clone()

            scores_orig = torch.cat([scores_orig[..., (sz[0] + 1) // 2:, :], scores_orig[..., :(sz[0] + 1) // 2, :]], -2)
            scores_orig = torch.cat([scores_orig[..., :, (sz[1] + 1) // 2:], scores_orig[..., :, :(sz[1] + 1) // 2]], -1)

            scores *= self.output_window

        # Shift scores back
        scores = torch.cat([scores[...,(sz[0]+1)//2:,:], scores[...,:(sz[0]+1)//2,:]], -2)
        scores = torch.cat([scores[...,:,(sz[1]+1)//2:], scores[...,:,:(sz[1]+1)//2]], -1)

        # Find maximum
        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - self.output_sz // 2
        translation_vec1 = target_disp1 * (self.img_support_sz / self.output_sz) * self.target_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'not_found'

        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            scores = scores_orig

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * self.target_sz / self.target_scale
        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores[scale_ind:scale_ind+1,...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - self.output_sz // 2
        translation_vec2 = target_disp2 * (self.img_support_sz / self.output_sz) * self.target_scale

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum(target_disp1**2))
            disp_norm2 = torch.sqrt(torch.sum(target_disp2**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'hard_negative'

        return translation_vec1, scale_ind, scores, None


    def extract_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        return self.params.features.extract(im, pos, scales, sz)

    def get_iou_features(self):
        return self.params.features.get_unique_attribute('iounet_features')

    def get_iou_backbone_features(self):
        return self.params.features.get_unique_attribute('iounet_backbone_features')

    def extract_processed_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor) -> (TensorList, TensorList):
        x = self.extract_sample(im, pos, scales, sz)
        return self.preprocess_sample(self.project_sample(x))

    def preprocess_sample(self, x: TensorList) -> (TensorList, TensorList):
        if getattr(self.params, '_feature_window', False):
            x = x * self.feature_window
        return x

    def project_sample(self, x: TensorList, proj_matrix = None):
        # Apply projection matrix

        if proj_matrix is None:
            proj_matrix = self.projection_matrix
        return TensorList([self.projection_activation(operation.conv2d(tmp_x, tmp_p)) for tmp_x,tmp_p in zip(x,proj_matrix)])

    def init_learning(self):
        # Get window function
        self.feature_window = TensorList([dcf.hann2d(sz).to(self.params.device) for sz in self.feature_sz])

        # Filter regularization
        self.filter_reg = self.fparams.attribute('filter_reg')
        
        # Activation function after the projection matrix (phi_1 in the paper)
        projection_activation = getattr(self.params, 'projection_activation', 'none')
        if isinstance(projection_activation, tuple):
            projection_activation, act_param = projection_activation

        if projection_activation == 'none':
            self.projection_activation = lambda x: x
        elif projection_activation == 'relu':
            self.projection_activation = torch.nn.ReLU(inplace=True)
        elif projection_activation == 'elu':
            self.projection_activation = torch.nn.ELU(inplace=True)
        elif projection_activation == 'mlu':
            self.projection_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

        # Activation function after the output scores (phi_2 in the paper)
        response_activation = getattr(self.params, 'response_activation', 'none')
        if isinstance(response_activation, tuple):
            response_activation, act_param = response_activation

        if response_activation == 'none':
            self.response_activation = lambda x: x
        elif response_activation == 'relu':
            self.response_activation = torch.nn.ReLU(inplace=True)
        elif response_activation == 'elu':
            self.response_activation = torch.nn.ELU(inplace=True)
        elif response_activation == 'mlu':
            self.response_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')


    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Generate augmented initial samples."""

        # Compute augmentation size
        aug_expansion_factor = getattr(self.params, 'augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift operator
        get_rand_shift = lambda: None
        random_shift_factor = getattr(self.params, 'random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor).long().tolist()

        # Create transofmations
        self.transforms = [augmentation.Identity(aug_output_sz)]
        if 'shift' in self.params.augmentation:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz) for shift in self.params.augmentation['shift']])
        if 'relativeshift' in self.params.augmentation:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz) for shift in self.params.augmentation['relativeshift']])
        if 'fliplr' in self.params.augmentation and self.params.augmentation['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in self.params.augmentation:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in self.params.augmentation['blur']])
        if 'scale' in self.params.augmentation:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in self.params.augmentation['scale']])
        if 'rotate' in self.params.augmentation:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in self.params.augmentation['rotate']])

        # Generate initial samples
        init_samples = self.params.features.extract_transformed(im, self.pos, self.target_scale, aug_expansion_sz, self.transforms)

        # Remove augmented samples for those that shall not have
        for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
            if not use_aug:
                init_samples[i] = init_samples[i][0:1, ...]

        # Add dropout samples
        if 'dropout' in self.params.augmentation:
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
                if use_aug:
                    init_samples[i] = torch.cat([init_samples[i], F.dropout2d(init_samples[i][0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        return init_samples


    def init_projection_matrix(self, x):
        # Set if using projection matrix
        self.params.use_projection_matrix = getattr(self.params, 'use_projection_matrix', True)

        if self.params.use_projection_matrix:
            self.compressed_dim = self.fparams.attribute('compressed_dim', None)[0]

            proj_init_method = getattr(self.params, 'proj_init_method', 'pca')
            if proj_init_method == 'pca':
                x_mat = TensorList([e.permute(1, 0, 2, 3).reshape(e.shape[1], -1).clone() for e in x])
                x_mat -= x_mat.mean(dim=1, keepdim=True)
                cov_x = x_mat @ x_mat.t()
                self.projection_matrix = TensorList(
                    [None if cdim is None else torch.svd(C)[0][:, :cdim].t().unsqueeze(-1).unsqueeze(-1).clone() for C, cdim in
                     zip(cov_x, self.compressed_dim)])
            elif proj_init_method == 'randn':
                self.projection_matrix = TensorList(
                    [None if cdim is None else ex.new_zeros(cdim,ex.shape[1],1,1).normal_(0,1/math.sqrt(ex.shape[1])) for ex, cdim in
                     zip(x, self.compressed_dim)])
            elif proj_init_method == 'msra':
                self.projection_matrix = TensorList(
                    [None if cdim is None else nn.init.kaiming_normal_(ex.new_zeros(cdim,ex.shape[1],1,1), a=1) for ex, cdim in
                     zip(x, self.compressed_dim)])
        else:
            self.compressed_dim = x.size(1)
            self.projection_matrix = TensorList([None]*len(x))

    def init_label_function(self, train_x):
        # Allocate label function
        self.y = TensorList([x.new_zeros(self.params.sample_memory_size, 1, x.shape[2], x.shape[3]) for x in train_x])
        # Output sigma factor
        output_sigma_factor = self.fparams.attribute('output_sigma_factor')
        # self.sigma = (self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)
        self.sigma = TensorList([(f / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor[0][num] * torch.ones(2) for num,f in enumerate(self.feature_sz)]).unroll()
        # Center pos in normalized coords
        target_center_norm = (self.pos - self.pos.round()) / (self.target_scale * self.img_support_sz)

        # Generate label functions
        for y, sig, sz, ksz, x in zip(self.y, self.sigma, self.feature_sz, self.kernel_size, train_x):
            center_pos = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * sz
                y[i, 0, ...] = dcf.label_function_spatial(sz, sig, sample_center)

        # Return only the ones to use for initial training
        return TensorList([y[:x.shape[0], ...] for y, x in zip(self.y, train_x)])


    def init_memory(self, train_x):
        # Initialize first-frame training samples
        self.num_init_samples = train_x.size(0)
        self.init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])
        self.init_training_samples = train_x

        # Sample counters and weights
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, self.init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, cdim, x.shape[2], x.shape[3]) for x, cdim in
             zip(train_x, self.compressed_dim)])

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, learning_rate = None):
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, self.fparams, learning_rate)
        self.previous_replace_ind = replace_ind
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x
        for y_memory, y, ind in zip(self.y, sample_y, replace_ind):
            y_memory[ind:ind+1,...] = y
        if self.hinge_mask is not None:
            for m, y, ind in zip(self.hinge_mask, sample_y, replace_ind):
                m[ind:ind+1,...] = (y >= self.params.hinge_threshold).float()
        self.num_stored_samples += 1


    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams, learning_rate = None):
        # Update weights and get index to replace in memory
        replace_ind = []
        for sw, prev_ind, num_samp, num_init, fpar in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams):
            lr = learning_rate
            if lr is None:
                lr = fpar.learning_rate

            init_samp_weight = getattr(fpar, 'init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw[s_ind:], 0)
                r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def get_label_function(self, sample_pos, sample_scale):
        # Generate label function
        train_y = TensorList()
        target_center_norm = (self.pos - sample_pos) / (sample_scale * self.img_support_sz)
        for sig, sz, ksz in zip(self.sigma, self.feature_sz, self.kernel_size):
            center = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            train_y.append(dcf.label_function_spatial(sz, sig, center))
        return train_y

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = 0.2
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates"""
        # print(self.iou_img_sample_sz,sample_scale)
        box_center = (pos - sample_pos) / sample_scale + (self.iou_img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        # print(target_ul,box_sz)
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])


    def init_dr_net(self):
        # Setup IoU net
        self.box_predictor = self.params.features.get_unique_attribute('iou_predictor')
        for p in self.box_predictor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.iou_target_box = self.get_iounet_box(self.pos, self.target_sz, self.pos.round(), self.target_scale)
        target_boxes = TensorList()
        target_boxes.append(self.iou_target_box.clone())
        target_boxes = torch.cat(target_boxes.view(1,4), 0).to(self.params.device)

        # Get iou features
        iou_backbone_features = self.get_iou_backbone_features()

        # Remove other augmentations such as rotation
        iou_backbone_features = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_features])

        # Extract target feat
        with torch.no_grad():
            target_feat = self.box_predictor.get_filter(iou_backbone_features, target_boxes)
        self.target_feat = TensorList([x.detach().mean(0) for x in target_feat])

        if getattr(self.params, 'iounet_not_use_reference', False):
            self.target_feat = TensorList([torch.full_like(tf, tf.norm() / tf.numel()) for tf in self.target_feat])
    def offset2box(self,init_box,offset):
        ctr_x = init_box[:,0]+0.5*init_box[:,2]
        ctr_y = init_box[:,1]+0.5*init_box[:,3]
        widths = init_box[:,2]
        heights = init_box[:,3]
        # ctr_x, ctr_y, widths, heights = init_box#xyxy2xywh(init_box)
        # print(ctr_x, ctr_y, widths, heights)

        wx, wy, ww, wh = 1,1,1,1
        dx = offset[:, 0::4] / wx
        dy = offset[:, 1::4] / wy
        dw = offset[:, 2::4] / ww
        dh = offset[:, 3::4] / wh

        # Prevent sending too large values into np.exp()
        dw = torch.clamp(dw, max=np.log(1000. / 16.))
        dh = torch.clamp(dh, max=np.log(1000. / 16.))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = offset.new_zeros(offset.shape)
        # # x1
        # pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # # y1
        # pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # # x2
        # pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # # y2
        # pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1
        pred_boxes[:, 0::4] = pred_ctr_x 
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y 
        # x2
        pred_boxes[:, 2::4] = pred_w 
        # y2
        pred_boxes[:, 3::4] = pred_h 
        return pred_boxes

    def predict_target_box(self, sample_pos, sample_scale, scale_ind, update_scale = True):
        # print(self.pos,sample_pos,self.target_sz)
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)
        init_box = init_box.unsqueeze(0)
        init_box = init_box.unsqueeze(0)
        init_box = init_box.cuda()
        # print(init_box.shape)
        iou_features = self.get_iou_features()
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])
        #预测回归值
        reg = self.box_predictor.predict_box(self.target_feat,iou_features,init_box)
        # print('reg',reg)
        init_box = init_box.view(-1,4)
        reg = reg.view(-1,4)
        
        predicted_box = self.offset2box(init_box,reg)
        
        # print(predicted_box.shape)
        predicted_box = predicted_box[0,:].cpu()
        # print(predicted_box.shape,self.iou_img_sample_sz.shape)
        new_pos = predicted_box[:2]  - (self.iou_img_sample_sz - 1) / 2
        new_pos = new_pos.flip((0,)) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        # Update position
        # new_pos = predicted_box[:2] + predicted_box[2:]/2 - (self.iou_img_sample_sz - 1) / 2
        # new_pos = new_pos.flip((0,)) * sample_scale + sample_pos
        # new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        # new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        self.pos_drnet = new_pos.clone()
        # print('pos',self.pos,new_pos)
        if getattr(self.params, 'use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()
        # print('target_sz',self.target_sz,new_target_sz)
        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

