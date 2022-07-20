from pytracking.tracker.base import BaseTracker
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch, torch_to_numpy
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
from ltr.models.layers import activation

import ltr.data.processing_utils as prutils

import matplotlib.pyplot as plt

import pdb


class SLTTrDiMP(BaseTracker):

    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def batch_init(self, images, template_bbox, initial_bbox):
        """
        For inference in sequence-level training
        Note that sequences are duplicated for argmax & sampling tracker
        image - template bbox : pair for template frame (image: list (num_seq) of np.ndarray (hxwx3))
        initial bbox : gt in the first frame of test sequences (bbox: np.ndarray: num_seq x 4)
        """
        debug_info = {}
        self.frame_num = 1

        # The DiMP network
        self.net = self.params.net

        # Convert image
        img_list = [numpy_to_torch(im) for im in images] # list of array 1x3xhxw

        # Convert bbox
        template_bbox = bbutils.batch_xywh2center(torch.from_numpy(template_bbox))
        initial_bbox = bbutils.batch_xywh2center(torch.from_numpy(initial_bbox))

        # Get target position and size
        self.poses = template_bbox[:, [1,0]]   # cy, cx # (num_seq)x2
        self.target_szs = template_bbox[:, [3,2]] # h, w # (num_seq)x2

        # Set sizes
        self.image_szs = torch.from_numpy(np.stack([[im.shape[2], im.shape[3]] for im in img_list])) # (num_seq)x2
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_sample_sz = sz # [352, 352]
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_szs * self.params.search_area_scale, 1)
        self.target_scales = torch.sqrt(search_area) / self.img_sample_sz.prod().sqrt() # Tensor(num_seq)

        # Target size in base scale
        self.base_target_szs = self.target_szs / self.target_scales.unsqueeze(1) # (num_seq)x2

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.min_scale_factors, _ = torch.max(10 / self.base_target_szs, dim=1) # Tensor(num_seq)
        self.max_scale_factors, _ = torch.min(self.image_szs / self.base_target_szs, dim=1) # Tensor(num_seq)

        # Extract and transform sample
        # init_backbone_feat: 'layer2': Tensor((13xnum_seq)x512x44x44), 'layer3': Tensor((13xnum_seq)x1024x22x22)
        init_backbone_feat, im_patches = self.batch_generate_init_samples(img_list)

        # Initialize classifier
        self.batch_init_classifier(init_backbone_feat, len(img_list))

        if self.params.get('use_iou_net', True):
            self.batch_init_iou_net(init_backbone_feat, len(img_list))

        ############# Set previous pos and target sz as initial bbox (for sequential tracking)
        self.poses = initial_bbox[:, [1,0]]   # cy, cx # (num_seq)x2
        self.target_szs = initial_bbox[:, [3,2]] # h, w # (num_seq)x2

        # Set search area
        search_area = torch.prod(self.target_szs * self.params.search_area_scale, 1)
        self.target_scales = torch.sqrt(search_area) / self.img_sample_sz.prod().sqrt() # Tensor(num_seq)

        # Target size in base scale
        self.base_target_szs = self.target_szs / self.target_scales.unsqueeze(1) # (num_seq)x2

        # Setup scale bounds
        self.min_scale_factors, _ = torch.max(10 / self.base_target_szs, dim=1) # Tensor(num_seq)
        self.max_scale_factors, _ = torch.min(self.image_szs / self.base_target_szs, dim=1) # Tensor(num_seq)

        # Save train_images(cropped image), train_anno (bbox(x,y,w,h) in cropped image), train_label (transformer label)
        # for sequence-level training
        num_seq = len(img_list)
        num_img = im_patches.shape[0] // num_seq    # 13 #TODO: consider dropout
        im_patches = im_patches.view(num_img, num_seq, *im_patches.shape[-3:])
        target_boxes = [t[:num_img, ...] for t in self.target_boxes]
        target_boxes = torch.stack(target_boxes, dim=1)
        transformer_label = self.transformer_label[:num_img, ...]

        out = {'template_images': im_patches,  # images x sequences x 3x352x352
               'template_anno': target_boxes,    # images x sequences x 4
               'template_label': transformer_label,       # images x sequences x 22 x 22
               'debug_info': debug_info}
        return out


    def batch_track(self, images, gt_boxes, action_mode='half'):
        self.frame_num += 1

        # Convert image
        img_list = [numpy_to_torch(im) for im in images] # list of array 1x3xhxw

        # ------- LOCALIZATION ------- #

        # Extract backbone features (Set crop window and extract backbone features.)
        backbone_feat, sample_coords, im_patches = self.batch_extract_backbone_features(img_list,
                                                                        self._batch_get_centered_sample_pos(),
                                                                        self.target_scales*self.params.scale_factors,
                                                                        self.img_sample_sz)

        # Extract classification features
        x_clf = self.get_classification_features(backbone_feat) # num_seqx512x22x22
        decoded_x, test_x = self.transformer_decoder(x_clf) # decoded_x: 484xnum_seqx512, test_x: num_seqx512x22x22
        test_x = test_x.unsqueeze(0) # 1xnum_seqx512x22x22

        # Location of sample
        sample_pos, sample_scales = self._batch_get_sample_location(sample_coords) # sample_pos: num_seqx2, sample_scales: num_seq

        # Compute classification scores
        scores_raw = self.batch_classify_target(test_x)

        # Localize the target
        translation_vec, selected_ind, scores_softmax, flag, selected_ind_1d, max_sel = self.batch_localize_target(
                                                                            scores_raw, sample_pos, sample_scales,
                                                                            action_mode)
        new_pos = sample_pos + translation_vec

        # Update position and scale
        if flag != 'not_found':
            if self.params.get('use_iou_net', True):
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
                if self.params.get('use_classifier', True):
                    self.batch_update_state(new_pos)
                self.batch_refine_target_box(backbone_feat, sample_pos, sample_scales, update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.batch_update_state(new_pos, sample_scales)

        # ------- UPDATE ------- #
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        #TODO: update
        if update_flag and self.params.get('update_classifier', False):
            # Get train sample
            train_x = test_x.clone()

            # Create target_box and label for spatial sample
            target_box = self._batch_get_iounet_box(self.poses, self.target_szs, sample_pos, sample_scales)

            # Update the classifier model
            self.batch_update_classifier(train_x, target_box, learning_rate, scores_softmax)

            if (self.frame_num - 1) % self.params.transformer_skipping == 0:
                # Update Transformer memory
                cur_tf_label = prutils.gaussian_label_function(target_box.cpu().view(-1, 4), 0.1,
                                                               self.net.classifier.filter_size,
                                                               self.feature_sz, self.img_sample_sz,
                                                               end_pad_if_even=False)  # num_seqx22x22

                if self.x_clf.shape[0] < self.params.transformer_memory_size:
                    self.transformer_label = torch.cat([cur_tf_label.unsqueeze(0).cuda(), self.transformer_label],
                                                       dim=0)  # num_memory x num_seq x 22 x 22
                    self.x_clf = torch.cat([x_clf.unsqueeze(0), self.x_clf],
                                           dim=0)  # num_memory x num_seq x 512 x 22 x 22
                else:
                    self.transformer_label = torch.cat(
                        [cur_tf_label.unsqueeze(0).cuda(), self.transformer_label[:-1, ...]], dim=0)
                    self.x_clf = torch.cat([x_clf.unsqueeze(0), self.x_clf[:-1, ...]], dim=0)
                self.transformer_memory, _ = self.net.classifier.transformer.encoder(self.x_clf, pos=None)

        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.poses = self.pos_iounet.clone()

        # Compute output bounding box
        new_state = torch.cat((self.poses[:, [1, 0]] - (self.target_szs[:, [1, 0]] - 1) / 2, self.target_szs[:, [1, 0]])
                              , dim=1)

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [[-1, -1, -1, -1]]*new_state.shape[0]
        else:
            output_state = new_state.tolist()

        # Generate labels for sequence-level training
        # search_images(cropped image), search_anno (bbox(x,y,w,h) in cropped image),
        # (bbox regressor) search_proposals, proposal_density, gt_density,
        # (classifier) search_label
        gt_bbox = bbutils.batch_xywh2center(torch.from_numpy(gt_boxes)) # xywh -> cxcywh
        gt_poses = gt_bbox[:, [1,0]]   # cy, cx # (num_seq)x2
        gt_target_szs = gt_bbox[:, [3,2]] # h, w # (num_seq)x2
        search_anno = self._batch_get_iounet_box(gt_poses, gt_target_szs, sample_pos, sample_scales)  # num_seqx4
        search_proposals, proposal_density, gt_density = zip(*[self._generate_proposals(a) for a in search_anno])
        search_label = self._generate_label_function(search_anno) # num_seqx23x23

        out = {'target_bbox': np.array(output_state),
               'search_images': im_patches,               # sequences x 3x352x352
               'search_anno': search_anno,                  # sequences x 4
               'search_label': search_label,                # sequences x 23 x 23
               'search_proposals': torch.stack(search_proposals),        # tuple(num_seq) of tensor(num_proposals=128x4) -> tensor(num_seqxnum_proposalx4)
               'proposal_density': torch.stack(proposal_density),    # tuple(num_seq) of tensor(num_proposal) -> tensor(num_seqxnum_proposal)
               'gt_density': torch.stack(gt_density),                # tuple(num_seq) of tensor(num_proposal) -> tensor(num_seqxnum_proposal)
               'selected_indices': selected_ind,        # sequences x 2
               'selected_indices_1d': selected_ind_1d}  # sequences
        return out

    # %---------- backbone -------------%

    def batch_generate_init_samples(self, img_list):
        """
        Perform data augmentation to generate initial training samples.
        return: init_backbone_feat: {'layer2': Tensor((num_patchxnum_seq)x512x44x44),
                                    'layer3': Tensor((num_patchxnum_seq)x1024x22x22)}
                im_patches: (num_patchxnum_seq)x3x352x352
        """

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            raise NotImplementedError
        else:
            self.init_sample_scales = self.target_scales
            global_shift = torch.zeros(2)

        self.init_sample_poses = self.poses.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        bs = len(img_list) // 2
        assert len(img_list) % 2 == 0
        self.transforms = [None]*len(img_list)
        for i in range(bs):
            transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

            augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

            # Add all augmentations
            if 'shift' in augs:
                transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
            if 'relativeshift' in augs:
                get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
                transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
            if 'fliplr' in augs and augs['fliplr']:
                transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
            if 'blur' in augs:
                transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
            if 'scale' in augs:
                transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
            if 'rotate' in augs:
                transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

            # duplicate twice for argmax tracker & sampling tracker
            self.transforms[i] = transforms.copy()
            self.transforms[i+bs] = transforms.copy()

        # Extract augmented image patches
        for i in range(len(img_list)):
            cur_im_patches = sample_patch_transformed(img_list[i], self.init_sample_poses[i],
                                                  self.init_sample_scales[i], aug_expansion_sz, self.transforms[i]) # num_patchx3x352x352
            if i==0:
                im_patches = cur_im_patches.unsqueeze(1)
            else:
                im_patches = torch.cat((im_patches, cur_im_patches.unsqueeze(1)), dim=1)
        im_patches = im_patches.reshape(-1, *im_patches.shape[-3:]) # num_patchxnum_seqx3x352x352 -> (num_patchxnum_seq)x3x352x352

        # Extract initial backbone features
        # 'layer2': Tensor((num_patchxnum_seq)x512x44x44), 'layer3': Tensor((num_patchxnum_seq)x1024x22x22)
        with torch.no_grad():
            init_backbone_feat, im_patches = self.net.preprocess_extract_backbone(im_patches)

        return init_backbone_feat, im_patches


    def batch_extract_backbone_features(self, img_list, pos, scales, sz):

        # Extract multiscale image patches
        im_patches_list = []
        patch_coords_list = []
        for im_, pos_, scales_ in zip(img_list, pos, scales):
            # scale * self.img_sample_sz = size to crop
            # self.img_sample_sz = size to be resized (size of im_patches)
            im_patches, patch_coords = sample_patch_multiscale(im_, pos_, scales_.unsqueeze(0), sz,
                                                            mode=self.params.get('border_mode', 'replicate'),
                                                            max_scale_change=self.params.get('patch_max_scale_change',
                                                                                             None))
            im_patches_list.append(im_patches)
            patch_coords_list.append(patch_coords)
        im_patches = torch.cat(im_patches_list, dim=0)
        patch_coords = torch.cat(patch_coords_list, dim=0)

        # Extract backbone features
        with torch.no_grad():
            backbone_feat, im_patches = self.net.preprocess_extract_backbone(im_patches) # 'layer2': Tensor(num_seqx512x44x44), 'layer3': Tensor(num_seqx1024x22x22)

        return backbone_feat, patch_coords, im_patches


    # %---------- classifier -------------%

    def batch_init_classifier(self, init_backbone_feat, num_seq):
        """
        Initialize classifier.
        init_backbone_feat: 'layer2': Tensor((13xnum_seq)x512x44x44), 'layer3': Tensor((13xnum_seq)x1024x22x22)
        """

        # Get classification features
        x = self.get_classification_features(init_backbone_feat) # (13xnum_seq)x512x22x22
        x = x.view(-1, num_seq, *x.shape[-3:]).contiguous() # 13xnum_seqxcxhxw

        # Overwrite some parameters in the classifier. (These are not generally changed)
        self._overwrite_classifier_params(feature_dim=x.shape[-3])

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']

            for i in range(num_seq):
                self.transforms[i].extend(self.transforms[i][:1]*num) # add identity

            bs = num_seq // 2
            assert num_seq % 2 == 0
            x_list = [None]*num_seq
            for i in range(bs):
                here_x = x[:,i,...]
                here_x = torch.cat([here_x, F.dropout2d(here_x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])
                x_list[i] = here_x.clone()
                x_list[i+bs] = here_x.clone()
            x = torch.stack(x_list, 1) # -> 15xnum_seqx512x22x22

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:])) # [22,22]
        ksz = self.net.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self._batch_init_target_boxes(num_seq) # num_samplexnum_seqx4

        # mask in Transformer
        self.transformer_label = prutils.gaussian_label_function(target_boxes.cpu().view(-1, 4), 0.1, self.net.classifier.filter_size,
                                                                 self.feature_sz, self.img_sample_sz, end_pad_if_even=False)
        # (15xnum_seq)x22x22 -> 15xnum_seqx22x22
        self.transformer_label = self.transformer_label.view(-1, num_seq, *self.transformer_label.shape[-2:]).cuda()

        # input should be (num_imgs, batch, dim, h, w) = (num_samples, num_seq, dim. h, w)
        self.x_clf = x
        self.transformer_memory, _ = self.net.classifier.transformer.encoder(self.x_clf, pos=None) # transformer_memory: (15x22x22)xnum_seqx512

        for i in range(self.x_clf.shape[0]):
            _, cur_encoded_feat = self.net.classifier.transformer.decoder(x[i,...].unsqueeze(0), memory=self.transformer_memory, pos=self.transformer_label, query_pos=None)
            if i == 0:
                encoded_feat = cur_encoded_feat.unsqueeze(0) # num_seqx512x22x22
            else:
                encoded_feat = torch.cat((encoded_feat, cur_encoded_feat.unsqueeze(0)), 0) # -> num_samplexnum_seqx512x22x22
        x = encoded_feat.contiguous()
        # x = encoded_feat.view(num_samples, num_seq, c, h, w).contiguous()

        # Set number of iterations
        num_iter = self.params.get('net_opt_iter', None)

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes, num_iter=num_iter)

        # Init memory
        if self.params.get('update_classifier', True):
            self._batch_init_memory(TensorList([x]))


    def batch_update_classifier(self, train_x, target_box, learning_rate=None, scores=None):
        # train_x : 1xnum_seqx512x22x22
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self._batch_update_memory(TensorList([train_x]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.params.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('net_opt_low_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        # plot_loss = self.params.debug > 0
        plot_loss=False
        if num_iter > 0:
            for i in range(train_x[0].shape[1]):
                # Get inputs for the DiMP filter optimizer module
                samples = self.training_samples[i][0][:self.num_stored_samples[i][0],...] # dim: num_samplex512x22x22
                target_boxes = self.target_boxes[i][:self.num_stored_samples[i][0],:].clone() # dim: num_samplesx4
                sample_weights = self.sample_weights[i][0][:self.num_stored_samples[i][0]] # dim: num_samples

                # Run the filter optimizer module
                with torch.no_grad():
                    self.target_filter[i], _, losses = self.net.classifier.filter_optimizer(self.target_filter[i],
                                                                                         num_iter=num_iter, feat=samples,
                                                                                         bb=target_boxes,
                                                                                         sample_weight=sample_weights,
                                                                                         compute_losses=plot_loss)
            '''
            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.cat(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
                elif self.params.debug >= 3:
                    plot_graph(self.losses, 10, title='Training Loss' + self.id_str)
            '''


    # %---------- iou net -------------%

    def batch_init_iou_net(self, backbone_feat, num_seq):
        """
        backbone_feat: {'layer2': Tensor((13xnum_seq)x512x44x44), 'layer3': Tensor((13xnum_seq)x1024x22x22)}
        """

        # Setup IoU net and objective
        for p in self.net.bb_regressor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        target_boxes = TensorList()
        self.classifier_target_box = self._batch_get_iounet_box(self.poses, self.target_szs, self.init_sample_poses,
                                                                self.init_sample_scales)
        for i in range(num_seq):
            if self.params.iounet_augmentation:
                for T in self.transforms[i]:
                    if not isinstance(T, (augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal,
                                          augmentation.FlipVertical, augmentation.Blur)):
                        break
                    target_boxes.append(self.classifier_target_box[i] + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
            else:
                target_boxes.append(self.classifier_target_box[i] + torch.Tensor([self.transforms[i][0].shift[1],
                                                                                  self.transforms[i][0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(-1, 4), 0).cuda()

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

        # Remove other augmentations such as rotation
        k = iou_backbone_feat[0].shape[0]
        num_samples_orig = k//num_seq
        num_samples_cut = target_boxes.shape[0]//num_seq
        idx = torch.arange(num_samples_orig*num_seq).reshape(num_samples_orig, num_seq)[:num_samples_cut, :].reshape(-1)
        iou_backbone_feat = TensorList([x[idx, ...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes) # (num_seqx256x1x1, num_seqx256x1x1)
        if torch.is_tensor(self.iou_modulation[0]):
            self.iou_modulation = TensorList([x.detach() for x in self.iou_modulation])


    def batch_refine_target_box(self, backbone_feat, sample_pos, sample_scale, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        if hasattr(self.net.bb_regressor, 'predict_bb'):
            # return self.direct_box_regression(backbone_feat, sample_pos, sample_scale, 0, update_scale)
            raise NotImplementedError

        # Initial box for refinement
        init_box = self._batch_get_iounet_box(self.poses, self.target_szs, sample_pos, sample_scale) # num_seqx4 (x y w h)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat) # (Tensor(num_seq,256,44,44), Tensor(num_seq,256,22,22))
        iou_features = TensorList([x for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(-1,4).clone() # num_seqx4
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[:, 2:].prod(1).sqrt() # num_seq
            rand_factor = square_box_sz.unsqueeze(1) * torch.cat([self.params.box_jitter_pos * torch.ones(2),
                                                     self.params.box_jitter_sz * torch.ones(2)]).unsqueeze(0) # num_seqx4

            minimal_edge_size, _ = init_box[:, 2:].min(1)
            minimal_edge_size = minimal_edge_size / 3 # num_seq
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5).unsqueeze(0) * rand_factor.unsqueeze(1) # num_seqx9x4
            new_sz = torch.max(init_box[:, 2:].unsqueeze(1) + rand_bb[..., 2:], minimal_edge_size.unsqueeze(1).unsqueeze(1)) # num_seqx9x2
            new_center = (init_box[:, :2] + init_box[:, 2:]/2).unsqueeze(1) + rand_bb[..., :2] # num_seqx9x2
            init_boxes = torch.cat([new_center - new_sz/2, new_sz], 2) # num_seqx9x4
            init_boxes = torch.cat([init_box.view(-1,4).unsqueeze(1), init_boxes],dim=1) # num_seqx10x4

        # Optimize the boxes
        output_boxes, output_iou = self._batch_optimize_boxes(iou_features, init_boxes)

        # Remove weird boxes
        self.poses_iounet = self.poses.clone()
        for i, (cur_box, cur_iou) in enumerate(zip(output_boxes, output_iou)):
            cur_box[:, 2:].clamp_(1)
            aspect_ratio = cur_box[:,2] / cur_box[:,3]
            keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)
            cur_output_boxes = cur_box[keep_ind,:]
            cur_output_iou = cur_iou[keep_ind]

            # If no box found
            if cur_output_boxes.shape[0] == 0:
                continue

            # Predict box
            k = self.params.get('iounet_k', 5)
            topk = min(k, cur_output_boxes.shape[0])
            _, inds = torch.topk(cur_output_iou, topk)
            predicted_box = cur_output_boxes[inds, :].mean(0)
            predicted_iou = cur_output_iou.view(-1, 1)[inds, :].mean(0)

            # Get new position and size
            new_pos = predicted_box[:2] + predicted_box[2:] / 2
            new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale[i] + sample_pos[i]
            new_target_sz = predicted_box[2:].flip((0,)) * sample_scale[i]
            new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_szs[i].prod())

            self.poses_iounet[i] = new_pos.clone()

            if self.params.get('use_iounet_pos_for_learning', True):
                self.poses[i] = new_pos.clone()

            self.target_szs[i] = new_target_sz

            if update_scale:
                self.target_scales[i] = new_scale

        # self.visualize_iou_pred(iou_features, predicted_box)


    def _batch_get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        sample_scale=sample_scale.unsqueeze(1)
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((1,)), box_sz.flip((1,))], dim=1)


    def _batch_optimize_boxes(self, iou_features, init_boxes):
        box_refinement_space = self.params.get('box_refinement_space', 'default')
        if box_refinement_space == 'default':
            raise NotImplementedError
        if box_refinement_space == 'relative':
            return self._batch_optimize_boxes_relative(iou_features, init_boxes)
        raise ValueError('Unknown box_refinement_space {}'.format(box_refinement_space))


    def _batch_optimize_boxes_relative(self, iou_features, init_boxes):
        """Optimize iounet boxes with the relative parametrization ised in PrDiMP"""
        # init_boxes: num_seq x num_random_box x 4
        output_boxes = init_boxes.cuda() # num_seqx10x4
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).cuda().view(1,1,4)

        sz_norm = output_boxes[:,:1,2:].clone() # num_seqx1x2
        output_boxes_rel = bbutils.rect_to_rel(output_boxes, sz_norm)

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init_rel = output_boxes_rel.clone().detach()
            bb_init_rel.requires_grad = True

            bb_init = bbutils.rel_to_rect(bb_init_rel, sz_norm)
            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init) # num_seqx10

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            # print(outputs)

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
            output_boxes_rel.detach_()

            step_length *= self.params.box_refinement_step_decay

        output_boxes = bbutils.rel_to_rect(output_boxes_rel, sz_norm)

        return output_boxes.cpu(), outputs.detach().cpu()


    # %---------- track functions -------------%

    def batch_localize_target_prev(self, scores, sample_pos, sample_scales, action_mode):
        """Run the target localization."""
        if sample_scales.dim() > 1:
            raise NotImplementedError #TODO: multiple patches

        scores = scores.squeeze() # 1,num_seq,23,23 -> num_seq,23,23

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            # Apply softmax
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            raise NotImplementedError

        if self.params.get('advanced_localization', False):
            raise NotImplementedError

        # half -> argmax, half -> sampling
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        if action_mode == 'max':
            max_scores, selected_indices = dcf.max2d(scores)
        elif action_mode == 'sample':
            scores_1d = scores.view(scores.shape[0], -1)
            prob = Categorical(scores_1d)
            selected_indices = prob.sample()
            selected_indices = torch.stack([selected_indices // score_sz[1], selected_indices % score_sz[1]], dim=1).int()
        elif action_mode == 'half':
            max_scores, max_indices = dcf.max2d(scores)
            scores_1d = scores.view(scores.shape[0], -1)
            prob = Categorical(scores_1d)
            sampled_indices = prob.sample()
            sampled_indices = torch.stack([sampled_indices // score_sz[1], sampled_indices % score_sz[1]], dim=1).int()
            bs = len(max_indices) // 2
            assert len(max_indices) % 2 == 0
            selected_indices = torch.cat([max_indices[:bs], sampled_indices[bs:]], dim=0)
        selected_indices = selected_indices.detach().cpu()

        # Get maximum
        score_center = (score_sz - 1)/2
        # _, scale_ind = torch.max(max_score, dim=0)
        # max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = selected_indices - score_center

        # Compute translation vector and scale change factor
        # output_sz = score_sz - (self.kernel_size.cuda() + 1) % 2
        # translation_vec = target_disp * (self.img_support_sz.cuda() / output_sz) * sample_scales.unsqueeze(1).repeat(1,2)
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz).unsqueeze(0) * sample_scales.unsqueeze(1)

        return translation_vec, selected_indices, scores, None


    def batch_localize_target(self, scores, sample_pos, sample_scales, action_mode):
        """Run the target localization."""
        if sample_scales.dim() > 1:
            raise NotImplementedError #TODO: multiple patches

        scores = scores.squeeze() # 1,num_seq,23,23 -> num_seq,23,23

        if self.params.get('advanced_localization', False):
            max_indices_advanced, scores, flag = self._batch_localize_advanced(scores, sample_pos, sample_scales)
        else:
            flag = [None]*scores.shape[0]

        # Apply softmax
        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            # Apply softmax
            scores_view = scores.view(scores.shape[0], -1)

            mean_val = torch.mean(scores_view, dim=-1)
            max_val = torch.max(scores_view, dim=-1)[0]
            scores_norm = (scores_view - mean_val.unsqueeze(1)) / (max_val - mean_val).unsqueeze(1)
            range_mean, range_max = self.params.score_norm_meanmax
            scores_norm = scores_norm*(range_max - range_mean) + range_mean

            scores_softmax = torch.softmax(scores_norm, dim=-1)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        # argmax
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        max_scores, max_indices = dcf.max2d(scores)
        if self.params.get('advanced_localization', False):
            for i in range(scores.shape[0]):
                if flag[i] == 'hard_negative':
                    print((max_indices[i] == max_indices_advanced[i]).items())
                    max_indices[i] = max_indices_advanced[i]
        max_indices_1d = (max_indices[:, 0]*score_sz[1] + max_indices[:, 1]).int()

        # sampling
        scores_1d = scores.view(scores.shape[0], -1)
        prob = Categorical(scores_1d)
        sampled_indices_1d = prob.sample()
        sampled_indices = torch.stack([torch.div(sampled_indices_1d, score_sz[1], rounding_mode='trunc'), sampled_indices_1d % score_sz[1]], dim=1).int()

        # half -> argmax, half -> sampling
        max_selected_1d = max_indices_1d == sampled_indices_1d
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        if action_mode == 'max':
            selected_indices = max_indices
            selected_indices_1d = max_indices_1d
        elif action_mode == 'sample':
            selected_indices = sampled_indices
            selected_indices_1d = sampled_indices_1d
        elif action_mode == 'half':
            bs = len(max_indices) // 2
            assert len(max_indices) % 2 == 0
            selected_indices = torch.cat([max_indices[:bs], sampled_indices[bs:]], dim=0)
            selected_indices_1d = torch.cat([max_indices_1d[:bs], sampled_indices_1d[bs:]], dim=0)
        selected_indices = selected_indices.detach().cpu()
        selected_indices_1d = selected_indices_1d.detach().cpu()

        # Get maximum
        score_center = (score_sz - 1)/2
        target_disp = selected_indices - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz).unsqueeze(0) * sample_scales.unsqueeze(1)

        return translation_vec, selected_indices, scores, flag, selected_indices_1d, max_selected_1d


    def _batch_localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz)) # [23,23]
        output_sz = score_sz - (self.kernel_size + 1) % 2 # [22,22]
        score_center = (score_sz - 1)/2 # [11,11]

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores) # get argmax. max_score1: maximum value, max_disp1: argmax position (ex. 11,11)
        max_disp1 = max_disp1.cpu()
        target_disp1 = max_disp1 - score_center

        flag = []
        selected_indices = max_disp1
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_szs / sample_scales.unsqueeze(1))\
                          * (output_sz / self.img_support_sz).unsqueeze(0)
        disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2
        for i in range(scores.shape[0]):
            flag.append(None)
            if max_score1[i].item() < self.params.target_not_found_threshold: # th = 0.25
                flag[i] = 'not found'
            elif max_score1[i].item() < self.params.get('uncertain_threshold', -float('inf')):
                flag[i] = 'uncertain'
            elif max_score1[i].item() < self.params.get('hard_sample_threshold', -float('inf')):
                flag[i] = 'hard_negative'

            if flag[i] != None:
                continue

            # Mask out target neighborhood
            tneigh_top = max(round(max_disp1[i,0].item() - target_neigh_sz[i,0].item() / 2), 0)
            tneigh_bottom = min(round(max_disp1[i,0].item() + target_neigh_sz[i,0].item() / 2 + 1), sz[0])
            tneigh_left = max(round(max_disp1[i,1].item() - target_neigh_sz[i,1].item() / 2), 0)
            tneigh_right = min(round(max_disp1[i,1].item() + target_neigh_sz[i,1].item() / 2 + 1), sz[1])
            scores_masked = scores_hn[i].clone()
            scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

            # Find new maximum
            new_max_score, new_max_disp = dcf.max2d(scores_masked)
            new_max_disp = new_max_disp.float().cpu().view(-1)
            new_target_disp = new_max_disp - score_center

            prev_target_vec = (self.poses[i] - sample_pos[i]) / ((self.img_support_sz / output_sz) * sample_scales[i])

            # Handle the different cases
            if new_max_score > self.params.distractor_threshold * max_score1[i]:
                disp_norm1 = torch.sqrt(torch.sum((target_disp1[i]-prev_target_vec)**2))
                disp_norm2 = torch.sqrt(torch.sum((new_target_disp-prev_target_vec)**2))

                if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                    flag[i] = 'hard_negative'
                elif disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                    selected_indices[i] = new_max_disp
                    flag[i] = 'hard_negative'
                elif disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                    flag[i] = 'uncertain'
                else:
                    if new_max_score > self.params.target_not_found_threshold:
                        flag[i] = 'hard_negative'
                    else:
                        # If also the distractor is close, return with highest score
                        flag[i] = 'uncertain'
            else:
                flag[i] = 'normal'

        return selected_indices.cuda(), scores_hn, flag


    def batch_transformer_decoder(self, x):
        """
        Transformer decoder for batch images.
        x : Tensor(num_seq, c, h, w)
        input should be (num_imgs, batch, dim, h, w) = (num_samples, num_seq, dim. h, w)
        """

        for i in range(x.shape[0]):
            with torch.no_grad():
                _, cur_decoded_feat = self.net.classifier.transformer.decoder(x[i,...].unsqueeze(0).unsqueeze(0),
                                                                              memory=self.transformer_memory[:,i,...].unsqueeze(1),
                                                                              pos=self.transformer_label[:,i,...].unsqueeze(1),
                                                                              query_pos=None)
            if i == 0:
                decoded_feat = cur_decoded_feat
            else:
                decoded_feat = torch.cat((decoded_feat, cur_decoded_feat), 0) #-> num_seqx512x22x22
        decoded_feat.view(x.shape).contiguous()
        return decoded_feat


    def batch_classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        # target_filter : num_seqx512x4x4
        # sample_x : 1xnum_seqx512x22x22
        for i in range(sample_x.shape[1]):
            with torch.no_grad():
                cur_scores = self.net.classifier.classify(self.target_filter[i].unsqueeze(0), sample_x[:,i])
            if i == 0:
                scores = cur_scores
            else:
                scores = torch.cat((scores, cur_scores), 1) # -> 1xnum_seqx23x23
        return scores


    def batch_update_state(self, new_poses, new_scales = None):
        # Update scale
        if new_scales is not None:
            self.target_scales = torch.max(torch.min(new_scales, self.max_scale_factors), self.min_scale_factors)
            # self.target_szs = self.base_target_szs * self.target_scales.unsqueeze(1) # -> this makes error when base size is changed

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_szs
        self.poses = torch.max(torch.min(new_poses, self.image_szs - inside_offset), inside_offset)

    def update_state_each(self, idx, new_poses, new_scales = None):
        # Update scale
        if new_scales is not None:
            self.target_scales[idx] = torch.max(torch.min(new_scales, self.max_scale_factors[idx]), self.min_scale_factors[idx])
            # self.target_szs = self.base_target_szs * self.target_scales.unsqueeze(1) # -> this makes error when base size is changed

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_szs[idx]
        self.poses[idx] = torch.max(torch.min(new_poses, self.image_szs[idx] - inside_offset), inside_offset)


    # %---------- collect training samples -------------%

    def _batch_init_target_boxes(self, num_seq):
        """
        Get the target bounding boxes for the initial augmented samples.
        return: init_target_boxes: tensor(num_samplexnum_seqx4)
        """
        self.classifier_target_box = [] # list (num_seq) of tensor(4)
        self.target_boxes = [] # list(num_seq) of tensor(50,4)

        for i in range(num_seq):
            # target box coordinate in the cropped sample, in the form used in IoUNet (x y w h)
            self.classifier_target_box.append(self.get_iounet_box(self.poses[i], self.target_szs[i],
                                                                self.init_sample_poses[i], self.init_sample_scales[i]))

            # perform transforms
            cur_init_target_boxes = TensorList()
            for T in self.transforms[i]:
                cur_init_target_boxes.append(self.classifier_target_box[i] + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
            cur_init_target_boxes = torch.cat(cur_init_target_boxes.view(1, 4), 0).cuda()

            target_boxes = cur_init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
            target_boxes[:cur_init_target_boxes.shape[0], :] = cur_init_target_boxes
            self.target_boxes.append(target_boxes)

            if i==0:
                init_target_boxes = cur_init_target_boxes.unsqueeze(1)
            else:
                init_target_boxes = torch.cat((init_target_boxes, cur_init_target_boxes.unsqueeze(1)), dim=1)
        return init_target_boxes


    def _batch_init_memory(self, train_x: TensorList):
        """
        train_x : TensorList [x], x = Tensor(num_samples, num_seq, c, h, w)
        """
        self.num_init_samples=[]
        self.num_stored_samples=[]
        self.previous_replace_ind=[]
        self.sample_weights=[]
        self.training_samples=[]
        num_samples, num_seq, c, h, w = train_x[0].shape
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])
        for i in range(num_seq):
            # Initialize first-frame spatial training samples
            self.num_init_samples.append(train_x.size(0)) # TensorList[num_samples]

            # Sample counters and weights for spatial
            self.num_stored_samples.append(self.num_init_samples[i].copy())
            self.previous_replace_ind.append([None] * len(self.num_stored_samples[i]))
            self.sample_weights.append(TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x]))
            for sw, init_sw, num in zip(self.sample_weights[i], init_sample_weights, self.num_init_samples[i]):
                sw[:num] = init_sw

            # Initialize memory
            self.training_samples.append(TensorList(
                [x.new_zeros(self.params.sample_memory_size, c, h, w) for x in train_x]))

            for ts, x in zip(self.training_samples[i], train_x):
                ts[:x.shape[0],...] = x[:, i, ...]


    def _batch_update_memory(self, sample_x: TensorList, target_box, learning_rate = None):
        """sample_x: TensorList [x], x = Tensor(1, num_seq, c, h, w)"""
        for i in range(sample_x[0].shape[1]):
            # Update weights and get replace ind
            replace_ind = self.update_sample_weights(self.sample_weights[i], self.previous_replace_ind[i],
                                                     self.num_stored_samples[i], self.num_init_samples[i], learning_rate)
            self.previous_replace_ind[i] = replace_ind

            # Update sample and label memory
            for train_samp, x, ind in zip(self.training_samples[i], sample_x, replace_ind):
                train_samp[ind:ind+1,...] = x[:, i, ...]

            # Update bb memory
            self.target_boxes[i][replace_ind[0],:] = target_box[i]

            self.num_stored_samples[i] += 1

    # %---------- transformation(coordinate) function -------------%

    def _batch_get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()

        # Update base target sz
        self.base_target_szs = self.target_szs / sample_scales.unsqueeze(1)
        return sample_pos, sample_scales

    def _batch_get_centered_sample_pos(self):
        # poses: dim=num_seqx2, target_scales: dim=num_seq
        # feature_sz, kernel_size, img_support_sz: dim=2
        return self.poses + ((self.feature_sz + self.kernel_size) % 2).unsqueeze(0) * self.target_scales.unsqueeze(1)\
               * (self.img_support_sz / (2 * self.feature_sz)).unsqueeze(0)

    def _generate_proposals(self, box):
        """ Generate proposal sample boxes from a GMM proposal distribution and compute their ground-truth density.
        This is used for ML and KL based regression learning of the bounding box regressor.
        args:
            box - input bounding box
        """
        # Generate proposals
        proposals, proposal_density, gt_density = prutils.sample_box_gmm(box,
                                                                         self.params.proposal_params['proposal_sigma'],
                                                                         gt_sigma=self.params.proposal_params['gt_sigma'],
                                                                         num_samples=self.params.proposal_params[
                                                                             'boxes_per_frame'],
                                                                         add_mean_box=self.params.proposal_params.get(
                                                                             'add_mean_box', False))
        return proposals, proposal_density, gt_density

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4),
                                                      self.params.label_params['sigma_factor'],
                                                      self.params.label_params['kernel_sz'],
                                                      self.params.label_params['feature_sz'], self.img_sample_sz[0].item(),
                                                      end_pad_if_even=self.params.label_params.get('end_pad_if_even', True))
        return gauss_label

    #####################################################################
    #####################################################################

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        # Convert image
        im = numpy_to_torch(image)

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat = self.generate_init_samples(im)

        # Initialize classifier
        self.init_classifier(init_backbone_feat)

        # Initialize IoUNet
        if self.params.get('use_iou_net', True):
            self.init_iou_net(init_backbone_feat)

        out = {'time': time.time() - tic}
        return out


    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # print(self.frame_num)

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features (Set crop window and extract backbone features.)
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                      self.target_scale * self.params.scale_factors,
                                                                      self.img_sample_sz)
        # Extract classification features
        x_clf = self.get_classification_features(backbone_feat)
        decoded_x, test_x = self.transformer_decoder(x_clf)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw = self.classify_target(test_x)

        # Localize the target
        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)
        new_pos = sample_pos[scale_ind,:] + translation_vec

        # Update position and scale
        if flag != 'not_found':
            if self.params.get('use_iou_net', True):
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
                if self.params.get('use_classifier', True):
                    self.update_state(new_pos)
                self.refine_target_box(backbone_feat, sample_pos[scale_ind,:], sample_scales[scale_ind], scale_ind, update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])


        # ------- UPDATE ------- #
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False):
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])

            # Update the classifier model
            self.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])
        
            if (self.frame_num - 1) % self.params.transformer_skipping == 0:
                # Update Transformer memory
                cur_tf_label = prutils.gaussian_label_function(target_box.cpu().view(-1, 4), 0.1, self.net.classifier.filter_size,
                                                            self.feature_sz, self.img_sample_sz, end_pad_if_even=False)

                if self.x_clf.shape[0] < self.params.transformer_memory_size:
                    self.transformer_label = torch.cat([cur_tf_label.unsqueeze(1).cuda(), self.transformer_label], dim=0)
                    self.x_clf = torch.cat([x_clf, self.x_clf], dim=0)   
                else:
                    self.transformer_label = torch.cat([cur_tf_label.unsqueeze(1).cuda(), self.transformer_label[:-1,...]], dim=0)
                    self.x_clf = torch.cat([x_clf, self.x_clf[:-1,...]], dim=0)
                self.transformer_memory, _ = self.net.classifier.transformer.encoder(self.x_clf.unsqueeze(1), pos=None)

        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()

        # Visualize and set debug info
        self.search_area_box = torch.cat((sample_coords[scale_ind,[1,0]], sample_coords[scale_ind,[3,2]] - sample_coords[scale_ind,[1,0]] - 1))
        self.debug_info['flag' + self.id_str] = flag
        self.debug_info['max_score' + self.id_str] = max_score
        if self.visdom is not None:
            self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
        elif self.params.debug >= 2:
            show_tensor(score_map, 5, title='Max score = {:.2f}'.format(max_score))

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        out = {'target_bbox': output_state}
        return out


    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.net.classifier.classify(self.target_filter, sample_x)
        return scores

    def transformer_decoder(self, sample_x: TensorList):
        """Transformer."""
        with torch.no_grad():
            decoded_feat, out_feat = self.net.classifier.transformer.decoder(sample_x.unsqueeze(0), memory=self.transformer_memory, pos=self.transformer_label, query_pos=None)  ######### self.transformer_label
        return decoded_feat, out_feat

    def localize_target(self, scores, sample_pos, sample_scales):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1,1,score_filter_ksz,score_filter_ksz)
            scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None


    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores) # get argmax
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found'
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind,:]) / ((self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        return translation_vec1, scale_ind, scores_hn, 'normal'

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat)
    
    def get_iou_backbone_features(self, backbone_feat):
        return self.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)


    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x


    def update_memory(self, sample_x: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box

        self.num_stored_samples += 1


    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
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

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)


    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))]) # x y w h


    def init_iou_net(self, backbone_feat):
        # Setup IoU net and objective
        for p in self.net.bb_regressor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        target_boxes = TensorList()
        if self.params.iounet_augmentation:
            for T in self.transforms:
                if not isinstance(T, (augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal, augmentation.FlipVertical, augmentation.Blur)):
                    break
                target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.classifier_target_box + torch.Tensor([self.transforms[0].shift[1], self.transforms[0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(1,4), 0).to(self.params.device)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

        # Remove other augmentations such as rotation
        iou_backbone_feat = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes)
        if torch.is_tensor(self.iou_modulation[0]):
            self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])


    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Overwrite some parameters in the classifier. (These are not generally changed)
        self._overwrite_classifier_params(feature_dim=x.shape[-3])

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = self.net.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)

        # mask in Transformer
        self.transformer_label = prutils.gaussian_label_function(target_boxes.cpu().view(-1, 4), 0.1, self.net.classifier.filter_size,
                                                                 self.feature_sz, self.img_sample_sz, end_pad_if_even=False)

        self.transformer_label = self.transformer_label.unsqueeze(1).cuda()   
        self.x_clf = x

        self.transformer_memory, _ = self.net.classifier.transformer.encoder(self.x_clf.unsqueeze(1), pos=None)

        for i in range(x.shape[0]):
            _, cur_encoded_feat = self.net.classifier.transformer.decoder(x[i,...].unsqueeze(0).unsqueeze(0), memory=self.transformer_memory, pos=self.transformer_label, query_pos=None)
            if i == 0:
                encoded_feat = cur_encoded_feat
            else:
                encoded_feat = torch.cat((encoded_feat, cur_encoded_feat), 0)
        x = encoded_feat.contiguous()

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes, num_iter=num_iter,
                                                                           compute_losses=plot_loss)

        # Init memory
        if self.params.get('update_classifier', True):
            self.init_memory(TensorList([x]))  

        '''
        if plot_loss:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.cat(losses)
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
            elif self.params.debug >= 3:
                plot_graph(self.losses, 10, title='Training Loss' + self.id_str)
        '''

    def _overwrite_classifier_params(self, feature_dim):
        # Overwrite some parameters in the classifier. (These are not generally changed)
        pred_module = getattr(self.net.classifier.filter_optimizer, 'score_predictor', self.net.classifier.filter_optimizer)
        if self.params.get('label_threshold', None) is not None:
            self.net.classifier.filter_optimizer.label_threshold = self.params.label_threshold
        if self.params.get('label_shrink', None) is not None:
            self.net.classifier.filter_optimizer.label_shrink = self.params.label_shrink
        if self.params.get('softmax_reg', None) is not None:
            self.net.classifier.filter_optimizer.softmax_reg = self.params.softmax_reg
        if self.params.get('filter_reg', None) is not None:
            pred_module.filter_reg[0] = self.params.filter_reg
            pred_module.min_filter_reg = self.params.filter_reg
        if self.params.get('filter_init_zero', False):
            self.net.classifier.filter_initializer = FilterInitializerZero(self.net.classifier.filter_size, feature_dim)


    def update_classifier(self, train_x, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.params.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('net_opt_low_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        plot_loss = self.params.debug > 0

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.net.classifier.filter_optimizer(self.target_filter,
                                                                                     num_iter=num_iter, feat=samples,
                                                                                     bb=target_boxes,
                                                                                     sample_weight=sample_weights,
                                                                                     compute_losses=plot_loss)
            '''
            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.cat(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
                elif self.params.debug >= 3:
                    plot_graph(self.losses, 10, title='Training Loss' + self.id_str)
            '''
            
    def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        if hasattr(self.net.bb_regressor, 'predict_bb'):
            return self.direct_box_regression(backbone_feat, sample_pos, sample_scale, scale_ind, update_scale)

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1,4).clone()
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat([self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])

            minimal_edge_size = init_box[2:].min()/3
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
            init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1,4), init_boxes])

        # Optimize the boxes
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)
        output_boxes = output_boxes[keep_ind,:]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # Predict box
        k = self.params.get('iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

        # self.visualize_iou_pred(iou_features, predicted_box)


    def optimize_boxes(self, iou_features, init_boxes):
        box_refinement_space = self.params.get('box_refinement_space', 'default')
        if box_refinement_space == 'default':
            return self.optimize_boxes_default(iou_features, init_boxes)
        if box_refinement_space == 'relative':
            return self.optimize_boxes_relative(iou_features, init_boxes)
        raise ValueError('Unknown box_refinement_space {}'.format(box_refinement_space))


    def optimize_boxes_default(self, iou_features, init_boxes):
        """Optimize iounet boxes with the default parametrization"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]], device=self.params.device).view(1,1,4)

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
            output_boxes.detach_()

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()


    def optimize_boxes_relative(self, iou_features, init_boxes):
        """Optimize iounet boxes with the relative parametrization ised in PrDiMP"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(self.params.device).view(1,1,4)

        sz_norm = output_boxes[:,:1,2:].clone()
        output_boxes_rel = bbutils.rect_to_rel(output_boxes, sz_norm)
        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init_rel = output_boxes_rel.clone().detach()
            bb_init_rel.requires_grad = True

            bb_init = bbutils.rel_to_rect(bb_init_rel, sz_norm)
            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
            output_boxes_rel.detach_()

            step_length *= self.params.box_refinement_step_decay

        #     for s in outputs.view(-1):
        #         print('{:.2f}  '.format(s.item()), end='')
        #     print('')
        # print('')

        output_boxes = bbutils.rel_to_rect(output_boxes_rel, sz_norm)

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()

    def direct_box_regression(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Implementation of direct bounding box regression."""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1, 1, 4).clone().to(self.params.device)

        # Optimize the boxes
        output_boxes = self.net.bb_regressor.predict_bb(self.iou_modulation, iou_features, init_boxes).view(-1,4).cpu()

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)

        predicted_box = output_boxes[0, :]

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale_bbr = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())
        new_scale = new_scale_bbr

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale


    def visualize_iou_pred(self, iou_features, center_box):
        center_box = center_box.view(1,1,4)
        sz_norm = center_box[...,2:].clone()
        center_box_rel = bbutils.rect_to_rel(center_box, sz_norm)

        pos_dist = 1.0
        sz_dist = math.log(3.0)
        pos_step = 0.01
        sz_step = 0.01

        pos_scale = torch.arange(-pos_dist, pos_dist+pos_step, step=pos_step)
        sz_scale = torch.arange(-sz_dist, sz_dist+sz_step, step=sz_step)

        bbx = torch.zeros(1, pos_scale.numel(), 4)
        bbx[0,:,0] = pos_scale.clone()
        bby = torch.zeros(pos_scale.numel(), 1, 4)
        bby[:,0,1] = pos_scale.clone()
        bbw = torch.zeros(1, sz_scale.numel(), 4)
        bbw[0,:,2] = sz_scale.clone()
        bbh = torch.zeros(sz_scale.numel(), 1, 4)
        bbh[:,0,3] = sz_scale.clone()

        pos_boxes = bbutils.rel_to_rect((center_box_rel + bbx) + bby, sz_norm).view(1,-1,4).to(self.params.device)
        sz_boxes = bbutils.rel_to_rect((center_box_rel + bbw) + bbh, sz_norm).view(1,-1,4).to(self.params.device)

        pos_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, pos_boxes).exp()
        sz_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, sz_boxes).exp()

        show_tensor(pos_scores.view(pos_scale.numel(),-1), title='Position scores', fig_num=21)
        show_tensor(sz_scores.view(sz_scale.numel(),-1), title='Size scores', fig_num=22)


    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
            self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')