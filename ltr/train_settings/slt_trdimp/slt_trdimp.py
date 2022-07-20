import torch.optim as optim

from ltr.dataset import Lasot, Got10k, TrackingNet
from ltr.data import sequence_sampler, SLTLoader
from ltr.models.tracking import dimpnet
import ltr.models.loss.kl_regression as klreg_losses
from ltr.actors.slt_trdimp_actor import SLTTrDiMPActor
from ltr.trainers.slt_trdimp_trainer import SLTTrDiMPTrainer
from ltr import MultiGPU
from ltr.admin import loading

from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone


def run(settings):
    settings.description = 'SLT-TrDiMP with default settings.'
    settings.num_workers = 4
    settings.multi_gpu = True
    settings.print_interval = 1

    # SLT settings
    settings.num_epoch = 40
    settings.num_per_epoch = 5000
    settings.num_seq = 8
    settings.num_seq_backward = 4
    settings.num_frames = 24
    settings.clip_grad = True
    settings.grad_max_norm = 40.0

    # TrDiMP
    settings.search_area_factor = 6.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 22
    settings.output_sz = settings.feature_sz * 16
    settings.reward_weight = 1
    settings.net_opt_iter = 5

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(6, 12)))

    # Train sampler and loader
    settings.max_gap = 300
    settings.max_interval = 10
    settings.interval_prob = 0.3
    dataset_train = sequence_sampler.SequenceSampler([lasot_train, got10k_train, trackingnet_train], [1,1,1],
                                        samples_per_epoch=settings.num_per_epoch, max_gap=settings.max_gap, max_interval=settings.max_interval,
                                                     num_search_frames=settings.num_frames, num_template_frames=1,
                                                     frame_sample_mode='random_interval', prob=settings.interval_prob)

    loader_train = SLTLoader('train', dataset_train, training=True, batch_size=settings.num_seq, num_workers=settings.num_workers,
                             shuffle=False, drop_last=True)

    # Create network and actor
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    net = dimpnet.dimpnet50(filter_size=settings.target_filter_sz, backbone_pretrained=True, optim_iter=settings.net_opt_iter,
                            clf_feat_norm=True, clf_feat_blocks=0, final_conv=True, out_feature_dim=512,
                            optim_init_step=0.9, optim_init_reg=0.1,
                            init_gauss_sigma=output_sigma * settings.feature_sz, num_dist_bins=100,
                            bin_displacement=0.1, mask_init_factor=3.0, target_mask_act='sigmoid', score_act='relu',
                            frozen_backbone_layers=['conv1', 'bn1', 'layer1', 'layer2'])

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = {'bb_ce': klreg_losses.KLRegression()}

    loss_weight = {'bb_ce': 0.01, 'sl_clf': 4}

    actor = SLTTrDiMPActor(net=net, objective=objective, loss_weight=loss_weight)
    actor.params = get_tracker_params(settings)    # Tracker parameters for inference mode

    # Optimizer
    optimizer = optim.Adam([{'params': actor.net.classifier.filter_initializer.parameters(), 'lr': 2e-6},
                            {'params': actor.net.classifier.filter_optimizer.parameters(), 'lr': 2e-5},
                            {'params': actor.net.classifier.feature_extractor.parameters(), 'lr': 2e-6},
                            {'params': actor.net.classifier.transformer.parameters(), 'lr': 4e-5},
                            {'params': actor.net.bb_regressor.parameters(), 'lr': 4e-5},
                            {'params': actor.net.feature_extractor.layer3.parameters(), 'lr': 8e-7}],
                           lr=8e-6)

    # lr_decay=0.5 by 8step
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # Create trainer
    trainer = SLTTrDiMPTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    # Load pretrained model
    net = actor.net.module if settings.multi_gpu else actor.net
    checkpoint_dict = loading.torch_load_legacy('{$PATH_TO_YOUR_MODELS}/trdimp.pth')
    net.load_state_dict(checkpoint_dict['net'])

    # Train
    trainer.train(settings.num_epoch, load_latest=True, fail_safe=True)


def get_tracker_params(settings):
    ############ Tracker
    params = TrackerParams()
    params.use_gpu = True
    params.image_sample_size = 22*16 # 352
    params.search_area_scale = 6
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.5

    # Learning parameters
    params.sample_memory_size = 50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20

    # Net optimization params
    params.update_classifier = False
    params.net_opt_iter = settings.net_opt_iter
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1

    # transformer memory update
    params.transformer_skipping = 5
    params.transformer_memory_size = 20

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [10, -10, 45, -45],
                           'blur': [(3,1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)]}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # Advanced localization parameters
    params.score_preprocess = 'softmax'
    params.score_norm_meanmax = (-7, 7)
    params.advanced_localization = False
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.use_iou_net = True
    params.box_refinement_space = 'relative'
    params.iounet_augmentation = False      # Use the augmented samples to compute the modulation vector
    params.iounet_k = 3                     # Top-k average to estimate final box
    params.num_init_random_boxes = 9        # Num extra random boxes in addition to the classifier prediction
    params.box_jitter_pos = 0.1             # How much to jitter the translation for random boxes
    params.box_jitter_sz = 0.5              # How much to jitter the scale for random boxes
    params.maximal_aspect_ratio = 6         # Limit on the aspect ratio
    params.box_refinement_iter = 10          # Number of iterations for refining the boxes
    params.box_refinement_step_length = 2.5e-3 # 1   # Gradient step length in the bounding box refinement
    params.box_refinement_step_decay = 1    # Multiplicative step length decay (1 means no decay)

    # Params for generating proposals/labels for sequence-level training
    params.proposal_params = {'boxes_per_frame': 128, 'gt_sigma': (0.05, 0.05), 'proposal_sigma': [(0.05, 0.05), (0.5, 0.5)]}
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    params.label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}

    params.net = NetWithBackbone(net_path='trdimp_pth', image_format='rgb', use_gpu=True)

    return params