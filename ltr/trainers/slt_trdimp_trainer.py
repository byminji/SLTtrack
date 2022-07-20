import os
from collections import OrderedDict
from ltr.trainers import BaseTrainer
from ltr.admin.stats import AverageMeter, StatValue
from ltr.admin.tensorboard import TensorboardWriter
from torch.nn.utils import clip_grad_norm_
import torch
import time
import numpy as np


class SLTTrDiMPTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            tracker
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        # self.tracker = tracker

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

        self.miou_list = []

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            # Exploration in Sequence-Level Training to collect data from inference mode
            self.actor.eval()   ###### DO NOT trun off eval mode to make batchnorm operation act as same ######
            explore_result = self.actor.explore(data)
            # get inputs
            num_seq = len(data['num_frames'])

            # For logging
            stats = {}
            reward_record = []
            miou_record = []
            e_miou_record = []

            # Calculate reward tensor
            reward_tensor = torch.zeros(explore_result['action_tensor'].size())
            baseline_iou = explore_result['baseline_iou']
            explore_iou = explore_result['explore_iou']
            for seq_idx in range(num_seq):
                num_frames = data['num_frames'][seq_idx] - 1
                b_miou = torch.mean(baseline_iou[:num_frames, seq_idx])
                e_miou = torch.mean(explore_iou[:num_frames, seq_idx])
                miou_record.append(b_miou.item())
                e_miou_record.append(e_miou.item())

                b_reward = b_miou.item()
                e_reward = e_miou.item()
                iou_gap = e_reward - b_reward
                reward_record.append(iou_gap)
                reward_tensor[:num_frames, seq_idx] = iou_gap

            # Training mode
            cursor = 0
            bs = self.settings.num_seq_backward
            self.optimizer.zero_grad()
            while cursor < num_seq:
                model_inputs = {}
                model_inputs['epoch'] = self.epoch
                model_inputs['settings'] = self.settings

                model_inputs['template_images'] = explore_result['template_images'][:, cursor:cursor + bs].cuda()    # num_samplesxbsx3x352x352
                model_inputs['template_anno'] = explore_result['template_anno'][:, cursor:cursor + bs].cuda()        # num_samplesxbsx4
                model_inputs['template_label'] = explore_result['template_label'][:, cursor:cursor + bs].cuda()      # num_samplesxbsx22x22

                model_inputs['search_images'] = explore_result['search_images'][:, cursor:cursor + bs].cuda()  # (num_frames-1)xbsx3x352x352
                model_inputs['search_anno'] = explore_result['search_anno'][:, cursor:cursor + bs].cuda()      # (num_frames-1)xbsx4
                model_inputs['search_label'] = explore_result['search_label'][:, cursor:cursor + bs].cuda()    # (num_frames-1)xbsx23x23
                model_inputs['search_proposals'] = explore_result['search_proposals'][:, cursor:cursor + bs].cuda()     # (num_frames-1)xbsxnum_proposalx4
                model_inputs['proposal_density'] = explore_result['proposal_density'][:, cursor:cursor + bs].cuda() # (num_frames-1)xbsxnum_proposal
                model_inputs['gt_density'] = explore_result['gt_density'][:, cursor:cursor + bs].cuda()             # (num_frames-1)xbsxnum_proposal

                model_inputs['action_tensor'] = explore_result['action_tensor'][:, cursor:cursor + bs].cuda()  # (num_frames-1) x bs
                model_inputs['reward_tensor'] = reward_tensor[:, cursor:cursor + bs].cuda()                     # (num_frames-1) x bs

                # forward pass
                loss, stats_cur = self.actor(model_inputs)

                # backward pass
                loss.backward()

                # Log stats
                for key, val in stats_cur.items():
                    if key in stats:
                        stats[key] += val*(bs / num_seq)
                    else:
                        stats[key] = val*(bs / num_seq)
                cursor += bs

            # update weights
            if self.settings.clip_grad:
                grad_norm = clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_max_norm)
                stats['grad_norm'] = grad_norm
            else:
                parameters = list(filter(lambda p: p.grad is not None, self.actor.net.parameters()))
                total_norm = 0
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                stats['grad_norm'] = total_norm

            self.optimizer.step()

            # Logging
            miou = np.mean(miou_record)
            self.miou_list.append(miou)
            stats['reward'] = np.mean(reward_record)
            stats['e_mIoU'] = np.mean(e_miou_record)
            stats['mIoU'] = miou
            stats['mIoU10'] = np.mean(self.miou_list[-10:])
            stats['mIoU100'] = np.mean(self.miou_list[-100:])

            # Update & print statistics
            batch_size = num_seq*np.max(data['num_frames'])
            self._update_stats(stats, batch_size, loader)
            self._print_stats(i, loader, batch_size)
            torch.cuda.empty_cache()

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)