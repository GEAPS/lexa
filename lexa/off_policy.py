import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd

import networks
import tools
import models
import numpy as np
from tools import get_data_for_off_policy_training

class GCOffPolicyOpt(tools.Module):
  def __init__(self, config):

    self._config = config
    from_images = not self._config.offpolicy_use_embed and (config.env_type == 'image')
    self.actor = networks.GC_Actor(config.num_actions, units = config.gc_actor_units, from_images=from_images)
  
    kw = dict(wd=config.weight_decay, opt=config.opt)
    self._actor_opt = tools.Optimizer(
        'actor', config.actor_lr, config.opt_eps, config.actor_grad_clip, **kw)
    self.action_scale = config.action_scale
  
  # def train_gcbc(self, obs, prev_actions, goals, achieved_goals, training_goals):
  def train_gcbc(self, obs, data, env_type):
    metrics = {}
    actions = data['action']
    goals = data['goal']
    if env_type == 'vector':
      next_achieved_goals = data['achieved_goal'][:, 1:]
    else:
      next_achieved_goals = obs[:, 1:]
    # In this task, we need to use achieved goals instead.
    # If we adopt GCRL for the policy optimization, it would also require both next_state and rewards.
    s_t, a_t = get_data_for_off_policy_training(obs[:,:-1], actions[:,1:], next_achieved_goals, goals[:, :-1],
                                                self._config.relabel_mode, relabel_fraction=1.0)
    with tf.GradientTape() as tape:
      if self._config.gcbc_distance_weighting:
        raise NotImplementedError
      else:
        pred_action = self.actor(s_t)
        loss = tf.reduce_mean((pred_action - a_t)**2)
   
    metrics.update(self._actor_opt(tape, loss, self.actor))
    metrics = {'replay_' + k: v for k, v in metrics.items()}
    return metrics
  
  def act(self, inputs, training=False):
    actions = self.actor(inputs) * self.action_scale
    return actions
    