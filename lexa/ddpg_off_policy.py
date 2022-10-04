from math import gamma
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd

import networks
import tools
import models
from tf_agents.utils import common, object_identity
import numpy as np
from tools import get_data_for_off_policy_training

class DDPGOpt(tools.Module):
  def __init__(self, config):
    self._config = config
    # You should also have value functions.
    self.actor = networks.GC_Actor(config.num_actions, units = config.gc_actor_units, from_images=False)
    self.actor_target = networks.GC_Actor(config.num_actions, units = config.gc_actor_units, from_images=False)
    # TODO(lisheng) Add new parameters to the config.
    # not sure whether we should use the same layer init - orthogonal
    self.critic = networks.GC_Critic(units = config.gc_critic_units, from_images=False)
    self.critic_target = networks.GC_Critic(units = config.gc_critic_units, from_images=False)
  
    kw = dict(wd=config.weight_decay, opt=config.opt)
    self._actor_opt = tools.Optimizer(
        'actor', config.gc_actor_lr, config.opt_eps, config.actor_grad_clip, **kw)
    self._critic_opt = tools.Optimizer(
      'critic', config.gc_critic_lr, config.opt_eps, config.value_grad_clip, **kw)
    self.action_scale = config.action_scale

    # actions are not scaled in the environments. (super important in the env.2p)
    # also needs to scale to the action scales.
    self.steps = 0
    self.num_actions = config.num_actions
    self.epsilon = config.epsilon
  
  # Shall I use action spaces instead
  def act(self, inputs, training=False):
    # use the same epsilon greedy methods.
    # I found mrl didn't use noises like gaussian noises.
    actions = self.actor(inputs)
    if training: #and tf.random.uniform(shape=[], maxval=1.) < self.epsilon:
      dtype = inputs.dtype
      noises = tf.random.normal(actions.shape, stddev=0.1, dtype=dtype)
      actions = actions + noises
      # actions = tf.random.uniform(shape=inputs.shape[:-1] + (self.num_actions,), minval=-1., maxval=1., dtype=dtype) * self.action_scale
    # else:
    return actions  * self.action_scale

  
  # def train_gcbc(self, obs, prev_actions, goals, achieved_goals, training_goals):
  # obtain data from her buffer.
  def train(self, states, actions, rewards, next_states, gammas):
    metrics = {}
    # trainable_critic_variables = list(object_identity.ObjectIdentitySet(
    #   self.critic.trainable_variables))
    
    # trainable_actor_variables = list(object_identity.ObjectIdentitySet(
    #   self.actor.trainable_variables))
    with tf.GradientTape(watch_accessed_variables=True) as tape:
      # tape.watch(trainable_critic_variables)
      # TODO(lisheng) the input to the critic should be adjusted.
      # actions have been scaled.
      q_next = self.critic_target(next_states, self.actor_target(next_states)*self.action_scale)
      target =  (rewards + gammas * q_next)
      q = self.critic(states, actions)
      critic_loss = tf.reduce_mean((q - target)**2)
    # import ipdb; ipdb.set_trace()
    metrics.update(self._critic_opt(tape, critic_loss, self.critic))

    with tf.GradientTape(watch_accessed_variables=True) as tape:
      # tape.watch(trainable_actor_variables)
      a = self.actor(states) * self.action_scale
      # noises was not used in mrl's code.
      actor_loss = tf.reduce_mean(-self.critic(states, a))
      if self._config.action_l2_regularization != 0:
        actor_loss += self._config.action_l2_regularization * tf.reduce_mean((a/self.action_scale)**2)
    
    metrics.update(self._actor_opt(tape, actor_loss, self.actor))

    # tau = 1 is completely update.
    self.steps += 1
    if self.steps % self._config.target_network_update_freq == 0:
      common.soft_variables_update(self.critic.variables, self.critic_target.variables,
                                  self._config.target_network_update_frac)
      common.soft_variables_update(self.actor.variables, self.actor_target.variables,
                                  self._config.target_network_update_frac)
   
    # metrics.update(self._actor_opt(tape, loss, self.actor))
    # metrics = {'replay_' + k: v for k, v in metrics.items()}
    return metrics
    