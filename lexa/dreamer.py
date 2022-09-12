import argparse
import collections
import functools
import os
import pathlib
import resource
import sys
import warnings
import pickle

import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import exploration as expl
import models
import tools
import wrappers
import gcdreamer_wm, gcdreamer_imag
import envs

# relabel the goal in the policy. 
class Dreamer(tools.Module):

  def __init__(self, config, logger, dataset, kde=None):
    self._config = config
    self._logger = logger
    self._float = prec.global_policy().compute_dtype
    self._should_log = tools.Every(config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    with tf.device('cpu:0'):
      self._step = tf.Variable(count_steps(config.traindir), dtype=tf.int64)
    # Schedules.
    config.actor_entropy = (
        lambda x=config.actor_entropy: tools.schedule(x, self._step))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    self._dataset = iter(dataset)
    # the fundamental world model would convert to flat version.
    if 'gcdreamer' in config.task_behavior:
      self._wm = gcdreamer_wm.GCWorldModel(self._step, config)
    else:
      self._wm = models.WorldModel(self._step, config)
    self._task_behavior = dict(
        dreamer=models.ImagBehavior(config, self._wm, config.behavior_stop_grad),
        gcdreamer=gcdreamer_imag.GCDreamerBehavior(config, self._wm, config.behavior_stop_grad),
    )[config.task_behavior]
    reward = lambda f, s, a: self._wm.heads['reward'](f).mode()
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(config),
        plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
    )[config.expl_behavior]()
    # Train step to initialize variables including optimizer statistics.
    self.kde = kde
    self._train(next(self._dataset))

  def __call__(self, obs, reset, state=None, training=True):
    # it would be important to view the codes of simulate. 
    # train + action decision.
    step = self._step.numpy().item()
    if self._should_reset(step):
      state = None
    if state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      state = tf.nest.map_structure(lambda x: x * mask, state)
    if training and self._should_train(step):
      steps = (
          self._config.pretrain if self._should_pretrain()
          else self._config.train_steps)
      for _ in range(steps):
        # pre-train or train
        _data = next(self._dataset)
        start, feat = self._train(_data)

      if self._should_log(step):
        for name, mean in self._metrics.items():
          self._logger.scalar(name, float(mean.result()))
          mean.reset_states()
        if self._config.env_type == 'image':
          openl = self._wm.video_pred(next(self._dataset))
          self._logger.video('train_openl', openl)
          self._logger.write(fps=True)

    if training:
      # also make actions.
      if self.kde is not None:
        self.kde.optimize()
      
      action, state = self._policy(obs, state, training, reset)
      self._step.assign_add(len(reset))
      self._logger.step = self._config.action_repeat \
          * self._step.numpy().item()
      return action, state

    else:
      action, state, reward = self._policy(obs, state, training, reset)
      return action, state, reward

  @tf.function
  def _policy(self, obs, state, training, reset, should_expl=False):

    obs = self._wm.preprocess(obs)
    feat, latent = self._wm.get_init_feat(
      obs, state, sample=self._config.collect_dyn_sample and not self._config.eval_state_mean)

    if not training:
      goal = self._wm.get_goal(obs, training=False)
      # figure out what's the offpolicy opt used for?
      # what's the reward used for if not training?
      if self._config.offpolicy_opt:
        if self._config.offpolicy_use_embed:
          # use the intermediate representation from the encoder.
          action = self._off_policy_handler.actor(tf.concat([self._wm.encoder(obs), goal], axis = -1))
        else:
          # stack on the channel dim.
          action = self._off_policy_handler.actor(tf.concat([obs['image'], obs['image_goal']], axis = -1))
        reward = tf.zeros((action.shape[0],1), dtype=tf.float32)
      else:
        action = self._task_behavior.act(feat, obs, latent).mode()
        actor_inp = tf.concat([feat, goal], -1) # actor input
        pad = lambda x : tf.expand_dims(x, 0)
        # image goal will be replaced by the vector goal.
        # the agent needs to calculate the reward immediately.
        # only supported by GCDreamerBehaviour.
        reward = self._task_behavior._gc_reward(pad(actor_inp), latent, action, pad(obs['image_goal']))
    elif self._should_expl(self._step) or should_expl:
      action = self._expl_behavior.act(feat, obs, latent).sample()
    elif self._config.offpolicy_opt:
      if self._config.offpolicy_use_embed:
        action = self._off_policy_handler.actor(tf.concat([self._wm.encoder(obs), self._wm.encoder({'image': obs['image_goal']})], axis = -1))
      else:
        action = self._off_policy_handler.actor(tf.concat([obs['image'], latent['image_goal']], axis = -1))
        # action = self._off_policy_handler.actor(tf.concat([obs['image'], obs['image_goal']], axis = -1))
    else:
      # otherwise, use the greedy behavior.
      action = self._task_behavior.act(feat, obs, latent).sample()
    if self._config.actor_dist == 'onehot_gumble':
      action = tf.cast(
          tf.one_hot(tf.argmax(action, axis=-1), self._config.num_actions),
          action.dtype)
    action = self._exploration(action, training)
    state = (latent, action)

    # state has the enough information for the world model to decide the future.
    if training:
      return action, state
    else:
      return action, state, reward

  def _exploration(self, action, training):
    amount = self._config.expl_amount if training else self._config.eval_noise
    if amount == 0:
      return action
    amount = tf.cast(amount, self._float)
    if 'onehot' in self._config.actor_dist:
      probs = amount / self._config.num_actions + (1 - amount) * action
      return tools.OneHotDist(probs=probs).sample()
    else:
      # amout denotes the variance here.
      return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
    raise NotImplementedError(self._config.action_noise)

  @tf.function
  def _get_imag_data(self, data, start):
    goal = self._wm.get_goal(self._wm.preprocess(data), training=True)
    imag_feat, imag_state, imag_action = self._task_behavior._imagine(
      start, self._task_behavior.actor, self._config.imag_horizon, goal=goal)
    return imag_feat, imag_action, goal

  @tf.function
  def _train(self, data):
    # the data depends on the dataset.
    # many different training behaviors
    metrics = {}
    embed, post, feat, kl, mets = self._wm.train(data)
    metrics.update(mets)
    start = post
    assert not self._config.pred_discount


    # GCSL won't train this part.
    # If we will train in the RL manner with the environment rewards, we need to change the way of training.
    # TODO(lisheng) Decode the imagined states.
    # Calculate rewards based on current states and decoded states.
    # NOTE The world model is used to provide on-policy rewards like N-steps return. 
    if self._config.imag_on_policy:
      # task policy will also be trained.
      metrics.update(self._task_behavior.train(start, obs=data)[-1])

    if self._config.gc_reward == 'dynamical_distance' and self._config.dd_train_off_policy:
      metrics.update(self._task_behavior.train_dd_off_policy(self._wm.encoder(self._wm.preprocess(data))))

    if self._config.expl_behavior != 'greedy':
      mets = self._expl_behavior.train(start, feat, embed, kl)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})

    if self._config.gcbc:
      _data = self._wm.preprocess(data)
      obs = self._wm.encoder(self._wm.preprocess(data)) if self._config.offpolicy_use_embed else _data['image']
      metrics.update(self._off_policy_handler.train_gcbc(obs, _data['action']))

    for name, value in metrics.items():
      self._metrics[name].update_state(value)

    return start, feat

def count_steps(folder):
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_dataset(episodes, config):
  example = episodes[next(iter(episodes.keys()))]
  types = {k: v.dtype for k, v in example.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in example.items()}
  generator = lambda: tools.sample_episodes(
      episodes, config.batch_length, config.oversample_ends)
  dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  dataset = dataset.batch(config.batch_size, drop_remainder=True)
  dataset = dataset.prefetch(10)
  return dataset


# Four environments to set up
# 1. AntMaze 2. FPP 3. FSK 4. PointMaze
# reqired the environment
# the number of maximum steps would be fixed,

def make_env(config, logger, mode, train_eps, eval_eps, use_goal_idx=False, log_per_goal=False):
  
  if 'dmc' in config.task:
    suite, task = config.task.split('_', 1)
    env = envs.DmcEnv(task, config.size, config.action_repeat, use_goal_idx, log_per_goal)
    env = wrappers.NormalizeActions(env)
  
  elif config.task == 'robobin':
    env = envs.RoboBinEnv(config.action_repeat, use_goal_idx, log_per_goal)
  
  elif config.task == 'kitchen':
    env = envs.KitchenEnv(config.action_repeat, use_goal_idx, log_per_goal)

  elif config.task == 'joint':
   
    kitchen_env = envs.KitchenEnv(config.action_repeat, use_goal_idx, False)
    robobin_env = envs.RoboBinEnv(config.action_repeat, use_goal_idx, False)
    # task - where is task defined.
    dmc_envs = list(envs.DmcEnv(task, config.size, config.action_repeat, use_goal_idx, log_per_goal) for task in ['walker_walk', 'quadruped_run'])

    env = envs.MultiplexedEnv([kitchen_env, robobin_env] + dmc_envs, config.action_repeat, config.size, use_goal_idx, log_per_goal)
    # pad actions.
    env = envs.PadActions(env, list(_env.action_space for _env in env.envs))
    env = wrappers.NormalizeActions(env)
  elif config.task == 'pointmaze':
    # TODO(lisheng) Add state normalizer to the policy.
    # TODO(lisheng) Check any scale to the final actions.
    env = envs.PointMaze2D(env_max_steps=50, test=use_goal_idx)

  # elif config.task == 'antmaze':
  #   env = AntMazeEnv(test=use_goal_idx)
  # elif "stack" or "pickplace" in config.task:
  #   env_type, external, internal = args.env.split('_')
  #   if external.lower() == 'obj':
  #     external = GoalTypes.OBJ
  #   else:
  #     raise NotImplementedError(external)
    
  #   if internal.lower() == 'obj':
  #     internal = GoalTypes.OBJ
  #   else:
  #     raise NotImplementedError(external)
    
  #   n_blocks = 0
  #   range_min = None # For pickplace
  #   range_max = None # For pickplace

  #   # hard and other parameters - pp_min_air & pp_max_air - pp_in_air_percentage.
  #   if 'pickplace' in env_type:
  #     Env = PickPlaceEnv
  #     n_blocks = config.pp_in_air_percentage
  #     range_min = config.pp_min_air # THIS IS THE MINIMUM_AIR
  #     range_max = config.pp_max_air # THIS IS THE MINIMUM_AIR
  #   elif 'stack' in env_type:
  #     Env = StackEnv
  #     n_blocks = int(env_type.replace('stack', ''))
    
  #   env_fn = Env(max_step=50, internal_goal = internal, external_goal = external, mode=0, 
  #               per_dim_threshold=0, hard=args.hard, distance_threshold=0, n = n_blocks,
  #               range_min=range_min, range_max=range_max)
  else:
    raise NotImplementedError(config.task)
  env = wrappers.TimeLimit(env, config.time_limit)
  callbacks = [functools.partial(
      process_episode, config, logger, mode, train_eps, eval_eps)]
  env = wrappers.CollectDataset(env, callbacks)
  env = wrappers.RewardObs(env)
  return env

def process_episode(config, logger, mode, train_eps, eval_eps, episode):
  directory = dict(train=config.traindir, eval=config.evaldir)[mode]
  cache = dict(train=train_eps, eval=eval_eps)[mode]
  filename = tools.save_episodes(directory, [episode])[0]
  length = len(episode['reward']) - 1
  score = float(episode['reward'].astype(np.float64).sum())
  video = episode['image']
  if mode == 'eval':
    cache.clear()
  if mode == 'train' and config.dataset_size:
    total = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
      if total <= config.dataset_size - length:
        total += len(ep['reward']) - 1
      else:
        del cache[key]
    logger.scalar('dataset_size', total + length)
  cache[str(filename)] = episode
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  logger.scalar(f'{mode}/return', score)
  logger.scalar(f'{mode}/length', length)
  logger.scalar(f'{mode}/episodes', len(cache))
  for key in filter(lambda k: 'metric_' in k, episode):

    metric_min =  np.min(episode[key].astype(np.float64))
    metric_max =  np.max(episode[key].astype(np.float64))
    metric_mean = float(episode[key].astype(np.float64).mean())
    metric_final = float(episode[key].astype(np.float64)[-1])
    key = key.replace('metric_', '')

    logger.scalar(f'{mode}/min_{key}', metric_min)
    logger.scalar(f'{mode}/max_{key}', metric_max)
    logger.scalar(f'{mode}/mean_{key}', metric_mean)
    logger.scalar(f'{mode}/final_{key}', metric_final)
  logger.write()

def setup_dreamer(config, logdir):
  logdir = pathlib.Path(logdir).expanduser()
  config.traindir = config.traindir or logdir / 'train_eps'
  config.evaldir = config.evaldir or logdir / 'eval_eps'
  config.steps //= config.action_repeat
  config.eval_every //= config.action_repeat
  config.log_every //= config.action_repeat
  config.time_limit //= config.action_repeat
  config.act = getattr(tf.nn, config.act)
  if config.debug:
    tf.config.experimental_run_functions_eagerly(True)
  if config.gpu_growth:
    # message = 'No GPU found. To actually train on CPU remove this assert.'
    # assert tf.config.experimental.list_physical_devices('GPU'), message
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    prec.set_policy(prec.Policy('mixed_float16'))
  print('Logdir', logdir)
  logdir.mkdir(parents=True, exist_ok=True)
  step = count_steps(config.traindir)
  logger = tools.Logger(logdir, config.action_repeat * step)
  # Save config
  tools.save_cmd(logdir)
  tools.save_git(logdir)
  with open(logdir / 'config.yaml', 'w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)
  return logdir, logger


def create_envs(config, logger):
  print('Create envs.')
  if config.offline_traindir:
    directory = config.offline_traindir.format(**vars(config))
  else:
    directory = config.traindir
  train_eps = tools.load_episodes(directory, limit=config.dataset_size)
  if config.offline_evaldir:
    directory = config.offline_evaldir.format(**vars(config))
  else:
    directory = config.evaldir
  eval_eps = tools.load_episodes(directory, limit=1)
  make = functools.partial(make_env, config, logger, train_eps=train_eps, eval_eps=eval_eps)
  train_envs = [make('train', log_per_goal=True) for _ in range(config.envs)]
  eval_envs = [make('eval', use_goal_idx=True, log_per_goal=config.test_log_per_goal) for _ in range(config.envs)]
  acts = train_envs[0].action_space
  config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]
  return eval_envs, eval_eps, train_envs, train_eps, acts


def parse_dreamer_args():
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', required=True)
  parser.add_argument('--logdir', required=True)
  args, remaining = parser.parse_known_args()
  configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / 'configs.yaml').read_text())
  config_ = {}
  for name in args.configs:
    config_.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in config_.items():
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  return args, parser.parse_args(remaining)


def main(logdir, config):
  logdir, logger = setup_dreamer(config, logdir)
  eval_envs, eval_eps, train_envs, train_eps, acts = create_envs(config, logger)
  prefill = max(0, config.prefill - count_steps(config.traindir))
  print(f'Prefill dataset ({prefill} steps).')
  random_agent = lambda o, d, s: ([acts.sample() for _ in d], s)
  tools.simulate(random_agent, train_envs, prefill)
  tools.simulate(random_agent, eval_envs, episodes=1)
  logger.step = config.action_repeat * count_steps(config.traindir)

  print('Simulate agent.')
  train_dataset = make_dataset(train_eps, config)
  eval_dataset = iter(make_dataset(eval_eps, config))
  agent = Dreamer(config, logger, train_dataset)
  if (logdir / 'variables.pkl').exists():
    agent.load(logdir / 'variables.pkl')
    agent._should_pretrain._once = False

  state = None
  while agent._step.numpy().item() < config.steps:
    logger.write()
    print('Start evaluation.')
    video_pred = agent._wm.video_pred(next(eval_dataset))
    logger.video('eval_openl', video_pred)
    eval_policy = functools.partial(agent, training=False)
    tools.simulate(eval_policy, eval_envs, episodes=1)
    print('Start training.')
    state = tools.simulate(agent, train_envs, config.eval_every, state=state)
    agent.save(logdir / 'variables.pkl')
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  args, remaining = parse_dreamer_args()
  main(args.logdir, remaining)
