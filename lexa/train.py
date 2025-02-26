import os
import functools
import tools
import tensorflow as tf
import numpy as np
import pickle
import pathlib
import off_policy
from density import RawKernelDensity
from dreamer import Dreamer, setup_dreamer, create_envs, count_steps, make_dataset, parse_dreamer_args
from normalizer import Normalizer, MeanStdNormalizer


class GCDreamer(Dreamer):
  def __init__(self, config, logger, dataset, kde=None, state_normalizer=None, goal_normalizer=None):
    if config.offpolicy_opt:
      self._off_policy_handler = off_policy.GCOffPolicyOpt(config)
    super().__init__(config, logger, dataset, kde, state_normalizer, goal_normalizer)
    self._should_expl_ep = tools.EveryNCalls(config.expl_every_ep)
    self.skill_to_use = tf.zeros([0], dtype=tf.float16)

  def get_one_time_skill(self):
    skill = self.skill_to_use
    self.skill_to_use = tf.zeros([0], dtype=tf.float16)
    return skill

  def _policy(self, obs, state, training, reset):
    # Choose the goal for the next episode
    # TODO this would be much more elegant if it was implemented in the training loop (can simulate a single episode)
    if state is None:
      if training and self._config.training_goals == 'batch':
        obs['image_goal'], obs['goal'] = self.sample_replay_goal(obs)
      obs['skill'] = self.get_one_time_skill()

      # The world model adds the observation to the state in this case

    if reset.any() and state is not None:
      # Replace the goal in the agent state at new episode
      # The actor always takes goal from the state, not the observation
      if training and self._config.training_goals == 'batch':
        state[0]['image_goal'], state[0]['goal'] = self.sample_replay_goal(obs)
        # normalize the goal in the beginning of episodes.
      else:
        state[0]['image_goal'] = tf.cast(obs['image_goal'], self._float) # / 255.0 - 0.5
        state[0]['goal'] = tf.cast(obs['goal'], self._float) # / 255.0 - 0.5
      state[0]['goal'] = self.goal_normalizer(training, state[0]['goal'])
      state[0]['skill'] = self.get_one_time_skill()
      
      # Toggle exploration
      self._should_expl_ep()
    
    obs = obs.copy()
    obs['goal'] = self.goal_normalizer(False, obs['goal']) # not the actual goal
    obs['image'] = self.state_normalizer(training, obs['image'])

    # TODO double check everything
    #return super()._policy(obs, state, training, reset, should_expl=self._should_expl_ep.value)
    #if not training:
    return super()._policy(obs, state, training, reset, should_expl=self._should_expl_ep.value)

  def sample_replay_goal(self, obs):
    """ Sample goals from replay buffer """
    # assert self._config.gc_input != 'state'
    random_batch = next(self._dataset)
    random_batch = self._wm.preprocess(random_batch)
    
    if self._config.env_type == 'vector':
      images = states = random_batch['achieved_goal']
    else:
      images = random_batch['image'] # does image only mean imaging.
      states = random_batch['state']
    if self._config.labelled_env_multiplexing:
      assert obs['env_idx'].shape[0] == 1
      env_ids = random_batch['env_idx'][:, 0]
      if tf.reduce_any(env_ids == obs['env_idx']):
        ids = np.nonzero(env_ids == obs['env_idx'])[0]
        images = tf.gather(images, ids)
        states = tf.gather(states, ids)
    
    random_goals = tf.reshape(images, (-1,) + tuple(images.shape[2:]))
    random_goal_states = tf.reshape(states, (-1,) + tuple(states.shape[2:]))
    # random_goals = tf.random.shuffle(random_goals)
    # only returned the first one out of thousands of goals.
    return random_goals[:obs['image_goal'].shape[0]], random_goal_states[:obs['image_goal'].shape[0]]


def process_eps_data(eps_data):
  # convert a list of dict to a dict of lists.
  keys = eps_data[0].keys()
  new_data = {}
  for key in keys:
    new_data[key] = np.array([eps_data[i][key] for i in range(len(eps_data))]).squeeze()
  return new_data

def main(logdir, config):
  logdir, logger = setup_dreamer(config, logdir)
  eval_envs, eval_eps, train_envs, train_eps, acts = create_envs(config, logger)
  
  print("setting the random seed to", config.seed)
  tools.set_global_seeds(config.seed)

  prefill = max(0, config.prefill - count_steps(config.traindir))
  # prefill = 300 # debug
  print(f'Prefill dataset ({prefill} steps).')
  random_agent = lambda o, d, s: ([acts.sample() for _ in d], s)
  tools.simulate(random_agent, train_envs, prefill)
  if count_steps(config.evaldir) == 0:
    tools.simulate(random_agent, eval_envs, episodes=1)
  logger.step = config.action_repeat * count_steps(config.traindir)

  print('Simulate agent.')
  train_dataset = make_dataset(train_eps, config)
  eval_dataset = iter(make_dataset(eval_eps, config))
  if config.env_type == 'vector':
    kde = RawKernelDensity(logdir, train_eps, config.time_limit, optimize_every=500, samples=10000,
        kernel='gaussian', bandwidth=0.1, normalize=True)  
  else:
    kde = None
  state_normalizer = Normalizer(MeanStdNormalizer())
  goal_normalizer = Normalizer(MeanStdNormalizer())
  agent = GCDreamer(config, logger, train_dataset, kde, state_normalizer, goal_normalizer)
  if (logdir / 'variables.pkl').exists():
    agent.load(logdir / 'variables.pkl')
    agent._should_pretrain._once = False

  pathlib.Path(logdir / "distance_func_logs_trained_model").mkdir(parents=True, exist_ok = True)

  state = None
  assert len(eval_envs) == 1
  while agent._step.numpy().item() < config.steps:
    logger.write()
    print('Start gc evaluation.')
    executions = []
    goals = []
    #rews_across_goals = []
    # num_goals = min(100, len(eval_envs[0].get_goals())) # 30
    num_eval = 30
    all_eps_data = []
    num_eval_eps = 1
    succ_count = 0
    for ep_idx in range(num_eval_eps):
      ep_data_across_goals = []
      for idx in range(num_eval):
        #eval_envs[0].set_goal_idx(idx)
        # randomly set the goals
        eval_policy = functools.partial(agent, training=False)
        sim_out = tools.simulate(eval_policy, eval_envs, episodes=1)
        obs, eps_data = sim_out[4], sim_out[6]

        ep_data_across_goals.append(process_eps_data(eps_data))
        if config.first_success:
          succ_count += np.any(np.array([o['reward'] for o in obs]) == 0.)
        else:
          succ_count += (obs[-1]['reward'] == 0.)
        # no video will be produced
        # video = eval_envs[0]._convert([t['image'] for t in eval_envs[0]._episode])
        # executions.append(video[None])
        # goals.append(obs[0]['image_goal'][None])

      all_eps_data.append(ep_data_across_goals)

    # if ep_idx == 0:
    #   executions = np.concatenate(executions, 0)
    #   goals = np.stack(goals, 0)
    #   goals = np.repeat(goals, executions.shape[1], 1)
    #   gc_video = np.concatenate([goals, executions], -3)
    #   agent._logger.video(f'eval_gc_policy', gc_video)
    #   logger.write()

    with pathlib.Path(logdir / ("distance_func_logs_trained_model/step_"+str(logger.step)+".pkl") ).open('wb') as f:
      pickle.dump(all_eps_data, f)

    with pathlib.Path(logdir / ('Success.csv')).open("a+") as f:
      f.write(str(logger.step) + ' ' + str(succ_count/num_eval) + '\n')

    if config.sync_s3:
      os.system('aws s3 sync '+str(logdir)+ ' s3://goalexp2021/research_code/goalexp_data/'+str(logdir))

    if not config.training:
        continue
    print('Start training.')
    # the state is a tuple
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
