import gym
from replays.core.shared_buffer import SharedMemoryTrajectoryBuffer as Buffer
import numpy as np

class OnlineHERBuffer(object):

  def __init__(
      self,
      env,
      config,
      state_normalizer=None,
      goal_normalizer=None):
    """
    Buffer that does online hindsight relabeling.
    Replaces the old combo of ReplayBuffer + HERBuffer.
    """
    self.size = None
    self.goal_space = None
    self.buffer = None
    self.save_buffer = None
    self.config = config
    self.env = env
    self.state_normalizer = state_normalizer
    self.goal_normalizer = goal_normalizer

    self.size = int(self.config.replay_size)

    env = self.env
    if type(env.observation_space) == gym.spaces.Dict:
      observation_space = env.observation_space.spaces["observation"]
      self.goal_space = env.observation_space.spaces["desired_goal"]
    else:
      observation_space = env.observation_space

    items = [("state", observation_space.shape),
             ("action", env.action_space.shape), ("reward", (1,)),
             ("next_state", observation_space.shape), ("done", (1,))]

    if self.goal_space is not None:
      items += [("previous_ag", self.goal_space.shape), # for reward shaping
                ("ag", self.goal_space.shape), # achieved goal
                ("bg", self.goal_space.shape), # behavioral goal (i.e., intrinsic if curious agent)
                ("dg", self.goal_space.shape)] # desired goal (even if ignored behaviorally)

    self.buffer = Buffer(self.size, items)
    # self._subbuffers = [[] for _ in range(self.env.num_envs)]
    # self.n_envs = self.env.num_envs
    self.n_envs = 1
    self._subbuffers = [[] for _ in range(self.n_envs)]

    # TODO (lisheng) Add a new hyper-parameter
    self.rel, self.fut, self.act, self.ach, self.beh = parse_hindsight_mode(self.config.her)

  # TODO(lisheng) Modify to process the whole episode. 
  # observation - achieved_goal - desired_goal - state - done - reward
  # The env should provide accesses to the API and
  def process_episode(self, eps):
    state = eps["image"][:-1]
    next_state = eps["image"][1:]
    # reward here is actually not meaningful.
    # which will be recalculated.
    action = eps["action"][1:]
    previous_achieved = eps["achieved_goal"][:-1]
    achieved = eps["achieved_goal"][1:]
    desired = eps["goal"][1:]
    # the bevavioral will be obtained from the agent state. 
    behavioral = eps["behavioral"][1:]
    # is done useful.
    # after extracting the data, I should also construct the policy module.
    # TODO(lisheng) Verfify the dataset at first.
    done = np.expand_dims(eps["done"][1:], 1)
    reward = np.expand_dims(eps["reward"][1:], 1)


    trajectory = [state, action, reward, next_state,
                  done, previous_achieved, achieved,
                  behavioral, desired]
    self.buffer.add_trajectory(*trajectory)

  def sample(self, batch_size):
    batch_idxs = np.random.randint(self.buffer.size, size=batch_size)

    if self.goal_space:
      # not sure whether this part will work.
      has_config_her = self.config.her
      
      if has_config_her:

        if len(self.buffer) > self.config.future_warm_up:
          fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = np.random.multinomial(
              batch_size, [self.fut, self.pst, self.act, self.ach, self.beh, self.rel])
        else:
          fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size  = batch_size, 0, 0, 0, 0

        fut_idxs, act_idxs, ach_idxs, beh_idxs, real_idxs = np.array_split(batch_idxs, 
          np.cumsum([fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size]))

          
        states, actions, rewards, next_states, dones, previous_ags, ags, goals, _ =\
            self.buffer.sample(real_batch_size, batch_idxs=real_idxs)

        states_fut, actions_fut, _, next_states_fut, dones_fut, previous_ags_fut, ags_fut, _, _, goals_fut =\
            self.buffer.sample_future(fut_batch_size, batch_idxs=fut_idxs)

        # Sample the actual batch
        states_act, actions_act, _, next_states_act, dones_act, previous_ags_act, ags_act, _, _, goals_act =\
          self.buffer.sample_from_goal_buffer('dg', act_batch_size, batch_idxs=act_idxs)

        # Sample the achieved batch
        states_ach, actions_ach, _, next_states_ach, dones_ach, previous_ags_ach, ags_ach, _, _, goals_ach =\
          self.buffer.sample_from_goal_buffer('ag', ach_batch_size, batch_idxs=ach_idxs)

        # Sample the behavioral batch
        states_beh, actions_beh, _, next_states_beh, dones_beh, previous_ags_beh, ags_beh, _, _, goals_beh =\
          self.buffer.sample_from_goal_buffer('bg', beh_batch_size, batch_idxs=beh_idxs)

        # Concatenate the five
        states = np.concatenate([states, states_fut, states_act, states_ach, states_beh], 0)
        actions = np.concatenate([actions, actions_fut, actions_act, actions_ach, actions_beh], 0)
        ags = np.concatenate([ags, ags_fut, ags_act, ags_ach, ags_beh], 0)
        goals = np.concatenate([goals, goals_fut, goals_act, goals_ach, goals_beh], 0)
        next_states = np.concatenate([next_states, next_states_fut, next_states_act, next_states_ach,\
           next_states_beh], 0)

        rewards = self.env.compute_reward(ags, goals, {'s':states, 'ns':next_states}).reshape(-1, 1).astype(np.float32)

        if self.config.first_visit_success:
          dones = np.round(rewards + 1.)
        else:
          dones = np.zeros_like(rewards, dtype=np.float32)

      else:
        # Uses the original desired goals
        states, actions, rewards, next_states, dones, _ , _, _, goals =\
                                                    self.buffer.sample(batch_size, batch_idxs=batch_idxs)

      
      if self.state_normalizer is not None:
        states = self.state_normalizer(False, states).astype(np.float32)
        next_states = self.state_normalizer(False, next_states).astype(np.float32)
      if self.goal_normalizer is not None:
        goals = self.state_normalizer(False, goals).astype(np.float32)
      states = np.concatenate((states, goals), -1)
      next_states = np.concatenate((next_states, goals), -1)

      gammas = self.config.discount * (1.-dones)

    else:
      raise ValueError("The env does not have goal space.")

    return (states, actions, rewards, next_states, gammas)

  def __len__(self):
    return len(self.buffer)

  # def save(self, save_folder):
  #   if self.config.save_replay_buf or self.save_buffer:
  #     state = self.buffer._get_state()
  #     with open(os.path.join(save_folder, "{}.pickle".format(self.module_name)), 'wb') as f:
  #       pickle.dump(state, f)

  # def load(self, save_folder):
  #   load_path = os.path.join(save_folder, "{}.pickle".format(self.module_name))
  #   if os.path.exists(load_path):
  #     with open(load_path, 'rb') as f:
  #       state = pickle.load(f)
  #     self.buffer._set_state(state)
  #   else:
  #     self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='cyan')
  #     self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='red')
  #     self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='yellow')

def parse_hindsight_mode(hindsight_mode : str):
  if 'future_' in hindsight_mode:
    _, fut = hindsight_mode.split('_')
    rel = 1. / (1. + float(fut))
    fut = float(fut) / (1. + float(fut))
    act = 0.
    ach = 0.
    beh = 0.
  elif 'futureactual_' in hindsight_mode:
    _, fut, act = hindsight_mode.split('_')
    non_hindsight_frac = 1. / (1. + float(fut) + float(act))
    rel = non_hindsight_frac
    fut = float(fut) * non_hindsight_frac
    act = float(act) * non_hindsight_frac
    ach = 0.
    beh = 0.
    clo = 0
  elif 'futureachieved_' in hindsight_mode:
    _, fut, ach = hindsight_mode.split('_')
    non_hindsight_frac = 1. / (1. + float(fut) + float(ach))
    rel = non_hindsight_frac
    fut = float(fut) * non_hindsight_frac
    act = 0.
    ach = float(ach) * non_hindsight_frac
    beh = 0.
  elif 'rfaa_' in hindsight_mode:
    _, real, fut, act, ach = hindsight_size
    fut = float(fut) / denom
    act = float(act) / denom
    ach = float(ach) / denom
    beh = 0.
    clo = 0
  elif 'rfaab_' in hindsight_mode:
    _, real, fut, act, ach, beh = hindsight_mode.split('_')
    denom = (float(real) + float(fut) + float(act) + float(ach) + float(beh))
    rel = float(real) / denom
    fut = float(fut) / denom
    act = float(act) / denom
    ach = float(ach) / denom
    beh = float(beh) / denom
  else:
    rel = 1.
    fut = 0.
    act = 0.
    ach = 0.
    beh = 0.

  return rel, fut, act, ach, beh

