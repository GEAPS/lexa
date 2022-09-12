import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.special import entr
import pathlib

# TODO (lisheng) Change the way of entropy esitmation.
# TODO (lisheng) Check the parameters
class RawKernelDensity(object):
  """
  A KDE-based density model for raw items in the replay buffer (e.g., states/goals).
  """
  def __init__(self, target_dir, train_eps, eps_length, optimize_every=10, samples=10000, kernel='gaussian', bandwidth=0.1, normalize=True):
    self.target_dir = target_dir
    self.step = 0
    # the dataset collected during training.
    self.train_eps = train_eps
    self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    self.optimize_every = optimize_every
    self.samples = samples
    self.kernel = kernel
    self.bandwidth = bandwidth
    self.normalize = normalize
    self.kde_sample_mean = 0.
    self.kde_sample_std = 1.
    self.fitted_kde = None
    self.ready = False
    self.num_samples_per_episode = 5 # sample 10 points from each episode.
    self.episode_length = eps_length

  def optimize(self, force=False):
    # Get samples here.
    self.step +=1

    if force or (self.step % self.optimize_every == 0 and len(self.train_eps)):
      self.ready = True
      num_episode_samples = self.samples // self.num_samples_per_episode
      all_episodes = list(self.train_eps.keys())
      sample_episode_idxs = np.random.randint(len(all_episodes), size=num_episode_samples)
      episode_sample_idxs = np.random.randint(self.episode_length, size=self.samples)
      goal_samples = []
      for i, e_idx in enumerate(sample_episode_idxs):
        eps_key = all_episodes[e_idx]
        s_idxs = episode_sample_idxs[self.num_samples_per_episode*i: self.num_samples_per_episode*(i+1)]
        goal_samples.append(self.train_eps[eps_key]['image'][s_idxs]) # TODO (lisheng) Update to achieved goals 

      kde_samples = np.concatenate(goal_samples, axis=0)
      # get_samples and only get the image as the dataset.
      #og_kde_samples = kde_samples

      if self.normalize:
        self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
        self.kde_sample_std  = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
        kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std

      #if self.item == 'ag' and hasattr(self, 'ag_interest') and self.ag_interest.ready:
      #  ag_weights = self.ag_interest.evaluate_disinterest(og_kde_samples)
      #  self.fitted_kde = self.kde.fit(kde_samples, sample_weight=ag_weights.flatten())
      #else:
      self.fitted_kde = self.kde.fit(kde_samples)

      # Now also log the entropy
      if self.step % 500 == 0:
        # Scoring samples is a bit expensive, so just use 1000 points
        num_samples = 1000
        s = self.fitted_kde.sample(num_samples)
        entropy = -self.fitted_kde.score(s)/num_samples + np.log(self.kde_sample_std).sum()
        with pathlib.Path(self.target_dir / ('entropy.csv')).open("a+") as f:
          f.write(str(self.step) + ' ' + str(entropy) + '\n')

  def evaluate_log_density(self, samples):
    assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
    return self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )

  def evaluate_elementwise_entropy(self, samples, beta=0.):
    """ Given an array of samples, compute elementwise function of entropy of the form:

        elem_entropy = - (p(samples) + beta)*log(p(samples) + beta)

    Args:
      samples: 1-D array of size N
      beta: float, offset entropy calculation

    Returns:
      elem_entropy: 1-D array of size N, elementwise entropy with beta offset
    """
    assert self.ready, "ENSURE READY BEFORE EVALUATING ELEMENT-WISE ENTROPY"
    log_px = self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )
    px = np.exp(log_px)
    elem_entropy = entr(px + beta)
    return elem_entropy
