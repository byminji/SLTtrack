import numpy as np


class UniformReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = []
        self.cursor = 0
        self.capacity = capacity

    def add(self, x):
        if len(self.memory) < self.capacity:
            self.memory.append(x)
        else:
            self.memory[self.cursor] = x
            self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, n):
        if n > len(self.memory):
            raise ValueError('not enough data in replay memory')

        sampled_elems = np.random.choice(self.memory, n, replace=False)
        return sampled_elems


class PrioritizedReplayMemory:
    #e = 0.01
    #a = 0.6 # prioritization의 정도를 결정, 너무 높으면 overfitting 우려
    #beta = 0.4
    #beta_increment_per_sampling = 0.001

    def __init__(self, capacity, epsilon=0.01, alpha=0.6, beta=0.4, beta_inc=0.001):
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc

        self.priority = np.zeros(capacity)
        self.data = np.zeros(capacity, dtype=object)
        self.capacity = capacity
        self.num_data = 0
        self.cursor = 0

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, error, sample):
        p = self._get_priority(error)
        self.priority[self.cursor] = p
        self.data[self.cursor] = sample
        self.cursor += 1
        if self.cursor >= self.capacity:
            self.cursor = 0
        if self.num_data < self.capacity:
            self.num_data += 1

    def clear(self):
        self.priority = np.zeros(self.capacity)
        self.data = np.zeros(self.capacity, dtype=object)
        self.num_data = 0
        self.cursor = 0

    def get_all(self):
        return self.data[:self.num_data]

    def update(self, idxs, errors):
        p = self._get_priority(errors)
        self.priority[idxs] = p

    def sample(self, n):
        self.beta = np.min([1., self.beta + self.beta_inc])

        prob = self.priority / np.sum(self.priority)

        sampled_idxs = np.random.choice(len(prob), n, p=prob, replace=False)
        sampled_prob = prob[sampled_idxs]
        sampled_batch = self.data[sampled_idxs]

        is_weight = np.power(sampled_prob * self.num_data, -self.beta)
        is_weight /= is_weight.max()

        return sampled_batch, sampled_idxs, is_weight

    def __len__(self):
        return self.num_data
