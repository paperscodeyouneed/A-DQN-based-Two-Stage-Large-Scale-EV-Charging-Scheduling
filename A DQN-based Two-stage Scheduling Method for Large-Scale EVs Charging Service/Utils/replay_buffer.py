import numpy as np

from Environment.environment import *


class ReplayBuffer(object):
    """
        a simple replay buffer implemented by list
    """

    def __init__(self,
                 is_ev: bool = True,
                 size: int = 100000,
                 batch_size: int = 256) -> None:
        if is_ev:
            self.current_state = np.zeros([size, 1, 903, 36], dtype=np.float32)
            self.action = np.zeros([size], dtype=np.float32)
            self.reward = np.zeros([size], dtype=np.float32)
            self.next_state = np.zeros([size, 1, 903, 36], dtype=np.float32)
            self.done = np.zeros([size], dtype=np.float32)
            self.current_ev_number = np.zeros([size], dtype=int)
            self.current_cs_number = np.zeros([size], dtype=int)
            self.current_slot_number = np.zeros([size], dtype=int)
            self.current_sel_ev_number = np.zeros([size], dtype=int)
            self.next_sel_ev_number = np.zeros([size], dtype=int)
        elif not is_ev:
            self.current_state = np.zeros([size, 1, 5, 36], dtype=np.float32)
            self.action = np.zeros([size], dtype=np.float32)
            self.reward = np.zeros([size], dtype=np.float32)
            self.next_state = np.zeros([size, 1, 5, 36], dtype=np.float32)
            self.done = np.zeros([size], dtype=np.float32)
            self.current_ev_number = np.zeros([size], dtype=int)
            self.current_cs_number = np.zeros([size], dtype=int)
            self.current_slot_number = np.zeros([size], dtype=int)
            self.current_sel_ev_number = np.zeros([size], dtype=int)
            self.next_sel_ev_number = np.zeros([size], dtype=int)
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """
        sample a batch of experience and return it back
        :return: a batch of experience
        """
        idx = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            current_state=self.current_state[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_state=self.next_state[idx],
            done=self.done[idx],
            current_ev_number=self.current_ev_number[idx],
            current_cs_number=self.current_cs_number[idx],
            current_slot_number=self.current_slot_number[idx],
            current_sel_ev_number=self.current_sel_ev_number[idx],
            next_sel_ev_number=self.next_sel_ev_number[idx]
        )

    def store(self,
              current_state: np.ndarray,
              action: np.ndarray,
              reward: float,
              next_state: np.ndarray,
              done: bool,
              current_ev_number: int,
              current_cs_number: int,
              current_slot_number: int,
              current_sel_ev_number: int,
              next_sel_ev_number: int
              ) -> None:
        """
        store an exprtience item into Replay Buffer
        :param current_state: current_state
        :param action: action
        :param reward: reward
        :param next_state: next_state
        :param done: done
        :param current_ev_number: current_ev_number
        :param current_cs_number: current_cs_number
        :param current_slot_number: current_slot_number
        :param current_sel_ev_number: current_sel_ev_number
        :param next_sel_ev_number: next_sel_ev_number
        :return:
        """
        self.current_state[self.ptr] = current_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.current_ev_number[self.ptr] = current_ev_number
        self.current_cs_number[self.ptr] = current_cs_number
        self.current_slot_number[self.ptr] = current_slot_number
        self.current_sel_ev_number[self.ptr] = current_sel_ev_number
        self.next_sel_ev_number[self.ptr] = next_sel_ev_number
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        return None


class PrioritizedReplayBuffer(ReplayBuffer):
    """
     PER : Prioritized Experience Buffer
    """
    def __init__(self,
                 is_ev: bool = True,
                 size: int = 100000,
                 batch_size: int = 256,
                 alpha: float = 0.7
                 ):
        """
            a prioritized experience replay buffer
        """
        assert alpha >= 0
        super(PrioritizedReplayBuffer,
              self).__init__(is_ev=is_ev,
                             size=size,
                             batch_size=batch_size)
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self,
              current_state: np.ndarray,
              action: np.ndarray,
              reward: float,
              next_state: np.ndarray,
              done: bool,
              current_ev_number: int,
              current_cs_number: int,
              current_slot_number: int,
              current_sel_ev_number: int,
              next_sel_ev_number: int
              ) -> None:
        """
        store an exprtience item into Replay Buffer
        :param current_state: current_state
        :param action: action
        :param reward: reward
        :param next_state: next_state
        :param done: done
        :param current_ev_number: current_ev_number
        :param current_cs_number: current_cs_number
        :param current_slot_number: current_slot_number
        :param current_sel_ev_number: current_sel_ev_number
        :param next_sel_ev_number: next_sel_ev_number
        :return: None
        """
        super().store(
            current_state=current_state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            current_ev_number=current_ev_number,
            current_cs_number=current_cs_number,
            current_slot_number=current_slot_number,
            current_sel_ev_number=current_sel_ev_number,
            next_sel_ev_number=next_sel_ev_number
        )
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        return None

    def _calculate_weight(self,
                          idx: int,
                          beta: float) -> float:
        """
        calculate weight
        :param idx: index
        :param beta: hyper-parameter
        :return: weight
        """
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        weight = (p_sample * len(self)) ** (-beta)
        weight /= max_weight
        return weight

    def _sample_proportional(self) -> List[int]:
        """
        None
        :return: List[int]
        """
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upper_bound=upperbound)
            indices.append(idx)
        return indices

    def sample_batch(self,
                     beta: float = 0.5) -> Dict[str, np.ndarray]:
        """
        sample a batch of experience
        :return: experience
        """
        assert len(self) >= self.batch_size
        assert beta > 0
        indices = self._sample_proportional()

        current_state = self.current_state[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_state = self.next_state[indices]
        done = self.done[indices]
        current_ev_number = self.current_ev_number[indices]
        current_cs_number = self.current_cs_number[indices]
        current_slot_number = self.current_slot_number[indices]
        current_sel_ev_number = self.current_sel_ev_number[indices]
        next_sel_ev_number = self.next_sel_ev_number[indices]

        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            current_state=current_state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            current_ev_number=current_ev_number,
            current_cs_number=current_cs_number,
            current_slot_number=current_slot_number,
            current_sel_ev_number=current_sel_ev_number,
            next_sel_ev_number=next_sel_ev_number,
            weights=weights,
            indices=indices
        )

    def update_priorities(self,
                          indices: List[int],
                          priorities: np.ndarray) -> None:
        """
        update priorities
        :param indices: indices of experience
        :param priorities: priorities of experience
        :return: None
        """
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
        return None