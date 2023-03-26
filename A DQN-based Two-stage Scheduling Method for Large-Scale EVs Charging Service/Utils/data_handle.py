from Environment.environment import *
from Utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


# noinspection PyUnreachableCode
class DataHandle(object):
    """
        handle with data
    """

    def __init__(self,
                 is_cs: bool = False,
                 gamma: float = 0.7):

        self.env = Environment()
        self.is_cs = is_cs
        self.gamma = gamma
        self.all_part = True
        if not self.is_cs:
            self.current_state = np.zeros(shape=(899, 1, 903, 36), dtype=np.float64)
        else:
            self.current_state = np.zeros(shape=(899, 1, 5, 36), dtype=np.float64)
        self.action = np.zeros(shape=(899,), dtype=np.int)
        self.reward = np.zeros(shape=(34, 5, 899), dtype=np.float64)
        self.reward_ = np.zeros(shape=(899,), dtype=np.float64)
        if not self.is_cs:
            self.next_state = np.zeros(shape=(899, 1, 903, 36), dtype=np.float64)
        else:
            self.next_state = np.zeros(shape=(899, 1, 5, 36), dtype=np.float64)
        self.done = np.zeros(shape=(899,), dtype=np.bool)
        self.current_ev_number = np.zeros(shape=(899,), dtype=np.int)
        self.current_cs_number = np.zeros(shape=(899,), dtype=np.int)
        self.current_slot_number = np.zeros(shape=(899,), dtype=np.int)
        self.current_sel_ev_number = np.zeros(shape=(899,), dtype=np.int)
        self.next_sel_ev_number = np.zeros(shape=(899,), dtype=np.int)

        self.map_order_to_position = dict()
        self.map_position_to_order = dict()
        self.reward_pointer = np.zeros(shape=(34, 5), dtype=np.int)  # an EV's queueing number at every slot
        self.schduling_order = 0

        self.scheduling_index = []

    def calculate(self) -> None:


        if not self.is_cs and not self.all_part:
            for i in range(34):
                for j in range(5):
                    if self.reward_pointer[i][j] >= 3:
                        for k in range(self.reward_pointer[i][j] - 2):
                            self.reward[i][j][k] += (self.gamma * self.reward[i][j][k + 1]) + \
                                                    (self.gamma * self.gamma * self.reward[i][j][k + 2])
                    else:
                        if self.reward_pointer[i][j] == 2:
                            self.reward[i][j][0] += (self.gamma * self.reward[i][j][1])

        if not self.is_cs and self.all_part:
            for i in range(len(self.reward_) - 2):
                self.reward_[i] += self.gamma * self.reward_[i + 1] + self.gamma * self.gamma * self.reward_[i + 2]
        return None

    def move_experience_from_file_to_replaybuffer(self,
                                                  replay_buffer: ReplayBuffer,
                                                  exp_num: int = 50000) -> None:

        filename = None
        # cs
        if self.is_cs and self.all_part:
            filename = "../Data/pre-trained data/pickle_pretrain_cs_data_all_2_times"
        if self.is_cs and not self.all_part:
            filename = "../Data/pre-trained data/pickle_pretrain_cs_data_part"
        # ev
        if not self.is_cs and self.all_part:
            filename = "../Data/pre-trained data/pickle_pretrain_ev_data_all_2_times"
        if not self.is_cs and not self.all_part:
            filename = "../Data/pre-trained data/pickle_pretrain_ev_data_part"
        i = 0
        with open(file=filename, mode="rb+") as file:
            while i < exp_num:
                try:
                    i += 1
                    if i % 5000 == 0:
                        print(i)
                    data = cPickle.load(file=file)
                    current_state = data[0]["current_state"]
                    action = data[0]["action"]
                    reward = data[0]["reward"]
                    next_state = data[0]["next_state"]
                    done = data[0]["done"]
                    current_ev_number = data[0]["current_ev_number"]
                    current_cs_number = data[0]["current_cs_number"]
                    current_slot_number = data[0]["current_sl_number"]
                    current_sel_ev_number = data[0]["current_sel_ev_number"]
                    next_sel_ev_number = data[0]["next_sel_ev_number"]
                    replay_buffer.store(current_state=current_state,
                                        action=action,
                                        reward=reward,
                                        next_state=next_state,
                                        done=done,
                                        current_ev_number=current_ev_number,
                                        current_cs_number=current_cs_number,
                                        current_slot_number=current_slot_number,
                                        current_sel_ev_number=current_sel_ev_number,
                                        next_sel_ev_number=next_sel_ev_number)

                except EOFError:
                    pass
        return None

    def move_exp(self,
                 replay_buffer,
                 exp_num: int = 70000) -> None:

        filename = None
        if self.is_cs and self.all_part:
            filename = "../Data/pre-trained data/pickle_pretrain_cs_data_all_2_times"
        if not self.is_cs and self.all_part:
            filename = "../Data/pre-trained data/pickle_pretrain_ev_data_all_2_times"
        i = 0
        with open(file=filename, mode="rb+") as file:
            while i < exp_num:
                try:
                    if i % 5000 == 0:
                        print(i)
                    data = cPickle.load(file=file)
                    current_state = data[0]["current_state"]
                    action = data[0]["action"]
                    reward = data[0]["reward"]
                    next_state = data[0]["next_state"]
                    done = data[0]["done"]
                    current_ev_number = data[0]["current_ev_number"]
                    current_cs_number = data[0]["current_cs_number"]
                    current_slot_number = data[0]["current_sl_number"]
                    current_sel_ev_number = data[0]["current_sel_ev_number"]
                    next_sel_ev_number = data[0]["next_sel_ev_number"]
                    replay_buffer.store(current_state=current_state,
                                        action=action,
                                        reward=reward,
                                        next_state=next_state,
                                        done=done,
                                        current_ev_number=current_ev_number,
                                        current_cs_number=current_cs_number,
                                        current_slot_number=current_slot_number,
                                        current_sel_ev_number=current_sel_ev_number,
                                        next_sel_ev_number=next_sel_ev_number)
                    i += 1
                except EOFError:
                    pass
        return None

    def move_pre_trained_data_from_temporary_space_to_file(self,
                                                           file_) -> None:
        filename = None

        if self.is_cs and self.all_part:
            filename = "../Data/pre-trained data/pickle_pretrain_cs_data_all_2_times"
        if self.is_cs and not self.all_part:
            filename = "../Data/pre-trained data/pickle_pretrain_cs_data_part"
        # ev
        if not self.is_cs and self.all_part:
            filename = "../Data/pre-trained data/pickle_pretrain_ev_data_all_2_times"
        if not self.is_cs and not self.all_part:
            filename = "../Data/pre-trained data/pickle_pretrain_ev_data_part"

        for i in range(len(self.scheduling_index)):
            # self.scheduling_index = [899 X
            # --> {current_ev_number, current_cs_number, current_slot_number, order in slot, scheduling_order}]
            ev_number = self.scheduling_index[i][0]
            cs_number = self.scheduling_index[i][1]
            sl_number = self.scheduling_index[i][2]
            or_in_slt = self.scheduling_index[i][3]
            sched_ord = self.scheduling_index[i][4]

            current_state = self.current_state[sched_ord]
            action = self.action[sched_ord]
            if not self.all_part:
                reward = self.reward[cs_number, sl_number, or_in_slt]
            else:
                reward = self.reward_[sched_ord]
            next_state = self.next_state[sched_ord]
            done = self.done[sched_ord]
            current_ev_number = self.current_ev_number[sched_ord]
            current_cs_number = self.current_cs_number[sched_ord]
            current_sl_number = self.current_slot_number[sched_ord]
            current_sel_ev_number = self.current_sel_ev_number[sched_ord]
            next_sel_ev_number = self.next_sel_ev_number[sched_ord]

            data_dict = [
                {
                    'current_state': current_state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done,
                    'current_ev_number': current_ev_number,
                    'current_cs_number': current_cs_number,
                    'current_sl_number': current_sl_number,
                    'current_sel_ev_number': current_sel_ev_number,
                    'next_sel_ev_number': next_sel_ev_number
                }
            ]
            cPickle.dump(data_dict, file_)

        return None

    def move_experience_to_replaybuffer_when_in_training_process(self,
                                                                 replay_buffer: ReplayBuffer) -> None:

        for i in range(len(self.scheduling_index)):

            ev_number = self.scheduling_index[i][0]
            cs_number = self.scheduling_index[i][1]
            sl_number = self.scheduling_index[i][2]
            or_in_slt = self.scheduling_index[i][3]
            sched_ord = self.scheduling_index[i][4]

            current_state = self.current_state[sched_ord]
            action = self.action[sched_ord]
            if not self.all_part:
                reward = self.reward[cs_number, sl_number, or_in_slt]
            else:
                reward = self.reward_[sched_ord]
            next_state = self.next_state[sched_ord]
            done = self.done[sched_ord]
            current_ev_number = self.current_ev_number[sched_ord]
            current_cs_number = self.current_cs_number[sched_ord]
            current_sl_number = self.current_slot_number[sched_ord]
            current_sel_ev_number = self.current_sel_ev_number[sched_ord]
            next_sel_ev_number = self.next_sel_ev_number[sched_ord]

            transition = [current_state, action, reward, next_state, done, current_ev_number,
                          current_cs_number, current_sl_number, current_sel_ev_number, next_sel_ev_number]

            replay_buffer.store(*transition)
            return None

    def reset(self) -> None:
        self.env = Environment()
        self.is_cs = self.is_cs
        self.gamma = self.gamma
        if not self.is_cs:
            self.current_state = np.zeros(shape=(899, 1, 903, 36), dtype=np.float64)
        else:
            self.current_state = np.zeros(shape=(899, 1, 5, 36), dtype=np.float64)
        self.action = np.zeros(shape=(899,), dtype=np.int)
        self.reward = np.zeros(shape=(34, 5, 899), dtype=np.float64)
        self.reward_ = np.zeros(shape=(899,), dtype=np.float64)
        if not self.is_cs:
            self.next_state = np.zeros(shape=(899, 1, 903, 36), dtype=np.float64)
        else:
            self.next_state = np.zeros(shape=(899, 1, 5, 36), dtype=np.float64)
        self.done = np.zeros(shape=(899,), dtype=np.bool)
        self.current_ev_number = np.zeros(shape=(899,), dtype=np.int)
        self.current_cs_number = np.zeros(shape=(899,), dtype=np.int)
        self.current_slot_number = np.zeros(shape=(899,), dtype=np.int)
        self.current_sel_ev_number = np.zeros(shape=(899,), dtype=np.int)
        self.next_sel_ev_number = np.zeros(shape=(899,), dtype=np.int)

        self.map_order_to_position = dict()
        self.map_position_to_order = dict()
        self.reward_pointer = np.zeros(shape=(34, 5), dtype=np.int)
        self.schduling_order = 0
        self.scheduling_index = []
        return None

    def store(self,
              current_state: np.ndarray,
              action: np.int,
              reward: np.float64,
              next_state: np.ndarray,
              done: np.bool,
              current_ev_number: np.int,
              current_cs_number: np.int,
              current_slot_number: np.int,
              current_sel_ev_number: np.int,
              next_sel_ev_number: np.int
              ) -> None:

        pointer = self.reward_pointer[current_cs_number][current_slot_number]
        self.map_order_to_position[self.schduling_order] = [current_ev_number,
                                                            current_cs_number,
                                                            current_slot_number,
                                                            pointer]
        self.map_position_to_order[str([current_ev_number,
                                        current_cs_number,
                                        current_slot_number,
                                        pointer])] = self.schduling_order

        self.current_state[self.schduling_order] = current_state
        self.action[self.schduling_order] = action
        self.reward[current_cs_number, current_slot_number, pointer] = reward
        self.reward_[self.schduling_order] = reward
        self.next_state[self.schduling_order] = next_state
        self.done[self.schduling_order] = done
        self.current_ev_number[self.schduling_order] = current_ev_number
        self.current_cs_number[self.schduling_order] = current_cs_number
        self.current_slot_number[self.schduling_order] = current_slot_number
        self.current_sel_ev_number[self.schduling_order] = current_sel_ev_number
        self.next_sel_ev_number[self.schduling_order] = next_sel_ev_number

        self.scheduling_index.append([current_ev_number,
                                      current_cs_number,
                                      current_slot_number,
                                      pointer,
                                      self.schduling_order])

        self.schduling_order += 1
        pointer += 1
        return None
