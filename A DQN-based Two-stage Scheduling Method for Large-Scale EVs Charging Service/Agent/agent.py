import sys
# sys.path.append("E:\EV_Charging_Scheduling")

from Environment.environment import *

import torch

from Environment.environment import *
from Model.model import EvNet, CsNet
from Utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from Utils.data_handle import DataHandle
from Utils.data_squeeze import squeeze_pre_trained_data


class DQNAgent(object):
    """
        define a class used as an Agent for reinforcement learning
    """

    def __init__(self,
                 memory_size: int = 150000,
                 batch_size: int = 2048,
                 target_update: int = 100,
                 epsilon_decay: int = 1 / 200,
                 max_epsilon: float = 0.15,
                 min_epsilon: float = 0.1,
                 gamma: float = 0.1,
                 ) -> None:
        self.env = Environment(teacher_version=True)
        self.batch_size = batch_size
        self.device = torch.device("cuda")
        self.total_steps = 0
        self.random_step = 899 * 50
        self.update_cnt = 0
        self.is_test = False
        self.ev_memory = ReplayBuffer(size=memory_size)
        self.ev_epsilon = max_epsilon
        self.ev_max_epsilon = max_epsilon
        self.ev_min_epsilon = min_epsilon
        self.ev_epsilon_decay = epsilon_decay
        self.ev_target_update = target_update
        self.ev_gamma = gamma
        self.EV_DQN = EvNet().to(self.device)
        self.EV_DQN_TARGET = EvNet().to(self.device)
        self.EV_DQN_TARGET.load_state_dict(self.EV_DQN.state_dict())
        self.EV_DQN_TARGET.eval()
        self.ev_optim = optim.Adagrad(self.EV_DQN.parameters(), lr=0.005)
        self.ev_transition = list()
        self.ev_current_state_ = list()
        self.ev_action_ = list()
        self.ev_reward_ = list()
        self.ev_next_state_ = list()
        self.ev_done_flag_ = list()
        self.ev_current_ev_number_ = list()
        self.ev_current_cs_number_ = list()
        self.ev_current_slot_number_ = list()
        self.ev_current_sel_ev_number_ = list()
        self.ev_next_sel_ev_number_ = list()
        self.cs_memory = ReplayBuffer(is_ev=False,
                                      size=memory_size)
        self.cs_epsilon = max_epsilon
        self.cs_max_epsilon = max_epsilon
        self.cs_min_epsilon = min_epsilon
        self.cs_epsilon_decay = epsilon_decay
        self.cs_target_update = target_update
        self.cs_gamma = gamma
        self.CS_DQN = CsNet().to(self.device)
        self.CS_DQN_TARGET = CsNet().to(self.device)
        self.CS_DQN_TARGET.load_state_dict(self.CS_DQN.state_dict())
        self.CS_DQN_TARGET.eval()
        self.cs_optim = optim.Adam(self.CS_DQN.parameters(), lr=0.005)
        self.cs_transition = list()
        self.cs_current_state_ = list()
        self.cs_action_ = list()
        self.cs_reward_ = list()
        self.cs_next_state_ = list()
        self.cs_done_flag_ = list()
        self.cs_current_ev_number_ = list()
        self.cs_current_cs_number_ = list()
        self.cs_current_slot_number_ = list()
        self.cs_current_sel_ev_number_ = list()
        self.cs_next_sel_ev_number_ = list()
        self.cs_data_hanlder = DataHandle(is_cs=True)
        self.ev_data_hanlder = DataHandle()
        self.pre_training_round_number = 60000
        self.trajectory_len = len(self.env.get_schedulable_ev())
        self.reachability_matrix = torch.Tensor(self.env.get_reachability_matrix()).to(self.device)

    def init_storage_pool(self) -> None:
        self.ev_current_state_ = list()
        self.ev_action_ = list()
        self.ev_reward_ = list()
        self.ev_next_state_ = list()
        self.ev_done_flag_ = list()
        self.ev_current_ev_number_ = list()
        self.ev_current_cs_number_ = list()
        self.ev_current_slot_number_ = list()
        self.ev_current_sel_ev_number_ = list()
        self.ev_next_sel_ev_number_ = list()
        self.cs_current_state_ = list()
        self.cs_action_ = list()
        self.cs_reward_ = list()
        self.cs_next_state_ = list()
        self.cs_done_flag_ = list()
        self.cs_current_ev_number_ = list()
        self.cs_current_cs_number_ = list()
        self.cs_current_slot_number_ = list()
        self.cs_current_sel_ev_number_ = list()
        self.cs_next_sel_ev_number_ = list()
        return None

    def compute_dqn_loss(self,
                         ev_sample: Dict[str, np.ndarray],
                         cs_sample: Dict[str, np.ndarray]):
        ev_current_state = torch.FloatTensor(ev_sample["current_state"]).to(self.device)
        ev_action = torch.LongTensor(ev_sample["action"]).reshape(-1, 1).to(self.device)
        ev_reward = torch.FloatTensor(ev_sample["reward"]).reshape(-1, 1).to(self.device)
        ev_next_state = torch.FloatTensor(ev_sample["next_state"]).to(self.device)
        ev_done = torch.LongTensor(ev_sample["done"]).reshape(-1, 1).to(self.device)
        ev_current_ev_number = torch.LongTensor(ev_sample["current_ev_number"]).reshape(-1, 1).to(self.device)
        ev_current_cs_number = torch.LongTensor(ev_sample["current_cs_number"]).reshape(-1, 1).to(self.device)
        ev_current_slot_number = torch.LongTensor(ev_sample["current_slot_number"]).reshape(-1, 1).to(self.device)
        ev_current_sel_ev_number = torch.LongTensor(ev_sample["current_sel_ev_number"]).reshape(-1, 1).to(self.device)
        ev_next_sel_ev_number = torch.LongTensor(ev_sample["next_sel_ev_number"]).reshape(-1, 1).to(self.device)
        cs_current_state = torch.FloatTensor(cs_sample["current_state"]).to(self.device)
        cs_action = torch.LongTensor(cs_sample["action"]).reshape(-1, 1).to(self.device)
        cs_reward = torch.FloatTensor(cs_sample["reward"]).reshape(-1, 1).to(self.device)
        cs_next_state = torch.FloatTensor(cs_sample["next_state"]).to(self.device)
        cs_done = torch.LongTensor(cs_sample["done"]).reshape(-1, 1).to(self.device)
        cs_current_ev_number = torch.LongTensor(cs_sample["current_ev_number"]).reshape(-1, 1).to(self.device)
        cs_current_cs_number = torch.LongTensor(cs_sample["current_cs_number"]).reshape(-1, 1).to(self.device)
        cs_current_slot_number = torch.LongTensor(cs_sample["current_slot_number"]).reshape(-1, 1).to(self.device)
        cs_current_sel_ev_number = torch.LongTensor(cs_sample["current_sel_ev_number"]).reshape(-1, 1).to(self.device)
        cs_next_sel_ev_number = torch.LongTensor(cs_sample["next_sel_ev_number"]).reshape(-1, 1).to(self.device)

        ev_action = ev_action.cpu().apply_(self.env.transfer_ev_no_to_order).to(self.device)
        ev_next_action = ev_next_sel_ev_number.cpu().apply_(self.env.transfer_ev_no_to_order).to(self.device)
        ev_current_q_value = self.EV_DQN(ev_current_state).gather(1, ev_action)
        ev_next_q_value = self.EV_DQN_TARGET(ev_next_state).gather(1, ev_next_action)
        ev_mask = 1 - ev_done
        ev_target_value = (ev_reward + ev_mask * self.ev_gamma * ev_next_q_value)
        ev_loss = f.smooth_l1_loss(ev_target_value, ev_reward)

        cs_current_q_value = self.CS_DQN(cs_current_state).gather(1, cs_action)
        cs_loss = f.smooth_l1_loss(cs_current_q_value, cs_reward)
        return cs_loss, ev_loss

    def get_total_step(self) -> int:
        return self.total_steps

    def select_action(self) -> Tuple:
        ev_sel = None
        cs_sel = None
        ev_state = None
        cs_state = None
        ev_state = self.env.get_current_ev_state()
        ev_val_list = self.EV_DQN(torch.FloatTensor(ev_state).unsqueeze(dim=0).to("cuda")).cpu().\
            squeeze(dim=0).data.numpy()
        ev_num = []
        ev_val = []
        for i in range(len(ev_val_list)):
            if self.env.transfer_ev_order_to_no(i) in self.env.get_not_scheduled_ev():
                ev_num.append(self.env.transfer_ev_order_to_no(i))
                ev_val.append(ev_val_list[i] * 0.5 +
                              self.env.get_average_first_k_distance_of_ev_to_cs(self.env.transfer_ev_order_to_no(i), 3) * 0.5)  # V5  24.583 - 6.454
        ev_sel = None
        if self.ev_epsilon < random.random():
            ev_sel = ev_num[np.array(ev_val).argmin()]
        else:
            ev_sel = ev_num[random.randint(0, len(ev_num) - 1)]
        cs_state = self.env.get_current_cs_state(sel_ev_number=ev_sel)
        cs_val_list = self.CS_DQN(torch.FloatTensor(cs_state).unsqueeze(dim=0).to("cuda")).\
            cpu().squeeze(dim=0).data.numpy()
        cs_val = []
        cs_num = []
        for i in range(len(cs_val_list)):
            if self.env.get_reachability_of_an_ev(ev_number=ev_sel)[i] == 1.0:
                cs_val.append(cs_val_list[i] + self.env.get_charging_ev_number_for_concrete_cs(i) * 2.5)  # V4
                cs_num.append(i)
        cs_sel = None
        if self.cs_epsilon < random.random():
            cs_sel = cs_num[np.array(cs_val).argmin()]
        else:
            cs_sel = cs_num[random.randint(0, len(cs_num) - 1)]
        return ev_sel, cs_sel, ev_state, cs_state

    def step(self,
             ev_number: int,
             cs_number: int) -> Tuple:

        slot = self.env.get_best_slot(cs_number=cs_number)
        self.env.step(ev_number=ev_number,
                      cs_number=cs_number,
                      slot_number=slot)
        ev_reward = self.env.get_reward_for_ev(q=1.0,
                                               i=0,
                                               ev_number=ev_number,
                                               cs_number=cs_number,
                                               slot_number=slot)
        cs_reward = self.env.get_reward_for_cs(cs_number=cs_number, ev_number=ev_number)
        done = self.env.is_done()
        self.total_steps += 1
        return ev_reward, cs_reward, done

    def update_model(self) -> Tuple:

        cs_samples = self.cs_memory.sample_batch()
        ev_samples = self.ev_memory.sample_batch()
        cs_loss, ev_loss = self.compute_dqn_loss(ev_samples, cs_samples)
        self.cs_optim.zero_grad()
        cs_loss.backward()
        self.cs_optim.step()
        self.ev_optim.zero_grad()
        ev_loss.backward()
        self.ev_optim.step()
        return cs_loss.data, ev_loss.data

    def ev_target_hard_update(self) -> None:

        self.EV_DQN_TARGET.load_state_dict(self.EV_DQN.state_dict())
        return None

    @staticmethod
    def cs_target_hard_update() -> None:
        return None

    def train(self) -> None:

        self.env.reset()
        pre_train_ev_loss = []
        pre_train_cs_loss = []
        formal_ev_loss = []
        formal_cs_loss = []
        ave_queue_time = []
        ave_idle_time = []
        weighted_value = []
        solution = []
        solution_value = []
        ev_epsilon = []
        cs_epsilon = []
        pre_train_queue = []
        pre_train_idle = []
        pre_train_ev_c_n = []
        """↓ Pre-training data preparation is over ↓"""
        pre_train_value_list = None
        self.is_test = False
        if os.path.getsize("../Data/pre-trained data/pickle_pretrain_ev_data_all_2_times") == 0 and \
                os.path.getsize("../Data/pre-trained data/pickle_pretrain_cs_data_all_2_times") == 0:
            print("Stage pre-training")
            pre_train_value_list = []
            pre_train_data = squeeze_pre_trained_data()
            file_ev = open(file="../Data/pre-trained data/pickle_pretrain_ev_data_all_2_times", mode="ab+")
            file_cs = open(file="../Data/pre-trained data/pickle_pretrain_cs_data_all_2_times", mode="ab+")
            for i in range(len(pre_train_data)):
                print("Extracting pre-train data. round {0}".format(i), end="\t")
                start = time.time()
                self.env.reset()
                first_round = True
                for j in range(len(pre_train_data[i])):
                    ev_number = self.env.transfer_ev_order_to_no(pre_train_data[i][j][2])
                    cs_number = pre_train_data[i][j][0]
                    sl_number = self.env.get_best_slot(cs_number=cs_number)
                    ev_current_state = self.env.get_current_ev_state()
                    self.env.step(ev_number=ev_number,
                                  cs_number=cs_number,
                                  slot_number=sl_number)
                    ev_action = ev_number
                    ev_reward = self.env.get_reward_for_ev(
                        q=1.0,
                        i=0,
                        ev_number=ev_number,
                        cs_number=cs_number,
                        slot_number=sl_number
                    )
                    if not first_round:
                        ev_next_state = ev_current_state
                        self.ev_next_state_.append(ev_next_state)
                    if self.env.is_done():
                        ev_next_state = ev_current_state
                        self.ev_next_state_.append(ev_next_state)
                    ev_done = self.env.is_done()
                    ev_current_ev_number = ev_number
                    ev_current_cs_number = cs_number
                    ev_current_slot_number = sl_number
                    ev_current_sel_ev_number = ev_number
                    if not first_round:
                        ev_next_sel_ev_number = ev_number
                        self.ev_next_sel_ev_number_.append(ev_next_sel_ev_number)
                    if self.env.is_done():
                        ev_next_sel_ev_number = ev_number
                        self.ev_next_sel_ev_number_.append(ev_next_sel_ev_number)
                    """ append all ev-side information into _ array """
                    self.ev_current_state_.append(ev_current_state)
                    self.ev_action_.append(ev_action)
                    self.ev_reward_.append(ev_reward)
                    self.ev_done_flag_.append(ev_done)
                    self.ev_current_ev_number_.append(ev_current_ev_number)
                    self.ev_current_cs_number_.append(ev_current_cs_number)
                    self.ev_current_slot_number_.append(ev_current_slot_number)
                    self.ev_current_sel_ev_number_.append(ev_current_sel_ev_number)
                    cs_current_state = self.env.get_current_cs_state(sel_ev_number=ev_number)
                    cs_action = cs_number
                    cs_reward = self.env.get_reward_for_cs(cs_number, ev_number)
                    if not first_round:
                        cs_next_state = cs_current_state
                        self.cs_next_state_.append(cs_next_state)
                    if self.env.is_done():
                        cs_next_state = cs_current_state
                        self.cs_next_state_.append(cs_next_state)
                    cs_done = self.env.is_done()
                    cs_current_ev_number = ev_number
                    cs_current_cs_number = cs_number
                    cs_current_slot_number = sl_number
                    cs_current_sel_ev_number = ev_number
                    if not first_round:
                        cs_next_sel_ev_number = ev_number
                        self.cs_next_sel_ev_number_.append(cs_next_sel_ev_number)
                    if self.env.is_done():
                        cs_next_sel_ev_number = ev_number
                        self.cs_next_sel_ev_number_.append(cs_next_sel_ev_number)
                    self.cs_current_state_.append(cs_current_state)
                    self.cs_action_.append(cs_action)
                    self.cs_reward_.append(cs_reward)
                    self.cs_done_flag_.append(cs_done)
                    self.cs_current_ev_number_.append(cs_current_ev_number)
                    self.cs_current_cs_number_.append(cs_current_cs_number)
                    self.cs_current_slot_number_.append(cs_current_slot_number)
                    self.cs_current_sel_ev_number_.append(cs_current_sel_ev_number)
                    first_round = False
                if self.env.is_done():
                    for s in range(len(self.ev_done_flag_)):
                        self.ev_data_hanlder.store(
                            current_state=self.ev_current_state_[s],
                            action=self.ev_action_[s],
                            reward=self.ev_reward_[s],
                            next_state=self.ev_next_state_[s],
                            done=self.ev_done_flag_[s],
                            current_ev_number=self.ev_current_ev_number_[s],
                            current_cs_number=self.ev_current_cs_number_[s],
                            current_slot_number=self.ev_current_slot_number_[s],
                            current_sel_ev_number=self.ev_current_sel_ev_number_[s],
                            next_sel_ev_number=self.ev_next_sel_ev_number_[s]
                        )
                    self.ev_data_hanlder.calculate()
                    self.ev_data_hanlder.move_pre_trained_data_from_temporary_space_to_file(file_=file_ev)
                    self.ev_data_hanlder.reset()
                    for t in range(len(self.cs_done_flag_)):
                        self.cs_data_hanlder.store(
                            current_state=self.cs_current_state_[t],
                            action=self.cs_action_[t],
                            reward=self.cs_reward_[t],
                            next_state=self.cs_next_state_[t],
                            done=self.cs_done_flag_[t],
                            current_ev_number=self.cs_current_ev_number_[t],
                            current_cs_number=self.cs_current_cs_number_[t],
                            current_slot_number=self.cs_current_slot_number_[t],
                            current_sel_ev_number=self.cs_current_sel_ev_number_[t],
                            next_sel_ev_number=self.cs_next_sel_ev_number_[t]
                        )
                    self.cs_data_hanlder.calculate()
                    self.cs_data_hanlder.move_pre_trained_data_from_temporary_space_to_file(file_=file_cs)
                    self.cs_data_hanlder.reset()
                self.init_storage_pool()
                end = time.time()
            if self.env.is_done():
                value = self.env.get_reward()
                pre_train_value_list.append(value)
                self.init_storage_pool()
                first_round = True
                self.env.reset()
            file_ev.close()
            file_cs.close()
        print("moving data...")
        self.ev_data_hanlder.move_experience_from_file_to_replaybuffer(self.ev_memory, 150000)
        self.cs_data_hanlder.move_experience_from_file_to_replaybuffer(self.cs_memory, 150000)
        if len(self.cs_memory) >= self.batch_size and len(self.ev_memory) >= self.batch_size:
            print("Stage pre-training for {} times...".format(self.pre_training_round_number))
            for i in range(self.pre_training_round_number):
                print("pre-training round {0}".format(i))
                cs_loss, ev_loss = self.update_model()
                # R
                pre_train_ev_loss.append(ev_loss)
                pre_train_cs_loss.append(cs_loss)
                if i % 30 == 0:
                    for j in range(self.trajectory_len):
                        ev_sel, cs_sel, ev_state, cs_state = self.select_action()
                        slot_number = self.env.get_best_slot(cs_number=cs_sel)
                        ev_reward, cs_reward, done = self.step(ev_sel, cs_sel)
                        if done:
                            idle = self.env.get_idling_time() / 170
                            queue = self.env.get_queueing_time() / 899
                            self.env.reset()
                            pre_train_queue.append(queue)
                            pre_train_idle.append(idle)
                            pre_train_ev_c_n.append(self.env.get_hour_charing_ev())
                self.update_cnt += 1
                if self.update_cnt % self.ev_target_update == 0:
                    self.ev_target_hard_update()
                if self.update_cnt % self.cs_target_update == 0:
                    self.cs_target_hard_update()
                print("cs_loss = {0}, ev_loss = {1}".format(cs_loss, ev_loss))
        # with open("../Data/result/pre_train_idle", "wb") as f:
        # with open("../Data/lr0_01/pre_train_idle", "wb") as f:
        #     cPickle.dump(pre_train_idle, f)
        #     f.close()
        # with open("../Data/result/pre_train_queue", "wb") as f:
        # with open("../Data/lr0_01/pre_train_queue", "wb") as f:
        #     cPickle.dump(pre_train_queue, f)
        #     f.close()
        # with open("../Data/result/pre_train_ev_c_n", "wb") as f:
        # with open("../Data/lr0_01/pre_train_queue", "wb") as f:
        #     cPickle.dump(pre_train_ev_c_n, f)
        #     f.close()

        """↓ Training model phase ↓"""
        print("Stage formal-training")
        for i in range(1, 2001):
            if i % 100 == 0:
                print("moving data...")
                self.ev_data_hanlder.move_exp(self.ev_memory, 60000)
                self.cs_data_hanlder.move_exp(self.cs_memory, 60000)
            print("formal training process ,round {0}".format(i))
            first_round = True
            cs_loss = None
            ev_loss = None
            self.env.reset()  # reset agent's environment after every single training process is over
            temp_solution = []
            for j in range(self.trajectory_len):  # for each round, one of the rounds has 899 cycles
                ev_sel, cs_sel, ev_state, cs_state = self.select_action()
                slot_number = self.env.get_best_slot(cs_number=cs_sel)
                ev_reward, cs_reward, done = self.step(ev_sel, cs_sel)
                """ side of electric vehicle """
                ev_current_state = ev_state
                ev_action = ev_sel
                ev_reward = ev_reward
                if not first_round:
                    ev_next_state = ev_current_state
                    self.ev_next_state_.append(ev_next_state)
                if self.env.is_done():
                    ev_next_state = ev_current_state
                    self.ev_next_state_.append(ev_next_state)
                ev_done = self.env.is_done()
                ev_current_ev_number = ev_sel
                ev_current_cs_number = cs_sel
                ev_current_slot_number = slot_number
                ev_current_sel_ev_number = ev_sel
                if not first_round:
                    ev_next_sel_ev_number = ev_sel
                    self.ev_next_sel_ev_number_.append(ev_next_sel_ev_number)
                if self.env.is_done():
                    ev_next_sel_ev_number = ev_sel
                    self.ev_next_sel_ev_number_.append(ev_next_sel_ev_number)
                """ append all ev-side information into _ array """
                self.ev_current_state_.append(ev_current_state)
                self.ev_action_.append(ev_action)
                self.ev_reward_.append(ev_reward)
                self.ev_done_flag_.append(ev_done)
                self.ev_current_ev_number_.append(ev_current_ev_number)
                self.ev_current_cs_number_.append(ev_current_cs_number)
                self.ev_current_slot_number_.append(ev_current_slot_number)
                self.ev_current_sel_ev_number_.append(ev_current_sel_ev_number)
                """ side of charging station """
                cs_current_state = cs_state
                cs_action = cs_sel
                cs_reward = cs_reward
                if not first_round:
                    cs_next_state = cs_current_state
                    self.cs_next_state_.append(cs_next_state)
                if self.env.is_done():
                    cs_next_state = cs_current_state
                    self.cs_next_state_.append(cs_next_state)
                cs_done = self.env.is_done()
                cs_current_ev_number = ev_sel
                cs_current_cs_number = cs_sel
                cs_current_slot_number = slot_number
                cs_current_sel_ev_number = ev_sel
                if not first_round:
                    cs_next_sel_ev_number = ev_sel
                    self.cs_next_sel_ev_number_.append(cs_next_sel_ev_number)
                if self.env.is_done():
                    cs_next_sel_ev_number = ev_sel
                    self.cs_next_sel_ev_number_.append(cs_next_sel_ev_number)
                """ append all cs-side information to _ array """
                self.cs_current_state_.append(cs_current_state)
                self.cs_action_.append(cs_action)
                self.cs_reward_.append(cs_reward)
                self.cs_done_flag_.append(cs_done)
                self.cs_current_ev_number_.append(cs_current_ev_number)
                self.cs_current_cs_number_.append(cs_current_cs_number)
                self.cs_current_slot_number_.append(cs_current_slot_number)
                self.cs_current_sel_ev_number_.append(cs_current_sel_ev_number)
                first_round = False
                if self.env.is_done():
                    self.env.optimize()
                    solution.append(self.env.ret_scheduling_consequence_list())
                    solution_value.append(self.env.get_reward())
                    weighted_value.append(self.env.get_reward())
                    ave_idle_time.append(self.env.get_idling_time())
                    ave_queue_time.append(self.env.get_queueing_time())
            if self.env.is_done():
                """ for ev part """
                for s in range(len(self.ev_done_flag_)):
                    self.ev_data_hanlder.store(
                        current_state=self.ev_current_state_[s],
                        action=self.ev_action_[s],
                        reward=self.ev_reward_[s],
                        next_state=self.ev_next_state_[s],
                        done=self.ev_done_flag_[s],
                        current_ev_number=self.ev_current_ev_number_[s],
                        current_cs_number=self.ev_current_cs_number_[s],
                        current_slot_number=self.ev_current_slot_number_[s],
                        current_sel_ev_number=self.ev_current_sel_ev_number_[s],
                        next_sel_ev_number=self.ev_next_sel_ev_number_[s]
                    )
                self.ev_data_hanlder.calculate()
                self.ev_data_hanlder.move_experience_to_replaybuffer_when_in_training_process(self.ev_memory)
                self.ev_data_hanlder.reset()
                for t in range(len(self.ev_done_flag_)):
                    self.cs_data_hanlder.store(
                        current_state=self.cs_current_state_[t],
                        action=self.cs_action_[t],
                        reward=self.cs_reward_[t],
                        next_state=self.cs_next_state_[t],
                        done=self.cs_done_flag_[t],
                        current_ev_number=self.cs_current_ev_number_[t],
                        current_cs_number=self.cs_current_cs_number_[t],
                        current_slot_number=self.cs_current_slot_number_[t],
                        current_sel_ev_number=self.cs_current_sel_ev_number_[t],
                        next_sel_ev_number=self.cs_next_sel_ev_number_[t]
                    )
                self.cs_data_hanlder.calculate()
                self.cs_data_hanlder.move_experience_to_replaybuffer_when_in_training_process(self.cs_memory)
                self.cs_data_hanlder.reset()
                self.init_storage_pool()
            if len(self.cs_memory) >= self.batch_size and len(self.ev_memory) >= self.batch_size:
                print("stage --> trainning for 500 times...")
                for it in range(500):
                    cs_loss, ev_loss = self.update_model()
                    formal_ev_loss.append(ev_loss)
                    formal_cs_loss.append(cs_loss)
                    self.update_cnt += 1
                    if self.update_cnt % self.ev_target_update == 0:
                        self.ev_target_hard_update()
                    if self.update_cnt % self.cs_target_update == 0:
                        self.cs_target_hard_update()
            if self.ev_epsilon >= 0 and self.cs_epsilon >= 0:
                ev_epsilon.append(self.ev_epsilon)
                cs_epsilon.append(cs_epsilon)
                self.ev_epsilon /= 1.5
                self.cs_epsilon /= 1.5
            self.env.reset()
        # with open("../Data/result/pre_train_ev_loss", "wb") as f:
        # with open("../Data/lr0_01/pre_train_ev_loss", "wb") as f:
        #     cPickle.dump(pre_train_ev_loss, f)
        #     f.close()
        # with open("../Data/result/pre_train_cs_loss", "wb") as f:
        # with open("../Data/lr0_01/pre_train_cs_loss", "wb") as f:
        #     cPickle.dump(pre_train_cs_loss, f)
        #     f.close()
        # with open("../Data/result/formal_ev_loss", "wb") as f:
        # with open("../Data/lr0_01/formal_ev_loss", "wb") as f:
        #     cPickle.dump(formal_ev_loss, f)
        #     f.close()
        # with open("../Data/result/formal_cs_loss", "wb") as f:
        # with open("../Data/lr0_01/formal_cs_loss", "wb") as f:
        #     cPickle.dump(formal_cs_loss, f)
        #     f.close()
        # with open("../Data/result/ave_queue_time", "wb") as f:
        # with open("../Data/lr0_01/ave_queue_time", "wb") as f:
        #     cPickle.dump(ave_queue_time, f)
        #     f.close()
        # with open("../Data/result/ave_idle_time", "wb") as f:
        # with open("../Data/lr0_01/ave_idle_time", "wb") as f:
        #     cPickle.dump(ave_idle_time, f)
        #     f.close()
        # with open("../Data/result/weighted_value", "wb") as f:
        # with open("../Data/lr0_01/weighted_value", "wb") as f:
        #     cPickle.dump(weighted_value, f)
        #     f.close()
        # with open("../Data/result/solution", "wb") as f:
        # with open("../Data/lr0_01/solution", "wb") as f:
        #     cPickle.dump(solution, f)
        #     f.close()
        # with open("../Data/result/solution_value", "wb") as f:
        # with open("../Data/lr0_01/solution_value", "wb") as f:
        #     cPickle.dump(solution_value, f)
        #     f.close()
        # with open("../Data/result/ev_epsilon", "wb") as f:
        # with open("../Data/lr0_01/ev_epsilon", "wb") as f:
        #     cPickle.dump(ev_epsilon, f)
        #     f.close()
        # with open("../Data/result/cs_epsilon", "wb") as f:
        # with open("../Data/lr0_01/cs_epsilon", "wb") as f:
        #     cPickle.dump(cs_epsilon, f)
        #     f.close()


if __name__ == '__main__':
    agent = DQNAgent()
    agent.train()
