import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import _pickle as cPickle
import random
import scipy
import symbol
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import warnings

from collections import deque, defaultdict
from IPython.display import clear_output
from scipy.special import softmax
from torch.nn.utils import clip_grad, clip_grad_norm, clip_grad_norm_, clip_grad_value_
from typing import Callable, Dict, List, NamedTuple, Tuple

np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")


class Environment(object):

    def __init__(self,
                 filename_of_ev_information: str = "../Data/ev_data_1000.csv",
                 filename_of_cs_information: str = "../Data/charging_stations.csv",
                 filename_of_cp_information: str = "../Data/ev_cp_34.csv",
                 slot_number: int = 5,
                 teacher_version: bool = True) -> None:

        assert filename_of_cs_information is not None
        assert filename_of_cp_information is not None
        assert filename_of_ev_information is not None
        assert slot_number is not None and slot_number != 0
        self.__filename_of_cp_information = filename_of_cp_information
        self.__filename_of_cs_information = filename_of_cs_information
        self.__filename_of_ev_information = filename_of_ev_information
        self.__slot_number = slot_number
        self.__cp_data = pd.read_csv(self.__filename_of_cp_information,
                                     delimiter=",").iloc[:, 1:].values
        self.__cs_data = pd.read_csv(self.__filename_of_cs_information,
                                     delimiter=";",
                                     header=None).iloc[:, 1:].values
        self.__ev_data = pd.read_csv(self.__filename_of_ev_information,
                                     delimiter=",").iloc[:, [1, 2, 3, 4]].values
        self.__ev_x = [j for i in self.__ev_data[:, 0:1] for j in i]
        self.__ev_y = [j for i in self.__ev_data[:, 1:2] for j in i]
        self.__ev_ld = [j for i in self.__ev_data[:, 2:3] for j in i]
        self.__ev_ct = [j for i in self.__ev_data[:, 3:4] for j in i]
        self.__cs_x = [j for i in self.__cs_data[:, 0:1] for j in i]
        self.__cs_y = [j for i in self.__cs_data[:, 1:2] for j in i]
        self.__cp = [j for i in self.__cp_data[:, 0:1] for j in i]
        self.__distance = np.zeros((len(self.__ev_data),
                                    len(self.__cs_data)))
        self.__reachability = np.zeros((len(self.__ev_data),
                                        len(self.__cs_data)))
        self.__reachable_cs_for_ev = list()
        self.__schedulable_ev_list = list()
        self.__transfer_n_o = dict()
        self.__transfer_o_n = dict()
        for i in range(len(self.__ev_data)):
            temp_reachable_cs_for_ev = list()
            for j in range(len(self.__cs_data)):

                d = np.sqrt((self.__ev_x[i] - self.__cs_x[j]) ** 2 + (self.__ev_y[i] - self.__cs_y[j]) ** 2)
                if d < self.__ev_ld[i]:
                    self.__reachability[i][j] = 1.0
                    temp_reachable_cs_for_ev.append(j)
                else:
                    self.__reachability[i][j] = 0.0
                self.__distance[i][j] = d
            self.__reachable_cs_for_ev.append(temp_reachable_cs_for_ev)
            if np.sum(self.__reachability[i]) != 0:
                self.__schedulable_ev_list.append(i)
        for it in range(len(self.__schedulable_ev_list)):
            self.__transfer_o_n[it] = self.__schedulable_ev_list[it]
            self.__transfer_n_o[self.__schedulable_ev_list[it]] = it

        self.__SI = np.zeros((len(self.__cs_data),
                              self.__slot_number,
                              len(self.__schedulable_ev_list)),
                             dtype=int)
        self.__SIP = np.zeros((len(self.__cs_data),
                               self.__slot_number),
                              dtype=int)
        self.__time = np.zeros((len(self.__cs_data),
                                self.__slot_number,
                                3))
        self.__ev_n = np.zeros((len(self.__cs_data),
                                self.__slot_number))
        self.__cd_wt = np.zeros((len(self.__cs_data),
                                 self.__slot_number))
        self.__cd_st = np.zeros((len(self.__cs_data),
                                 self.__slot_number))
        self.__ev_cs = np.zeros(len(self.__cs_data))
        self.__time_for_ev = np.zeros((len(self.__cs_data),
                                       self.__slot_number,
                                       len(self.__schedulable_ev_list),
                                       3))

        self.__st = 0
        self.__qt = 0
        self.__tt = 0

        self.__average_speed = 1.0
        self.__index = 0
        self.__not_scheduled_ev = list()
        self.__scheduled_ev = list()
        self.__trace = np.zeros((len(self.__schedulable_ev_list) + 1,
                                 3))
        self.__backup_for_schedulable_ev_number = copy.deepcopy(self.__schedulable_ev_list)
        self.__not_scheduled_ev = copy.deepcopy(self.__schedulable_ev_list)
        self.__whether_ev_was_scheduled = np.array(
            len(self.__schedulable_ev_list) * [0.0])
        self.__teacher_version = teacher_version
        if not self.__teacher_version:
            self.calculate()
        else:
            self.calculate_teacher_version()
        self.__state = self.get_current_ev_state()
        self.__basedir = r"E:\EV_Charging_Scheduling"
        self.filelists = []
        self.whitelist = ['php', 'py']
        self.hour_charging_ev = 0

    def calculate(self) -> None:
        self.__ev_n = np.zeros((len(self.__cs_data),
                                self.__slot_number))
        self.__cd_wt = np.zeros((len(self.__cs_data),
                                 self.__slot_number))
        self.__cd_st = np.zeros((len(self.__cs_data),
                                 self.__slot_number))
        self.__st = 0
        self.__qt = 0
        self.__tt = 0
        self.__time = np.zeros((len(self.__cs_data),
                                self.__slot_number,
                                3))
        self.__time_for_ev = np.zeros((len(self.__cs_data),
                                       self.__slot_number,
                                       len(self.__schedulable_ev_list),
                                       3))
        for i in range(len(self.__cs_data)):
            for j in range(self.__slot_number):
                for k in range(int(self.__SIP[i][j])):
                    if (self.__distance[int(self.__SI[i][j][k])][i] / self.__average_speed) < self.__cd_wt[i][j]:
                        # if queueing is needed
                        self.__qt += self.__cd_wt[i][j] - self.__distance[int(self.__SI[i][j][k])][i]
                        self.__st += 0
                        self.__tt += self.__cd_wt[i][j] + self.__ev_ct[int(self.__SI[i][j][k])]
                        self.__time_for_ev[i][j][k][0] = self.__cd_wt[i][j] - self.__distance[int(self.__SI[i][j][k])][
                            i]
                        self.__time_for_ev[i][j][k][1] = 0
                        self.__time_for_ev[i][j][k][2] = self.__ev_ct[int(self.__SI[i][j][k])]
                        self.__time[i][j][0] += (self.__cd_wt[i][j] - self.__distance[int(self.__SI[i][j][k])][i])
                        self.__time[i][j][1] += 0
                        self.__time[i][j][2] += self.__ev_ct[int(self.__SI[i][j][k])]
                        self.__cd_st[i][j] += 0
                        self.__cd_wt[i][j] += self.__ev_ct[int(self.__SI[i][j][k])]
                    else:  # if queueing is not needed
                        self.__qt += 0
                        self.__st += self.__distance[int(self.__SI[i][j][k])][i] - self.__cd_wt[i][j]
                        self.__tt += self.__distance[int(self.__SI[i][j][k])][i] + self.__ev_ct[int(self.__SI[i][j][k])]
                        self.__time_for_ev[i][j][k][0] = 0
                        self.__time_for_ev[i][j][k][1] = self.__distance[int(self.__SI[i][j][k])][i] - \
                                                         self.__cd_wt[i][j]
                        self.__time_for_ev[i][j][k][2] = self.__ev_ct[int(self.__SI[i][j][k])]
                        self.__time[i][j][0] += 0
                        self.__time[i][j][1] += (self.__distance[int(self.__SI[i][j][k])][i] - self.__cd_wt[i][j])
                        self.__time[i][j][2] += self.__ev_ct[int(self.__SI[i][j][k])]
                        self.__cd_st[i][j] += (self.__distance[int(self.__SI[i][j][k])][i] - self.__cd_wt[i][j])
                        self.__cd_wt[i][j] = (self.__distance[int(self.__SI[i][j][k])][i] +
                                              self.__ev_ct[int(self.__SI[i][j][k])])
        self.__state = self.get_current_ev_state()
        return None

    def calculate_teacher_version(self) -> None:
        self.__ev_n = np.zeros((len(self.__cs_data),
                                self.__slot_number))
        self.__cd_wt = np.zeros((len(self.__cs_data),
                                 self.__slot_number))
        self.__cd_st = np.zeros((len(self.__cs_data),
                                 self.__slot_number))
        self.__st = 0
        self.__qt = 0
        self.__tt = 0
        self.__time = np.zeros((len(self.__cs_data),
                                self.__slot_number,
                                3))
        self.__time_for_ev = np.zeros((len(self.__cs_data),
                                       self.__slot_number,
                                       len(self.__schedulable_ev_list),
                                       3))
        for i in range(len(self.__cs_data)):
            for j in range(self.__slot_number):

                for k in range(int(self.__SIP[i][j])):
                    if (self.__distance[int(self.__SI[i][j][k])][i] / self.__average_speed) < self.__cd_wt[i][j]:

                        self.__qt += self.__cd_wt[i][j] - self.__distance[int(self.__SI[i][j][k])][i]
                        self.__st += 0
                        self.__tt += self.__cd_wt[i][j] + self.__ev_ct[self.__transfer_o_n[i]]
                        self.__time_for_ev[i][j][k][0] = self.__cd_wt[i][j] - self.__distance[int(self.__SI[i][j][k])][
                            i]
                        self.__time_for_ev[i][j][k][1] = 0
                        self.__time_for_ev[i][j][k][2] = self.__ev_ct[self.__transfer_o_n[i]]
                        self.__time[i][j][0] += (self.__cd_wt[i][j] - self.__distance[int(self.__SI[i][j][k])][i])
                        self.__time[i][j][1] += 0
                        self.__time[i][j][2] += self.__ev_ct[self.__transfer_o_n[i]]
                        self.__cd_st[i][j] += 0
                        self.__cd_wt[i][j] += self.__ev_ct[self.__transfer_o_n[i]]
                    else:
                        self.__qt += 0
                        self.__st += self.__distance[int(self.__SI[i][j][k])][i] - self.__cd_wt[i][j]
                        self.__tt += self.__distance[int(self.__SI[i][j][k])][i] + self.__ev_ct[self.__transfer_o_n[i]]
                        self.__time_for_ev[i][j][k][0] = 0
                        self.__time_for_ev[i][j][k][1] = self.__distance[int(self.__SI[i][j][k])][i] - \
                                                         self.__cd_wt[i][j]
                        self.__time_for_ev[i][j][k][2] = self.__ev_ct[self.__transfer_o_n[i]]
                        self.__time[i][j][0] += 0
                        self.__time[i][j][1] += (self.__distance[int(self.__SI[i][j][k])][i] - self.__cd_wt[i][j])
                        self.__time[i][j][2] += self.__ev_ct[self.__transfer_o_n[i]]
                        self.__cd_st[i][j] += (self.__distance[int(self.__SI[i][j][k])][i] - self.__cd_wt[i][j])
                        self.__cd_wt[i][j] = (self.__distance[int(self.__SI[i][j][k])][i] +
                                              self.__ev_ct[self.__transfer_o_n[i]])
        self.__state = self.get_current_ev_state()
        return None

    def count(self) -> None:
        self.get_file()
        totalline = 0
        for filelist in self.filelists:
            totalline += self.count_line(filelist)
        print('total lines:', totalline)
        return None

    @staticmethod
    def count_line(fname) -> int:
        count = 0
        for file_line in open(fname, encoding="utf=8").readlines():
            if file_line != '':
                count += 1
        print(fname + '----', count)
        return count

    def find_position(self,
                      ev_number: int) -> Tuple:
        assert ev_number in self.__scheduled_ev
        cs_number = None
        slot_number = None
        __index = None
        charging_order = None
        for i in range(len(self.__scheduled_ev)):
            if self.__trace[i][0] == ev_number:
                __index = i
                cs_number = self.__trace[i][1]
                slot_number = self.__trace[i][2]
        for i in range(int(self.__SIP[int(cs_number)][int(slot_number)])):
            if self.__SI[int(cs_number)][int(slot_number)][i] == ev_number:
                charging_order = i
        return int(__index), int(ev_number), int(cs_number), int(slot_number), int(charging_order)

    def get_average_charging_time(self,
                                  cs_number: int) -> float:
        assert cs_number in range(len(self.__cs_data))
        c_time = 0
        num = 0
        for i in range(self.__slot_number):
            c_time += self.__time[cs_number][i][2]
            num += self.__SIP[cs_number][i]
        average_charging_time = c_time / num
        return average_charging_time


    def get_average_distance_of_ev_to_cs(self,
                                         ev_number: int) -> float:
        distance = self.__distance[ev_number]
        sum_ = 0
        count = 0
        for i in range(len(distance)):
            if self.__reachability[ev_number][i] == 1.0:
                sum_ += distance[i]
                count += 1
        sum_ /= count
        return sum_

    def get_average_first_k_distance_of_ev_to_cs(self,
                                                 ev_number: int,
                                                 k: int) -> float:
        distance = self.__distance[ev_number]
        sum_ = 0
        count = 0
        for i in range(len(distance)):
            if self.__reachability[ev_number][i] == 1.0:
                sum_ += distance[i]
                count += 1
                if count >= k:
                    break
        sum_ /= count
        return sum_

    def get_average_idling_time(self,
                                cs_number: int) -> float:
        assert cs_number in range(len(self.__cs_data))
        i_time = 0
        num = 0
        for i in range(self.__slot_number):
            i_time += self.__time[cs_number][i][1]
            num += self.__SIP[cs_number][i]
        average_idle_time = i_time / num
        return average_idle_time

    def get_average_queueing_time(self,
                                  cs_number: int) -> float:
        assert cs_number in range(len(self.__cs_data))
        q_time = 0
        num = 0
        for i in range(self.__slot_number):
            q_time += self.__time[cs_number][i][0]
            num += self.__SIP[cs_number][i]
        average_queueing_time = q_time / num
        return average_queueing_time

    def get_average_time(self) -> Tuple:
        ave_queue = self.__qt / len(self.__scheduled_ev)
        ave_idle = self.__st / (len(self.__cs_data) * self.__slot_number)
        ave_charging = 0
        for i in range(len(self.__scheduled_ev)):
            ave_charging += self.__ev_ct[self.__scheduled_ev[i]]
        ave_charging /= len(self.__scheduled_ev)
        return ave_idle, ave_queue, ave_charging

    def get_average_time_for_cs(self) -> Tuple:
        average_charging_time = np.zeros(len(self.__cs_data))
        average_idling_time = np.zeros(len(self.__cs_data))
        average_queueing_time = np.zeros(len(self.__cs_data))
        for i in range(len(self.__cs_data)):
            average_charging_time_temp = 0.0
            average_idling_time_temp = 0.0
            average_queueing_time_temp = 0.0
            for j in range(self.__slot_number):
                average_queueing_time_temp += self.__time[i][j][0]
                average_idling_time_temp += self.__time[i][j][1]
                average_charging_time_temp += self.__time[i][j][2]
            average_charging_time_temp /= self.__slot_number
            average_idling_time_temp /= self.__slot_number
            average_queueing_time_temp /= self.__slot_number
            average_charging_time[i] = average_charging_time_temp
            average_idling_time[i] = average_idling_time_temp
            average_queueing_time[i] = average_queueing_time_temp
        return average_queueing_time, average_idling_time, average_charging_time

    def get_best_slot(self,
                      cs_number: int) -> int:
        assert cs_number in range(len(self.__cs_data))
        best_slot = int(sum(self.__SIP[cs_number]) % self.__slot_number)
        return best_slot

    def get_brief_time_matrix(self) -> np.ndarray:
        ret = self.__time
        return ret

    def get_charging_ev_number_for_concrete_cs(self,
                                               cs_number: int) -> int:
        sum_ = 0
        for i in range(self.__slot_number):
            sum_ += self.__SIP[cs_number][i]
        return sum_

    def get_charging_time(self,
                          ev_number: int) -> float:
        return self.__ev_ct[ev_number]

    def get_cs_cp_coordination(self,
                               cs_number: int):
        price = self.__cp[cs_number]
        return price

    def get_cs_x_coordination(self,
                              cs_number: int):
        x = self.__cs_x[cs_number]
        return x

    def get_cs_y_coordination(self,
                              cs_number: int):
        y = self.__cs_y[cs_number]
        return y

    def get_current_cs_state(self,
                             sel_ev_number: int) -> np.ndarray:
        assert sel_ev_number in self.__schedulable_ev_list
        ev_left_travel_distance = np.array([self.__ev_ld[i] for i in self.__backup_for_schedulable_ev_number])
        ev_expecting_charging_time = np.array([self.__ev_ct[i] for i in self.__backup_for_schedulable_ev_number])
        distance_between_ev_and_cs = np.array(self.__distance)[self.__backup_for_schedulable_ev_number]
        for i in range(len(self.__schedulable_ev_list)):
            if self.__transfer_o_n[i] in self.__scheduled_ev:
                ev_left_travel_distance[i] *= 0.0
                ev_expecting_charging_time[i] *= 0.0
                distance_between_ev_and_cs[i] *= 0.0
        average_queueing_time = np.concatenate(
            (
                self.get_average_time_for_cs()[0],
                np.array([0.0, 0.0])
            ),
            axis=-1
        )
        average_idling_time = np.concatenate(
            (
                self.get_average_time_for_cs()[1],
                np.array([0.0, 0.0])
            ),
            axis=-1
        )
        charging_number_on_every_cs = np.concatenate(
            (
                np.sum(self.__SIP, axis=-1),
                np.array([0.0, 0.0])
            ),
            axis=-1
        )
        charging_price_of_every_cs = np.concatenate(
            (
                np.array(self.__cp),
                np.array([0.0, 0.0])
            ),
            axis=-1
        )
        occupied_time_of_every_cs = np.concatenate(
            (
                np.sum(self.__cd_wt, axis=1) / 5,
                np.array([0.0, 0.0])
            )
        )
        ev_part = np.concatenate(
            (
                distance_between_ev_and_cs,
                ev_expecting_charging_time.reshape(-1, 1),
                ev_left_travel_distance.reshape(-1, 1),
            ),
            axis=-1
        )[self.__transfer_n_o[sel_ev_number]].reshape(1, -1)
        cs_part = np.concatenate(
            (
                average_queueing_time.reshape(1, -1),
                average_idling_time.reshape(1, -1),
                charging_number_on_every_cs.reshape(1, -1),
                # charging_price_of_every_cs.reshape(1, -1)
                occupied_time_of_every_cs.reshape(1, -1)
            ),
            axis=0
        )
        cs_state = np.concatenate(
            (
                ev_part,
                cs_part
            ),
            axis=0
        )[np.newaxis, :]
        return cs_state

    def get_current_ev_state(self) -> np.ndarray:
        ev_left_travel_distance = np.array([self.__ev_ld[i] for i in self.__backup_for_schedulable_ev_number])
        ev_expecting_charging_time = np.array([self.__ev_ct[i] for i in self.__backup_for_schedulable_ev_number])
        distance_between_ev_and_cs = np.array(self.__distance)[self.__backup_for_schedulable_ev_number]
        for i in range(len(self.__schedulable_ev_list)):
            if self.__transfer_o_n[i] in self.__scheduled_ev:
                ev_left_travel_distance[i] *= 0.0
                ev_expecting_charging_time[i] *= 0.0
                distance_between_ev_and_cs[i] *= 0.0
        average_queueing_time = np.concatenate(
            (
                self.get_average_time_for_cs()[0],
                np.array([0.0, 0.0])
            ),
            axis=-1
        )
        average_idling_time = np.concatenate(
            (
                self.get_average_time_for_cs()[1],
                np.array([0.0, 0.0])
            ),
            axis=-1
        )
        charging_number_on_every_cs = np.concatenate(
            (
                np.sum(self.__SIP, axis=-1),
                np.array([0.0, 0.0])
            ),
            axis=-1
        )
        charging_price_of_every_cs = np.concatenate(
            (
                np.array(self.__cp),
                np.array([0.0, 0.0])
            ),
            axis=-1
        )
        occupied_time_of_every_cs = np.concatenate(
            (
                np.sum(self.__cd_wt, axis=1) / 5,
                np.array([0.0, 0.0])
            )
        )
        ev_part = np.concatenate(
            (
                distance_between_ev_and_cs,
                ev_expecting_charging_time.reshape(-1, 1),
                ev_left_travel_distance.reshape(-1, 1),
            ),
            axis=-1
        )
        cs_part = np.concatenate(
            (
                average_queueing_time.reshape(1, -1),
                average_idling_time.reshape(1, -1),
                charging_number_on_every_cs.reshape(1, -1),
                occupied_time_of_every_cs.reshape(1, -1)
            ),
            axis=0
        )
        ev_state = np.concatenate(
            (
                ev_part,
                cs_part
            ),
            axis=0
        )[np.newaxis, :]
        return ev_state

    def get_detailed_time_matrix(self) -> np.ndarray:
        ret = self.__time_for_ev
        return ret

    def get_distance(self,
                     ev_number: int,
                     cs_number: int) -> float:
        distance = self.__distance[ev_number][cs_number]
        return distance

    def get_ev_charging_time(self,
                             ev_number: int) -> float:
        return self.__ev_ct[ev_number]

    def get_ev_left_travel_distance(self,
                                    ev_number: int) -> float:
        return self.__ev_ld[ev_number]

    def get_ev_position(self,
                        ev_number: int) -> Tuple:
        x = self.__ev_x[ev_number]
        y = self.__ev_y[ev_number]
        return x, y

    def get_ev_x_coordination(self,
                              ev_number: int) -> float:
        x = self.__ev_x[ev_number]
        return x

    def get_ev_y_coordination(self,
                              ev_number: int) -> float:
        y = self.__ev_y[ev_number]
        return y

    def get_file(self) -> None:
        for parent, dirnames, filenames in os.walk(self.__basedir):
            for filename in filenames:
                ext = filename.split('.')[-1]
                if ext in self.whitelist:
                    self.filelists.append(os.path.join(parent, filename))
        return None

    def get_how_many_ev_need_to_be_scheduled(self) -> int:
        return len(self.__not_scheduled_ev)

    def get_how_many_ev_have_been_scheduled(self) -> int:
        return len(self.__scheduled_ev)

    def get_idling_time(self) -> float:
        return self.__st


    def get_not_scheduled_ev(self) -> list:
        return self.__not_scheduled_ev


    def get_queueing_time(self):
        return self.__qt

    def get_raw_state_matrix(self) -> np.ndarray:
        return self.__state

    def get_reachable_cs_list_for_ev(self,
                                     ev_number: int) -> np.ndarray:
        reachable_cs_list_for_ev = np.array(self.__reachable_cs_for_ev)[ev_number]
        return reachable_cs_list_for_ev

    def get_reachability_of_an_ev(self,
                                  ev_number: int) -> np.ndarray:
        reachability = np.array(self.__reachability[ev_number])
        return reachability

    def get_reachability_matrix(self) -> np.ndarray:
        return self.__reachability

    def get_reward(self,
                   q: float = 0.8,
                   i: float = 0.2
                   ) -> float:
        reward = self.__qt / len(self.__scheduled_ev) * q + self.__st / (len(self.__cs_data) * self.__slot_number) * i
        return reward

    def get_reward_for_cs(self,
                          cs_number: int,
                          ev_number: int) -> float:
        distance = self.__distance[ev_number][cs_number]
        occupied = np.sum(self.__cd_wt[cs_number]) / self.__slot_number
        reward = occupied - distance
        return reward

    def get_reward_for_ev(self,
                          ev_number: int,
                          cs_number: int,
                          slot_number: int,
                          q: float = 0.8,
                          i: float = 0.2
                          ) -> float:
        idle = 0
        queue = 0
        for pointer in range(int(self.__SIP[cs_number][slot_number])):
            if self.__SI[cs_number][slot_number][pointer] == ev_number:
                queue = self.__time_for_ev[cs_number][slot_number][pointer][0]
                idle = self.__time_for_ev[cs_number][slot_number][pointer][1]
                break
        weighted_reward = idle * i + queue * q
        return weighted_reward

    def get_schedulable_ev(self) -> np.ndarray:
        result = np.array(self.__schedulable_ev_list)
        return result

    def get_scheduled_ev(self) -> list:
        return self.__scheduled_ev

    def get_sip(self) -> np.ndarray:
        return self.__SIP

    def get_slot_count_on_one_charging_station(self) -> int:
        return self.__slot_number

    def get_travel_time_from_ev_to_cs(self,
                                      ev_number: int,
                                      cs_numer: int) -> float:
        return self.__distance[ev_number][cs_numer]


    def is_done(self) -> bool:
        boolean_value = len(self.__not_scheduled_ev) == 0
        return boolean_value

    def optimize(self) -> List:
        old_result = []
        new_result = []
        for i in range(len(self.__cs_x)):
            sub_ = []
            for j in range(self.__slot_number):
                if self.__SIP[i][j] == 0:
                    pass
                if self.__SIP[i][j] != 0:
                    r_in_slot_ = self.optimize_(i, j)
                    sub_.append(r_in_slot_)
            new_result.append(sub_)
        self.reset()
        for i in range(len(new_result)):
            for j in range(len(new_result[i])):
                for k in range(len(new_result[i][j])):
                    self.step(new_result[i][j][k], i, j)
        return new_result

    def optimize_(self,
                  cs_number: int,
                  sl_number: int) -> List:  ## old

        assert self.__SIP[cs_number][sl_number] != 0
        SI = self.__SI[cs_number][sl_number]
        SIP = self.__SIP[cs_number][sl_number]
        distance = []
        for i in range(SIP):
            distance.append(self.get_distance(SI[i], cs_number))
        ev_no_list = []
        occup_list = []
        final = []
        cs_occup = 0
        for i in range(SIP):
            charging_time = self.get_ev_charging_time(SI[i])
            travel_time = self.get_distance(SI[i], cs_number)
            ev_no_list.append(SI[i])
            occup_list.append(charging_time + 2*travel_time)
        first_ev = True
        for i in range(SIP):
            if cs_occup == 0:
                ev_index = distance.index(min(distance))
                ev_no = ev_no_list[ev_index]
            else:
                ev_index = occup_list.index(min(occup_list))
                ev_no = ev_no_list[ev_index]
            ev_no_list.remove(ev_no)
            occup_list.remove(occup_list[ev_index])
            final.append(ev_no)
            if first_ev:
                cs_occup += self.__distance[ev_no][cs_number] + self.get_ev_charging_time(ev_no)
                first_ev = False
            else:
                if cs_occup >= self.__distance[ev_no][cs_number]:
                    cs_occup += self.get_ev_charging_time(ev_no)
                else:
                    cs_occup = self.__distance[ev_no][cs_number] + self.get_ev_charging_time(ev_no)
            for i_ in range(len(occup_list)):
                ev = ev_no_list[i_]
                cs = cs_number
                if cs_occup >= self.__distance[ev][cs]:
                    occup_list[ev_no_list.index(ev)] = self.get_ev_charging_time(ev)
                else:
                    occup_list[ev_no_list.index(ev)] = (self.get_ev_charging_time(ev) + 2 * (self.__distance[ev][cs] - cs_occup))
        return final

    def print_scheduling_info_for_concrete_slot(self,
                                                cs_number: int,
                                                slot_number: int) -> None:

        print("CS--{:5s},SLOT--{:5s}--({:})---->".format(str(cs_number),
                                                         str(slot_number),
                                                         int(self.__SIP[cs_number][slot_number])), end=" ")
        for k in range(int(self.__SIP[cs_number][slot_number])):
            print(int(self.__SI[cs_number][slot_number][k]))
        return None

    def print_scheduling_info(self) -> None:
        for i in range(len(self.__cs_data)):
            for j in range(self.__slot_number):
                print("CS--{:5s},SLOT--{:5s}--({:})---->".format(str(i),
                                                                 str(j),
                                                                 int(self.__SIP[i][j])), end=" ")
                for k in range(int(self.__SIP[i][j])):
                    print(int(self.__SI[i][j][k]), end="\t")
                print("")
        return None

    def print_scheduling_info_for_one_cs(self,
                                         cs_number: int) -> None:
        for j in range(self.__slot_number):
            print("CS--{:5s},SLOT--{:5s}--({:})---->".format(str(cs_number),
                                                             str(j),
                                                             int(self.__SIP[cs_number][j])), end=" ")
            for k in range(int(self.__SIP[cs_number][j])):
                print(int(self.__SI[cs_number][j][k]), end="\t")
            print("")
        return None

    def print_scheduling_consequence_info(self) -> None:
        print("|||",
              "average idling time = {:.3f}".format(self.__st / 170),
              "|||",
              "average queueing time = {:.3f}".format(self.__qt / 899),
              "|||", end="\t")
        return None

    def print_scheduling_consequence_list(self) -> list:
        assert self.is_done() == True
        result = []
        all_ev = 0
        for i in range(34):
            for j in range(5):
                for k in range(self.__SIP[i][j]):
                    temp = [self.__SI[i][j][k], i, j]
                    result.append(temp)
                    all_ev += 1
                    if (all_ev + 1) % 20 == 0:
                        print()
        return result

    def print_state_of_cs(self,
                          sel_ev_number: int) -> None:
        pd.set_option('display.max_columns',
                      None)
        pd.set_option('display.max_rows',
                      None)
        print(pd.DataFrame(data=self.get_current_cs_state(sel_ev_number=sel_ev_number)))
        return None


    def print_state_of_ev(self) -> None:
        pd.set_option('display.max_columns',
                      None)
        pd.set_option('display.max_rows',
                      None)
        print(pd.DataFrame(data=self.get_current_ev_state()))
        return None

    def reset(self) -> None:
        self.__init__(self.__filename_of_ev_information,
                      self.__filename_of_cs_information,
                      self.__filename_of_cp_information)
        return None

    def ret_scheduling_consequence_list(self) -> list:
        assert self.is_done() == True
        result = []
        for i in range(34):
            for j in range(5):
                for k in range(self.__SIP[i][j]):
                    temp = [self.__SI[i][j][k], i, j]
                    result.append(temp)
        return result

    def step(self,
             ev_number: int,
             cs_number: int,
             slot_number: int) -> None:
        assert self.__reachability[ev_number][cs_number] == 1.0
        assert ev_number in self.__not_scheduled_ev
        pointer = int(self.__SIP[cs_number][slot_number])
        self.__SI[cs_number][slot_number][pointer] = ev_number
        self.__SIP[cs_number][slot_number] += 1
        self.__scheduled_ev.append(ev_number)
        self.__not_scheduled_ev.remove(ev_number)
        self.__trace[self.__index][0] = ev_number
        self.__trace[self.__index][1] = cs_number
        self.__trace[self.__index][2] = slot_number
        self.__index += 1
        self.__whether_ev_was_scheduled[self.__transfer_n_o[ev_number]] = 1.0
        self.__ev_n[cs_number][slot_number] += 1
        self.__ev_cs[cs_number] += 1

        if self.__cd_wt[cs_number][slot_number] < 60:
            self.hour_charging_ev += 1

        if not self.__teacher_version:
            self.calculate()
        else:
            self.calculate_teacher_version()
        return None

    def get_hour_charing_ev(self):
        return self.hour_charging_ev

    def transfer_ev_no_to_order(self,
                                ev_number: int) -> int:
        assert ev_number in self.__schedulable_ev_list
        return self.__transfer_n_o[ev_number]

    def transfer_ev_order_to_no(self,
                                ev_order: int) -> int:
        return self.__transfer_o_n[ev_order]

    def unstep(self,
               ev_number: int) -> None:
        assert ev_number in self.__scheduled_ev
        __index, ev_number, cs_number, slot_number, charging_order = self.find_position(ev_number)
        for i in range(charging_order, self.__SIP[cs_number][slot_number]):
            self.__SI[cs_number][slot_number][i] = self.__SI[cs_number][slot_number][i + 1]
        self.__SI[cs_number][slot_number][self.__SIP[cs_number][slot_number]] = 0
        self.__SIP[cs_number][slot_number] -= 1
        for i in range(__index, len(self.__scheduled_ev)):
            self.__trace[i] = self.__trace[i + 1]
        self.__scheduled_ev.remove(ev_number)
        self.__not_scheduled_ev.append(ev_number)
        self.__index -= 1
        self.__whether_ev_was_scheduled[self.__transfer_n_o[ev_number]] = 0.0
        self.__ev_n[cs_number][slot_number] -= 1
        self.__ev_cs[cs_number] -= 1
        if not self.__teacher_version:
            self.calculate()
        else:
            self.calculate_teacher_version()
        return None
