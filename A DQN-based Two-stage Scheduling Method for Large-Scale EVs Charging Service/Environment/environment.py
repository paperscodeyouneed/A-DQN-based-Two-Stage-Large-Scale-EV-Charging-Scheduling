""" This python script is used to define an environment for execute reinforcement learning,
    contains 1122 lines and should not be trimmed"""

import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import _pickle as cPickle
# import prettytable as pt
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
#from Utils.segment_tree import MinSegmentTree, SumSegmentTree
from torch.nn.utils import clip_grad, clip_grad_norm, clip_grad_norm_, clip_grad_value_
from typing import Callable, Dict, List, NamedTuple, Tuple

np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")


class Environment(object):
    """
        define a class used as a simple Rl-Environment, around 50 useful method was defined
    """

    # 88.6 ms ± 228 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    def __init__(self,
                 filename_of_ev_information: str = "../Data/ev_data_1000.csv",
                 filename_of_cs_information: str = "../Data/charging_stations.csv",
                 filename_of_cp_information: str = "../Data/ev_cp_34.csv",
                 slot_number: int = 5,
                 teacher_version: bool = True) -> None:
        """
            initialize all the variables that scheduling environment needs and get filenames that store information
        """
        assert filename_of_cs_information is not None
        assert filename_of_cp_information is not None
        assert filename_of_ev_information is not None
        assert slot_number is not None and slot_number != 0
        self.__filename_of_cp_information = filename_of_cp_information
        self.__filename_of_cs_information = filename_of_cs_information
        self.__filename_of_ev_information = filename_of_ev_information
        self.__slot_number = slot_number
        # reading datas from csv files
        self.__cp_data = pd.read_csv(self.__filename_of_cp_information,
                                     delimiter=",").iloc[:, 1:].values  # get raw charging price
        self.__cs_data = pd.read_csv(self.__filename_of_cs_information,
                                     delimiter=";",
                                     header=None).iloc[:, 1:].values  # get raw charging station data
        self.__ev_data = pd.read_csv(self.__filename_of_ev_information,
                                     delimiter=",").iloc[:, [1, 2, 3, 4]].values  # get raw ev data
        # transfer each kind of data into 1-dimensional list
        self.__ev_x = [j for i in self.__ev_data[:, 0:1] for j in i]  # x-axis of ev
        self.__ev_y = [j for i in self.__ev_data[:, 1:2] for j in i]  # y-axis of ev
        self.__ev_ld = [j for i in self.__ev_data[:, 2:3] for j in i]  # remaining traveling distance of tram
        self.__ev_ct = [j for i in self.__ev_data[:, 3:4] for j in i]  # estimated charging time of tram
        self.__cs_x = [j for i in self.__cs_data[:, 0:1] for j in i]  # x-axis of cs
        self.__cs_y = [j for i in self.__cs_data[:, 1:2] for j in i]  # y-axis of cs
        self.__cp = [j for i in self.__cp_data[:, 0:1] for j in i]  # charging price of charging station
        # calculate accessibility matrix, distance matrix and number conversion dictionary
        self.__distance = np.zeros((len(self.__ev_data),
                                    len(self.__cs_data)))  # distance matrix
        self.__reachability = np.zeros((len(self.__ev_data),
                                        len(self.__cs_data)))  # reachability matrix
        self.__reachable_cs_for_ev = list()  # reachable cs number for all ev
        self.__schedulable_ev_list = list()  # schedulable ev list in real number
        self.__transfer_n_o = dict()  # dict used to transfer ev number into relative ev number
        self.__transfer_o_n = dict()  # dict used to transfer relative ev number into ev number
        for i in range(len(self.__ev_data)):
            temp_reachable_cs_for_ev = list()
            for j in range(len(self.__cs_data)):
                # calculate reachability and other neccessary features between every cs and ev
                d = np.sqrt((self.__ev_x[i] - self.__cs_x[j]) ** 2 + (self.__ev_y[i] - self.__cs_y[j]) ** 2)
                if d < self.__ev_ld[i]:  # generate reachability
                    self.__reachability[i][j] = 1.0
                    temp_reachable_cs_for_ev.append(j)
                else:
                    self.__reachability[i][j] = 0.0
                self.__distance[i][j] = d
            self.__reachable_cs_for_ev.append(temp_reachable_cs_for_ev)
            if np.sum(self.__reachability[i]) != 0:
                self.__schedulable_ev_list.append(i)  # if an ev is schedulable, then record it
        # calculate trasnfer dictionary for relative ev order and real ev order
        for it in range(len(self.__schedulable_ev_list)):
            self.__transfer_o_n[it] = self.__schedulable_ev_list[it]
            self.__transfer_n_o[self.__schedulable_ev_list[it]] = it
        # define some lists and matrices to store scheduling information
        self.__SI = np.zeros((len(self.__cs_data),
                              self.__slot_number,
                              len(self.__schedulable_ev_list)),
                             dtype=int)  # SI == Scheduled Ev Information
        self.__SIP = np.zeros((len(self.__cs_data),
                               self.__slot_number),
                              dtype=int)  # SIP == Scheduled Ev Information Pointer
        self.__time = np.zeros((len(self.__cs_data),
                                self.__slot_number,
                                3))  # used to store accumulated time information about scheduling
        # define matrices used in the time infomation calculation
        self.__ev_n = np.zeros((len(self.__cs_data),
                                self.__slot_number))  # store information about the number of dispatched trams for posts
        self.__cd_wt = np.zeros((len(self.__cs_data),
                                 self.__slot_number))  # store the time occupied of any slot
        self.__cd_st = np.zeros((len(self.__cs_data),
                                 self.__slot_number))  # store the spare time of any slot
        self.__ev_cs = np.zeros(len(self.__cs_data))  # store number of ev charging at any charging station
        self.__time_for_ev = np.zeros((len(self.__cs_data),
                                       self.__slot_number,
                                       len(self.__schedulable_ev_list),
                                       3))  # used to store queueing idle  charging time for ev in every slot of cs
        # initialize the variables used to store middle result in ev scheduling
        self.__st = 0  # total spare time of one trajectory
        self.__qt = 0  # total queueing time of one trajectory
        self.__tt = 0  # total wasted time for generating a trajectory
        # initialize other neccessary variables used in this environemt
        self.__average_speed = 1.0  # define the default average ev running speed when driving on streets
        self.__index = 0  # trace tracking index for one trajectory
        self.__not_scheduled_ev = list()  # used to store the ev number that was not scheduled yet
        self.__scheduled_ev = list()  # used to record the scheduled-ev ev_number
        self.__trace = np.zeros((len(self.__schedulable_ev_list) + 1,  # +1 for special handle in unstep
                                 3))  # used to record details about scheduling process
        # initialize state and backup-infomation for every ev with ev-number
        self.__backup_for_schedulable_ev_number = copy.deepcopy(self.__schedulable_ev_list)
        self.__not_scheduled_ev = copy.deepcopy(self.__schedulable_ev_list)
        self.__whether_ev_was_scheduled = np.array(
            len(self.__schedulable_ev_list) * [0.0])  # used to record whether an ev was scheduled
        self.__teacher_version = teacher_version  # true
        if not self.__teacher_version:
            self.calculate()  # initialize all critical elements related to "time"
        else:
            self.calculate_teacher_version()
        # define state
        self.__state = self.get_current_ev_state()  # set initial state acquired from environment
        # define variables used to calculate all line numbers
        self.__basedir = r"E:\EV_Charging_Scheduling"
        self.filelists = []
        self.whitelist = ['php', 'py']
        # 设定一个变量表示单位时间内可以调度的电动汽车数
        self.hour_charging_ev = 0

    # 14.2 ms ± 1.14 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
    def calculate(self) -> None:
        """
        calculate all time-elements changes happend in last scheduling process
        :return: None
        """
        self.__ev_n = np.zeros((len(self.__cs_data),
                                self.__slot_number))  # store information about the number of dispatched trams for posts
        self.__cd_wt = np.zeros((len(self.__cs_data),
                                 self.__slot_number))  # store the time occupied of any slot
        self.__cd_st = np.zeros((len(self.__cs_data),
                                 self.__slot_number))  # store the spare time of any slot
        self.__st = 0  # total spare time of one trajectory
        self.__qt = 0  # total queueing time of one trajectory
        self.__tt = 0  # total wasted time for generating a trajectory
        self.__time = np.zeros((len(self.__cs_data),
                                self.__slot_number,
                                3))  # used to store accumulated time information about scheduling
        self.__time_for_ev = np.zeros((len(self.__cs_data),
                                       self.__slot_number,
                                       len(self.__schedulable_ev_list),
                                       3))  # used to store queueing idle  charging time for ev in every slot of cs
        for i in range(len(self.__cs_data)):  # for all charging stations
            for j in range(self.__slot_number):  # and for all slots
                # calculate all information about time
                for k in range(int(self.__SIP[i][j])):  # for all ev that charging at this slot
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

    # 12.6 ms ± 229 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    def calculate_teacher_version(self) -> None:
        """
        calculate all time-elements changes happend in last scheduling process
        :return: None
        """
        self.__ev_n = np.zeros((len(self.__cs_data),
                                self.__slot_number))  # store information about the number of dispatched trams for posts
        self.__cd_wt = np.zeros((len(self.__cs_data),
                                 self.__slot_number))  # store the time occupied of any slot
        self.__cd_st = np.zeros((len(self.__cs_data),
                                 self.__slot_number))  # store the spare time of any slot
        self.__st = 0  # total spare time of one trajectory
        self.__qt = 0  # total queueing time of one trajectory
        self.__tt = 0  # total wasted time for generating a trajectory
        self.__time = np.zeros((len(self.__cs_data),
                                self.__slot_number,
                                3))  # used to store accumulated time information about scheduling
        self.__time_for_ev = np.zeros((len(self.__cs_data),
                                       self.__slot_number,
                                       len(self.__schedulable_ev_list),
                                       3))  # used to store queueing idle  charging time for ev in every slot of cs
        for i in range(len(self.__cs_data)):  # for all charging stations
            for j in range(self.__slot_number):  # and for all slots
                # calculate all information about time
                for k in range(int(self.__SIP[i][j])):  # for all ev that charging at this slot
                    if (self.__distance[int(self.__SI[i][j][k])][i] / self.__average_speed) < self.__cd_wt[i][j]:
                        # if queueing is needed
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
                    else:  # if queueing is not needed
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

    # it could be run in an extremely short time
    def count(self) -> None:
        """
        a function used to print the result of all lines number of this project
        :return: None
        """
        self.get_file()
        totalline = 0
        for filelist in self.filelists:
            totalline += self.count_line(filelist)
        print('total lines:', totalline)
        return None

    # it could be run in an extremely short time
    @staticmethod
    def count_line(fname) -> int:
        """
        count lines
        :param fname:
        :return:
        """
        count = 0
        for file_line in open(fname, encoding="utf=8").readlines():
            # if file_line != '' and file_line != '\n':
            if file_line != '':
                count += 1
        print(fname + '----', count)
        return count

    # 14.2 s ± 136 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    def correctness_scheduling_verification(self) -> None:
        """
        rough verification of the correctness of scheduling results. after execution, this env will be resat
        :return: whether scheduling is correct
        """
        teacher_experience = [[0, 2.8542749914893566, 124], [0, 3.260904228533951, 692], [0, 3.471349650610491, 66],
                              [0, 5.127881367716187, 192], [0, 5.337553526047183, 636], [0, 5.488284672219265, 239],
                              [0, 5.5926920231823, 155], [0, 5.71152948911729, 622], [0, 5.782234530723398, 221],
                              [0, 6.538921886929341, 827], [0, 9.28764577134645, 878], [0, 10.515463308158955, 462],
                              [0, 13.373587719310553, 153], [0, 14.413186457607742, 111], [0, 14.549853641407715, 389],
                              [0, 14.721615340788091, 43], [0, 14.743797385135629, 774], [0, 15.07343707270713, 177],
                              [0, 16.56289980385837, 89], [0, 19.2763587395647, 215], [0, 21.51791477352479, 870],
                              [0, 23.745892542799545, 375], [0, 25.801981757029782, 879], [0, 26.58906534179099, 327],
                              [0, 29.985784143164494, 525], [1, 2.247266678626551, 518], [1, 3.192277320094282, 871],
                              [1, 3.551280208657212, 701], [1, 4.388864798586297, 621], [1, 4.621958949663597, 687],
                              [1, 5.274784668537927, 225], [1, 5.519772145020828, 483], [1, 6.20989445516768, 439],
                              [1, 7.0838947481952514, 735], [1, 7.707494229796107, 571], [1, 8.727150816007939, 614],
                              [1, 9.818589833695333, 275], [1, 10.100247269958986, 664], [1, 10.492905039493282, 666],
                              [1, 10.611917306371693, 445], [1, 10.992030654792245, 574], [1, 11.532627569365888, 390],
                              [1, 13.100111885465028, 618], [1, 14.954737271414933, 384], [1, 15.751018285956459, 524],
                              [1, 17.240765909895448, 134], [1, 18.50497144036201, 582], [1, 22.13855687658344, 222],
                              [1, 22.616875880595284, 264], [1, 23.33795463121604, 855], [1, 26.68696167267619, 211],
                              [1, 26.86277283218876, 686], [1, 27.38594283884404, 440], [1, 31.343878060993234, 269],
                              [2, 1.0740130436477342, 55], [2, 2.207026658634045, 345], [2, 2.2911174901742335, 230],
                              [2, 2.730639689559021, 185], [2, 3.0636166352854444, 684], [2, 3.445091925750604, 453],
                              [2, 3.608122418948828, 306], [2, 4.623036562880656, 160], [2, 5.765686487295914, 695],
                              [2, 6.554210126955433, 588], [2, 7.4313519866400135, 738], [2, 8.00080132441466, 113],
                              [2, 8.821408329727578, 387], [2, 8.82282244758061, 867], [2, 10.534524037573535, 783],
                              [2, 10.53966250797875, 885], [2, 10.62550090921586, 515], [2, 13.223251889454772, 426],
                              [2, 15.975356131577074, 844], [2, 19.44691237949239, 691], [2, 21.64779853740284, 502],
                              [2, 21.657810491225522, 685], [2, 23.545875056896318, 487], [2, 23.568090599341122, 326],
                              [2, 24.27351791521127, 272], [2, 24.635379408139777, 485], [2, 26.90911014306495, 680],
                              [2, 27.30579324420042, 859], [2, 32.38992626902326, 138], [2, 32.630847221540066, 189],
                              [2, 36.40875054602023, 869], [3, 0.22159762365400587, 896], [3, 1.679436761570663, 532],
                              [3, 3.6836760711069285, 40], [3, 6.008709498848122, 358], [3, 8.001159495583996, 300],
                              [3, 8.910283721388044, 386], [3, 9.277456375552356, 473], [3, 10.55501630099548, 880],
                              [3, 12.046755886370706, 83], [3, 12.207171720003249, 80], [3, 13.46677414109042, 713],
                              [3, 15.508022461999257, 289], [3, 17.761800098899247, 281], [3, 19.10216915071216, 158],
                              [3, 19.710500428263842, 142], [3, 20.366893187424424, 447], [3, 20.4569839359834, 887],
                              [3, 21.03934852703164, 277], [3, 21.310522875447532, 641], [3, 24.016281378547358, 698],
                              [3, 29.135439915331947, 367], [3, 30.819754005762775, 422], [4, 1.74214844316713, 437],
                              [4, 1.7918948371775574, 194], [4, 2.165300847919102, 379], [4, 2.322076547654568, 488],
                              [4, 2.556112193013977, 39], [4, 2.679032668809943, 337], [4, 2.988937365123516, 646],
                              [4, 3.4764517024142676, 583], [4, 4.760144789149691, 334], [4, 5.050746188969361, 608],
                              [4, 5.133805559818504, 333], [4, 5.260501297334461, 654], [4, 5.309454152929756, 378],
                              [4, 5.352215050805391, 768], [4, 5.4774210153128315, 179], [4, 5.604176219084018, 464],
                              [4, 5.678841375526133, 188], [4, 6.041321180969837, 38], [4, 6.092742693337868, 665],
                              [4, 6.287891190598109, 452], [4, 6.436420863232319, 99], [4, 6.937211614055662, 767],
                              [4, 7.005304850767645, 205], [4, 7.020052447118294, 342], [4, 7.839528773280979, 468],
                              [4, 8.534791516400317, 232], [4, 8.692076454938656, 212], [4, 9.356538616870317, 433],
                              [4, 10.794122179054488, 690], [4, 11.410055850464458, 415], [4, 11.563714068111645, 324],
                              [4, 11.683093451948597, 860], [4, 12.203625033596827, 729], [4, 12.60215475050443, 800],
                              [4, 14.659380629714036, 385], [4, 15.112387734013554, 140], [4, 21.6062360276451, 533],
                              [4, 25.985517275936164, 6], [4, 29.67465771262443, 599], [4, 33.07640638503448, 593],
                              [5, 1.2291040033470175, 458], [5, 2.6597586030068525, 644], [5, 2.66300192679401, 874],
                              [5, 2.7017844363997585, 708], [5, 2.8808120703999465, 771], [5, 3.0953033486454533, 85],
                              [5, 3.1685782751200473, 341], [5, 3.300483393907268, 454], [5, 3.3148487415864905, 315],
                              [5, 3.389713249504466, 435], [5, 3.400875796200744, 529], [5, 3.574512230112647, 254],
                              [5, 3.6203618341193926, 497], [5, 4.20521777073762, 607], [5, 4.273163955647827, 723],
                              [5, 4.347493122075731, 703], [5, 4.909245774388936, 96], [5, 5.059248804480454, 746],
                              [5, 5.109367834507216, 27], [5, 5.176601329979234, 611], [5, 5.177510587130849, 490],
                              [5, 5.624610467965313, 777], [5, 6.724968933819453, 71], [5, 6.79479980106914, 531],
                              [5, 7.695751308829434, 581], [5, 8.191586218734498, 235], [5, 8.353773385868172, 26],
                              [5, 13.03561868879367, 313], [5, 13.425429965603193, 41], [5, 13.672806347877657, 846],
                              [5, 29.917939911380827, 286], [6, 1.5192225824913435, 369], [6, 2.3623696613371528, 539],
                              [6, 2.626525096842659, 354], [6, 3.4600491815899974, 554], [6, 6.814870884618313, 196],
                              [6, 7.53412081392351, 276], [6, 8.200494912560865, 821], [6, 8.323319273103925, 318],
                              [6, 8.362074322770159, 242], [6, 11.835231606531996, 163], [6, 12.458048163586074, 117],
                              [6, 13.266386548927152, 858], [6, 13.44511034212124, 648], [6, 14.800153772158017, 652],
                              [6, 15.13670331492913, 238], [6, 15.42024144226896, 734], [6, 15.469470302193248, 866],
                              [6, 16.05621541152211, 513], [6, 17.471321512818843, 237], [6, 19.94123654774079, 296],
                              [6, 20.792439400723815, 35], [6, 21.925496526348013, 294], [6, 22.245355827062724, 679],
                              [6, 22.524703893477167, 178], [6, 22.727766616579732, 716], [6, 22.920344345531888, 406],
                              [6, 23.185416118247446, 284], [6, 23.306787459248664, 166], [6, 23.548938355983367, 198],
                              [6, 24.21521864107141, 700], [6, 24.808767603981927, 338], [6, 27.094472707876417, 119],
                              [7, 1.332276825816984, 702], [7, 1.3412321518119104, 61], [7, 1.4441845547575125, 207],
                              [7, 2.3299863215008583, 579], [7, 3.8160043167838316, 346], [7, 4.607466405259504, 470],
                              [7, 5.748780354734596, 718], [7, 7.7004741823716385, 773], [7, 7.942010953644975, 625],
                              [7, 8.262720134559261, 450], [7, 9.398350346850625, 875], [7, 9.660424639379832, 278],
                              [7, 10.62038131352441, 491], [7, 11.123614906592781, 430], [7, 11.311158451061226, 883],
                              [7, 12.678294855997777, 828], [7, 18.402118418468433, 724], [7, 18.572726314750405, 125],
                              [7, 18.780814971072914, 803], [7, 20.07818154795795, 851], [7, 20.225162507205084, 115],
                              [7, 20.26019742359961, 711], [7, 20.899744821539578, 645], [7, 22.135015038450533, 37],
                              [7, 23.39963909948828, 675], [7, 23.7041477793148, 852], [8, 0.5518941116888147, 201],
                              [8, 3.011845466184543, 541], [8, 5.177949511503593, 70], [8, 7.619708028672937, 209],
                              [8, 8.663709350554614, 807], [8, 8.820873079299792, 872], [8, 11.648876466578297, 782],
                              [8, 14.554573502445002, 884], [8, 18.311620186234645, 697], [8, 18.85532600898005, 251],
                              [8, 20.16576102193194, 609], [8, 20.17789174653966, 849], [8, 20.827331555978173, 376],
                              [8, 23.742430168145322, 68], [8, 24.838755727025447, 683], [8, 25.46846113811424, 443],
                              [8, 25.736620206002318, 586], [9, 1.4436290133099838, 298], [9, 1.774426283583906, 542],
                              [9, 2.149191596275398, 123], [9, 2.6138660428293767, 193], [9, 2.763600212982949, 383],
                              [9, 3.963889176364802, 802], [9, 4.03997069071383, 484], [9, 4.522691982615911, 423],
                              [9, 4.834134584980252, 82], [9, 5.779979484710894, 102], [9, 5.886483984986953, 12],
                              [9, 6.98858316689825, 619], [9, 7.592450847198266, 854], [9, 11.107103929151963, 197],
                              [9, 13.803353090815467, 709], [9, 16.317024035906154, 850], [9, 18.884786250120193, 787],
                              [9, 20.252844685692725, 736], [9, 23.15746694372435, 551], [9, 24.282417585522218, 659],
                              [9, 25.253817907877615, 417], [9, 27.249850897810088, 835], [9, 33.05853829679443, 323],
                              [9, 34.04567673021572, 427], [10, 0.31857581179112815, 224],
                              [10, 0.3819050598334596, 891], [10, 0.8824182781742184, 577],
                              [10, 1.399635350774825, 268], [10, 3.6933579561861007, 560], [10, 4.12324171064642, 127],
                              [10, 4.256861662324506, 292], [10, 4.304073893863734, 595], [10, 4.960591510423503, 737],
                              [10, 5.464930680429164, 414], [10, 5.673923969872031, 811], [10, 5.807790355738313, 466],
                              [10, 6.620273028508626, 370], [10, 6.885040889735152, 460], [10, 6.96338617976114, 23],
                              [10, 9.42214564284147, 343], [10, 9.435418344486818, 527], [10, 10.106357084539674, 514],
                              [10, 10.225643324136596, 373], [10, 14.319181865548815, 523],
                              [10, 17.45962629125738, 508], [10, 17.520252128495233, 293], [10, 20.28169003404062, 810],
                              [10, 21.699936428878324, 170], [10, 23.21016606140897, 620], [10, 24.36614119913473, 779],
                              [10, 24.70782018365848, 203], [10, 25.034505057393655, 552],
                              [10, 25.600523215130906, 219], [10, 29.683084170692865, 630],
                              [10, 31.73179418653089, 162], [10, 33.93848578722408, 694], [11, 0.7165744398816495, 420],
                              [11, 1.2407497389302962, 4], [11, 1.5503851280045806, 181], [11, 1.8019316409629542, 202],
                              [11, 1.9822279082244294, 722], [11, 2.288739338395981, 575],
                              [11, 2.6055085640249103, 489], [11, 3.7848965851997245, 559],
                              [11, 4.2014772819343635, 639], [11, 4.603465476435041, 561], [11, 6.487887056579753, 765],
                              [11, 9.06237166928021, 176], [11, 9.76963438332021, 22], [11, 9.916973806434358, 253],
                              [11, 13.000322924412309, 868], [11, 13.6548092030708, 555], [11, 20.25239114602367, 503],
                              [11, 22.66838801516276, 557], [11, 24.131265937806923, 297],
                              [11, 26.986449518633506, 482], [11, 27.241176188592952, 643],
                              [11, 28.45897361061762, 136], [11, 30.406669623331496, 770],
                              [12, 2.4421445122813172, 658], [12, 3.0653413806047056, 84], [12, 3.0813710745497986, 72],
                              [12, 4.028864400382857, 250], [12, 4.6054908322071855, 168], [12, 4.629895536793815, 148],
                              [12, 6.1671975874294205, 411], [12, 7.348558848291515, 2], [12, 7.992570292186764, 108],
                              [12, 8.16342116629156, 350], [12, 9.64180751743681, 133], [12, 10.935434831524192, 87],
                              [12, 10.984185954030702, 840], [12, 11.11685038952682, 97], [12, 13.513734967893146, 506],
                              [12, 14.293360709765027, 584], [12, 14.339088996737614, 492],
                              [12, 15.328033888820096, 501], [12, 16.00730770187925, 456], [12, 16.28329607360035, 129],
                              [12, 16.49723539223175, 332], [12, 18.135205292207427, 603], [12, 18.91750401663683, 130],
                              [12, 22.197122725398916, 769], [12, 23.121196236292114, 733],
                              [12, 23.152714861075985, 75], [12, 23.38895950750323, 553], [12, 29.885815385032224, 714],
                              [12, 34.89465062081943, 626], [13, 2.0687153635579607, 200], [13, 2.083465714873809, 255],
                              [13, 2.566084301610784, 837], [13, 3.256512172815148, 562], [13, 8.116375419473727, 118],
                              [13, 8.785018681153622, 861], [13, 10.089443084013608, 637],
                              [13, 10.213216977451232, 633], [13, 11.202008978039382, 631],
                              [13, 11.819980458041888, 784], [13, 14.340778950920718, 371],
                              [13, 17.824782563650526, 833], [13, 19.862733071988124, 888],
                              [13, 20.609217758738495, 653], [13, 21.42338871945428, 434], [13, 21.504697859013692, 86],
                              [13, 22.098496260706913, 429], [13, 22.272843540809802, 667],
                              [13, 22.425740370713974, 785], [13, 22.832324560467768, 7], [13, 23.367401045255065, 141],
                              [13, 24.600812063923357, 388], [13, 26.17687871477733, 183],
                              [14, 0.38917999345514026, 791], [14, 1.796033030088167, 260],
                              [14, 1.9481038095367351, 58], [14, 2.7056176596754686, 519], [14, 3.368892619424043, 288],
                              [14, 3.4289180764263736, 801], [14, 3.643926402551208, 442], [14, 5.589262356246572, 766],
                              [14, 6.663418481040331, 407], [14, 6.747854183457143, 150], [14, 8.346982520027181, 693],
                              [14, 8.728977790422789, 243], [14, 10.871578121813084, 335], [14, 10.95236636916003, 365],
                              [14, 11.1058674066484, 36], [14, 11.968678655547661, 229], [14, 12.138028113586891, 258],
                              [14, 12.96663759158966, 776], [14, 12.966846493567349, 750],
                              [14, 12.989080599400978, 688], [14, 16.758918393168635, 156],
                              [14, 17.53607355977392, 216], [14, 17.710514629179897, 815], [14, 18.85439557065636, 587],
                              [14, 19.46164670029829, 544], [14, 21.759922628678687, 459],
                              [14, 21.914026857305736, 314], [14, 24.55598058106954, 299],
                              [15, 0.27581445214785516, 727], [15, 0.8172888854147714, 20],
                              [15, 1.227566948018818, 613], [15, 3.28098200839631, 496], [15, 4.5506272994731365, 241],
                              [15, 6.096625991604267, 231], [15, 6.669215139202778, 18], [15, 9.775258745604976, 463],
                              [15, 12.70113586976094, 380], [15, 13.436074282031646, 730],
                              [15, 13.762827283856288, 267], [15, 13.9198867149964, 628], [15, 14.573977726744152, 763],
                              [15, 16.4855167573436, 228], [15, 19.277164301345625, 146], [15, 22.155824517744374, 10],
                              [15, 25.865604052433266, 495], [15, 26.277062714439197, 627],
                              [15, 28.38573741000108, 740], [15, 32.924801494223594, 248],
                              [16, 0.2717738810381847, 285], [16, 1.982468024193665, 147],
                              [16, 2.5124598011700767, 528], [16, 5.239517337752193, 563], [16, 6.200691645678309, 696],
                              [16, 7.09727807177579, 394], [16, 9.435863362929616, 173], [16, 11.59772189702177, 786],
                              [16, 12.858772997112332, 573], [16, 14.429668901020623, 479],
                              [16, 15.369105144101626, 164], [16, 16.005356890655722, 576],
                              [16, 16.142336866499928, 781], [16, 16.338045720432756, 657],
                              [16, 16.673585799726226, 441], [16, 16.90182677628166, 172],
                              [16, 17.003456801043537, 876], [16, 17.73403536016741, 825],
                              [16, 18.020226656151554, 873], [16, 18.365077181812776, 602], [16, 18.6814133397927, 721],
                              [16, 21.29946960072547, 651], [16, 21.310852335733724, 536], [16, 22.2707883002061, 809],
                              [16, 24.27119547497984, 47], [16, 24.351513014570518, 546], [16, 26.927726189330702, 304],
                              [16, 29.733110215359588, 617], [16, 31.63801493734339, 98], [17, 1.9701423325701928, 331],
                              [17, 2.2325144462550512, 751], [17, 4.097531358992328, 596], [17, 4.489651999876228, 144],
                              [17, 6.1443018312119095, 568], [17, 7.485118633340577, 310],
                              [17, 10.138902539570484, 823], [17, 10.622376191930522, 512],
                              [17, 12.815805964743205, 756], [17, 12.937217364290127, 154],
                              [17, 13.363915537450866, 521], [17, 17.85253109168902, 707], [17, 19.0786233279442, 820],
                              [17, 19.195466261986724, 236], [17, 22.890130343870222, 682],
                              [17, 23.436787614255042, 681], [17, 29.678936848447925, 374],
                              [17, 29.82263442458129, 132], [18, 0.7361621258043375, 283], [18, 1.1205280517787, 814],
                              [18, 1.5604368907894486, 471], [18, 1.7843798094025216, 642],
                              [18, 5.519706788683494, 344], [18, 10.142484238864114, 175],
                              [18, 10.412246326164407, 106], [18, 12.246957746357783, 817], [18, 14.4057564058272, 808],
                              [18, 14.547371525093562, 92], [18, 15.921239924205445, 351], [18, 16.90799732263043, 731],
                              [18, 17.977893266796734, 863], [18, 23.938768837774134, 246],
                              [18, 27.181124265900078, 841], [18, 27.608366449446486, 805],
                              [18, 27.995570948866053, 62], [18, 28.304993639587003, 877],
                              [18, 29.328031373188523, 677], [18, 29.896540098385866, 634],
                              [19, 0.3187883334525373, 316], [19, 1.0994227068923326, 348],
                              [19, 1.9682064773113674, 535], [19, 3.617796919757494, 372], [19, 4.916703274101941, 395],
                              [19, 4.95803040929799, 478], [19, 5.85105545274326, 46], [19, 9.038686930808794, 465],
                              [19, 12.745178633545853, 165], [19, 13.398154722042563, 151],
                              [19, 13.942574217186671, 290], [19, 13.944394885324249, 195],
                              [19, 14.991090914120669, 226], [19, 15.31151656870144, 392], [19, 17.830553313360276, 8],
                              [19, 20.425866643703895, 77], [19, 20.57569706618367, 30], [19, 23.772992612496793, 438],
                              [19, 24.578345281734688, 73], [19, 28.038048974010835, 308],
                              [19, 29.174802416183883, 303], [20, 0.904009406955425, 591],
                              [20, 1.4801649692077787, 510], [20, 1.880592951861829, 897], [20, 2.739516608902084, 530],
                              [20, 3.055385730504768, 743], [20, 3.907893424148533, 5], [20, 4.28643881118596, 564],
                              [20, 6.329212593044785, 116], [20, 7.123479946047918, 266], [20, 7.284633680402385, 886],
                              [20, 7.627626054365918, 208], [20, 9.015017153145987, 806], [20, 9.690283708395265, 233],
                              [20, 9.822977806295475, 662], [20, 10.559057843771365, 864],
                              [20, 10.726844385116735, 594], [20, 11.669634835425073, 798],
                              [20, 12.350155236724197, 755], [20, 12.657442331037817, 31],
                              [20, 13.674055873084194, 719], [20, 14.268000007925314, 404],
                              [20, 16.578296777886457, 101], [20, 17.248225057804433, 11], [20, 18.64760413232147, 425],
                              [20, 20.649842492751265, 597], [20, 21.473021626618547, 45],
                              [20, 21.700255046323907, 409], [20, 24.4834380539374, 717], [20, 25.27333033140807, 486],
                              [20, 25.452752049615608, 307], [20, 26.171698303525705, 16],
                              [20, 26.324514011903904, 794], [20, 27.16085896656578, 42], [20, 28.57913397571482, 54],
                              [20, 28.700151746012754, 629], [20, 29.30172599048404, 504], [20, 29.83810018452004, 355],
                              [20, 31.831108243730515, 63], [20, 33.37523328815844, 74], [20, 34.74291159793168, 640],
                              [21, 0.44557301273643607, 361], [21, 1.4745865203307778, 472],
                              [21, 2.134951250639949, 206], [21, 2.253416404313774, 274], [21, 3.315552619015818, 366],
                              [21, 3.498888671096807, 520], [21, 5.258782325422073, 50], [21, 5.472770601245345, 186],
                              [21, 7.545868133428161, 279], [21, 7.970309189338636, 505], [21, 9.106133869935888, 830],
                              [21, 9.469872726100096, 121], [21, 11.804390481074227, 396], [21, 11.90655786459016, 676],
                              [21, 12.625800534119703, 792], [21, 13.11088686946614, 689],
                              [21, 13.247576260904939, 120], [21, 13.405257682609257, 749], [21, 13.5891797024747, 110],
                              [21, 15.433263351839372, 797], [21, 15.679027851397123, 865],
                              [21, 15.859721643314309, 302], [21, 16.689120673993095, 44], [21, 18.516693389254005, 29],
                              [21, 18.619151669243674, 590], [21, 21.644752069673213, 890],
                              [21, 23.656304259890494, 257], [21, 24.202143164458505, 252],
                              [21, 24.20694997874346, 139], [21, 25.710109419486834, 114],
                              [21, 26.386970862055488, 273], [21, 32.648496521105606, 280],
                              [21, 32.85649376783633, 270], [21, 34.30193523016633, 793], [22, 2.026744295710672, 616],
                              [22, 2.1695969367497354, 321], [22, 2.385995775642585, 15], [22, 2.435541585580408, 669],
                              [22, 2.7500315200313525, 381], [22, 3.737179364223732, 401], [22, 5.301357277034275, 218],
                              [22, 5.631067143595251, 461], [22, 6.86468743207327, 543], [22, 8.750001895155735, 799],
                              [22, 8.989155607553789, 261], [22, 9.636461057719726, 408], [22, 10.047818121240283, 839],
                              [22, 10.169401360810994, 759], [22, 11.502486911612818, 671],
                              [22, 11.590433671131581, 353], [22, 12.487620701440571, 881],
                              [22, 12.611516171707088, 214], [22, 12.746831687497, 660], [22, 14.535854571865414, 14],
                              [22, 14.56697111800196, 804], [22, 15.237555315819163, 600], [22, 15.36321574500699, 431],
                              [22, 15.97900015236291, 516], [22, 16.726956436396854, 477],
                              [22, 18.014588599002668, 741], [22, 19.602356010617882, 610],
                              [22, 21.263883510301095, 325], [22, 22.155408324608327, 467],
                              [22, 23.063296995462615, 234], [22, 24.975627827507655, 17],
                              [22, 27.209410347965658, 650], [22, 28.75470069029521, 522], [23, 0.7809933699085436, 21],
                              [23, 1.6543449540400208, 500], [23, 1.6557428088681954, 352],
                              [23, 1.7414369621993866, 131], [23, 7.988585386870857, 549], [23, 8.174714833718777, 758],
                              [23, 12.051043484905772, 578], [23, 13.401764480514926, 393],
                              [23, 13.449173707976378, 34], [23, 13.492897900988718, 829],
                              [23, 15.190656669394842, 715], [23, 15.298909363946303, 78],
                              [23, 16.848404054012796, 760], [23, 17.35989675546187, 764],
                              [23, 17.579332189327214, 834], [23, 19.43862283856749, 357],
                              [23, 19.605570807348048, 898], [23, 19.892158169219915, 360],
                              [23, 21.979328349346847, 663], [23, 24.1786327553095, 363], [23, 25.69729399473961, 567],
                              [23, 25.845200520862555, 259], [24, 1.0327160081516071, 347],
                              [24, 2.026633239585445, 832], [24, 2.426398713604111, 79], [24, 3.385102264546905, 382],
                              [24, 6.2455613976467985, 81], [24, 11.043890804933435, 772],
                              [24, 11.162386165053945, 862], [24, 12.266651217839517, 732],
                              [24, 12.579955062492152, 612], [24, 13.227175913128905, 895],
                              [24, 13.417268231118893, 534], [24, 16.057153885490063, 247],
                              [24, 17.192269591755565, 744], [24, 18.331601842980227, 184],
                              [24, 23.238716216276114, 753], [24, 24.526670851667568, 413],
                              [24, 25.222906069470916, 672], [24, 25.227412396268488, 52], [24, 25.99515171232567, 190],
                              [25, 0.5298746188689057, 775], [25, 0.8416922177404513, 656],
                              [25, 1.6074738993043431, 889], [25, 2.1150717697553594, 556],
                              [25, 4.094257357440219, 451], [25, 4.679772882593173, 359], [25, 5.733068737200606, 699],
                              [25, 6.329516742194804, 217], [25, 6.455295932378784, 598], [25, 6.830097052757497, 403],
                              [25, 8.655755091233985, 410], [25, 8.68199194780101, 710], [25, 8.741699449342384, 509],
                              [25, 9.724756004756246, 831], [25, 10.112966734125493, 469],
                              [25, 10.456895225034117, 706], [25, 11.14510413014582, 580],
                              [25, 12.347760045990206, 661], [25, 13.838812059152056, 287], [25, 25.5449172452882, 356],
                              [25, 27.447766841892015, 143], [25, 29.99842281892296, 305],
                              [25, 37.366453828583445, 126], [26, 2.0647977005410545, 339],
                              [26, 2.527247305878453, 592], [26, 2.9802337792239433, 705], [26, 3.225494246397286, 240],
                              [26, 3.4355641276377913, 570], [26, 4.192570230631793, 48], [26, 4.799770035831143, 826],
                              [26, 5.3179460276953, 843], [26, 6.179371325697038, 94], [26, 10.596890743073963, 511],
                              [26, 14.47244580760179, 589], [26, 15.227166870288423, 540],
                              [26, 15.967435660960644, 789], [26, 18.718802525161937, 853],
                              [26, 18.985814991313134, 28], [26, 19.473202274313298, 550], [26, 21.02292871334653, 24],
                              [26, 21.144109202099084, 761], [26, 22.602106772868233, 537],
                              [26, 24.14587518999626, 424], [26, 24.16337885688468, 137], [26, 24.827185643661714, 762],
                              [26, 25.34013349791225, 33], [26, 26.866847497199277, 655], [27, 0.9708168236635908, 400],
                              [27, 1.461755648535874, 64], [27, 1.7448020978417418, 1], [27, 2.0075113435384973, 95],
                              [27, 4.774571190358121, 670], [27, 5.466815838230392, 754], [27, 7.841985393499345, 845],
                              [27, 7.979253968177206, 894], [27, 9.034151659412363, 187], [27, 10.303788183136113, 191],
                              [27, 13.568273158378643, 199], [27, 16.331873154913055, 93],
                              [27, 16.356212165611137, 548], [27, 17.06388519341438, 53], [27, 17.55155261203497, 405],
                              [27, 18.070789578693436, 526], [27, 19.31666087248074, 448],
                              [27, 20.206767590853264, 295], [27, 20.25767667071941, 182],
                              [27, 20.639224414597127, 329], [27, 20.96512468018471, 457], [27, 21.10770881485843, 739],
                              [27, 21.373305039680528, 328], [27, 21.525248971076383, 362],
                              [27, 24.156720963925263, 419], [27, 31.067650825943662, 109],
                              [28, 0.3934834116290016, 822], [28, 1.4694802926754502, 647],
                              [28, 1.6051840041610854, 91], [28, 2.403220427018134, 842], [28, 2.8789745860407265, 69],
                              [28, 3.15673015671437, 210], [28, 6.1478804605831545, 856], [28, 6.2742405071199565, 349],
                              [28, 6.8527400814926525, 838], [28, 9.026277614522938, 632], [28, 11.075626298022456, 9],
                              [28, 11.876788781478517, 674], [28, 12.357377316238942, 103],
                              [28, 12.420823232193287, 572], [28, 17.60244858648088, 220],
                              [28, 18.045116284630957, 836], [28, 18.694320217202026, 446],
                              [28, 18.957579617524697, 249], [28, 19.0875081027705, 416], [28, 19.635099678671764, 725],
                              [28, 19.709729324501865, 819], [28, 20.108781671215517, 60],
                              [28, 23.102104674773447, 847], [28, 23.980024848213038, 780],
                              [28, 24.583835156552993, 51], [29, 0.9891666652427508, 159], [29, 1.195381764738635, 882],
                              [29, 2.4230256697770423, 517], [29, 3.411909781849632, 745],
                              [29, 3.6609925717286727, 848], [29, 6.63654665506651, 565], [29, 6.858551857264927, 32],
                              [29, 8.980997877872815, 790], [29, 9.681461097490093, 49], [29, 10.40315536590313, 76],
                              [29, 10.646576262906207, 712], [29, 11.164914805614119, 152],
                              [29, 11.40849930992218, 788], [29, 11.723391297948485, 728],
                              [29, 13.637637159154549, 476], [29, 15.080625117449223, 428],
                              [29, 15.720431515187789, 317], [29, 17.430507584576226, 157],
                              [29, 18.09024564941643, 145], [29, 18.852027196467063, 245],
                              [29, 19.193337544849395, 265], [29, 21.32575923887616, 481],
                              [29, 21.877802004146954, 752], [29, 22.27029383497033, 499],
                              [29, 22.538530179258675, 601], [29, 22.69181632592546, 13], [29, 23.069634389958974, 322],
                              [29, 24.455806558348357, 673], [29, 24.931095338996776, 319],
                              [30, 0.17659125796474862, 330], [30, 2.0492570526972202, 169],
                              [30, 2.059181872164599, 455], [30, 2.2306344362862203, 19], [30, 4.0831112414736825, 57],
                              [30, 4.989485580998836, 742], [30, 5.013695968796833, 474], [30, 5.793417841430602, 105],
                              [30, 8.962693325908642, 585], [30, 8.993521479975106, 605], [30, 9.576482226699373, 204],
                              [30, 10.679000270206059, 480], [30, 13.368005709726619, 449],
                              [30, 13.790476506362978, 432], [30, 16.567742612984173, 263],
                              [30, 16.854845372758273, 494], [30, 17.4451369023158, 180], [30, 18.97185361458089, 65],
                              [30, 21.326677021206116, 67], [30, 22.518051747526357, 493],
                              [30, 23.348853338426004, 678], [30, 24.14726075144374, 795],
                              [30, 30.768976196890318, 649], [31, 0.7304340104283741, 720],
                              [31, 1.226703134320665, 421], [31, 1.9591936434893877, 107],
                              [31, 1.9651727137194823, 174], [31, 3.390072031194399, 256],
                              [31, 3.5029110250482582, 668], [31, 6.022540751929228, 167], [31, 7.705340591385496, 436],
                              [31, 7.775354861453729, 824], [31, 8.536717159217726, 122], [31, 9.015645897547106, 0],
                              [31, 10.823458468035092, 135], [31, 11.443403332593164, 812],
                              [31, 11.822223240586709, 704], [31, 12.159905416844618, 368],
                              [31, 12.61949252052002, 507], [31, 12.940321425733634, 223],
                              [31, 13.873025105255405, 398], [31, 15.08418939757373, 100], [31, 15.37895645432692, 340],
                              [31, 18.201092539219125, 558], [31, 18.722366455598028, 312],
                              [31, 18.879964899959173, 498], [31, 19.3825665464977, 25], [31, 20.035691300403148, 244],
                              [31, 20.69408886057042, 635], [31, 20.92887429120686, 818], [31, 21.325748435168936, 291],
                              [31, 21.97639410385256, 538], [31, 23.625031261452698, 128], [31, 26.631217755608237, 56],
                              [32, 1.031553002127626, 569], [32, 2.780956825377602, 171], [32, 3.4547334082064824, 112],
                              [32, 5.088425730322433, 397], [32, 5.441956089399635, 161], [32, 5.556313960014237, 227],
                              [32, 10.14802549001151, 336], [32, 12.74168795222386, 615], [32, 14.498918897621484, 399],
                              [32, 14.58020181545153, 88], [32, 18.95986767756369, 213], [32, 22.279339544017077, 149],
                              [32, 24.370830626292566, 402], [32, 25.107883782694998, 757],
                              [32, 25.318626628803305, 59], [32, 25.776997712728217, 547],
                              [32, 26.742558016841183, 309], [32, 27.25429543460832, 796], [32, 27.305183024376213, 90],
                              [32, 32.880187653533696, 778], [33, 1.5920635439569022, 893],
                              [33, 1.8339977689323184, 604], [33, 2.561367831784845, 816], [33, 2.776522043980697, 444],
                              [33, 2.8028208320559957, 282], [33, 3.076229127515621, 892], [33, 5.795517977824036, 748],
                              [33, 7.7531399568683295, 262], [33, 7.883868018403869, 606], [33, 10.3960002737905, 271],
                              [33, 10.646355025210706, 320], [33, 11.177373073267846, 364],
                              [33, 12.047147242334205, 104], [33, 14.872888209074665, 311],
                              [33, 16.23853705640845, 412], [33, 16.55282546638311, 377], [33, 16.815436692209193, 301],
                              [33, 18.8832396089005, 418], [33, 20.21376538787716, 747], [33, 20.65393680413508, 3],
                              [33, 20.94083255363692, 624], [33, 21.24401294481816, 566], [33, 22.724354803999162, 475],
                              [33, 23.37187599643037, 813], [33, 24.817828392931528, 726],
                              [33, 24.863171747437285, 545], [33, 25.777211626348375, 638],
                              [33, 26.260080750885948, 391], [33, 27.302498940049894, 623],
                              [33, 28.03272477394249, 857]]
        qt_t = 20.917899054286224
        st_t = 2.6259086087332046
        student_experience = [[25, 14.69600375239442, 164], [26, 20.086467669956555, 84], [19, 5.54003512747389, 636],
                              [28, 22.701762449817274, 388], [5, 16.862157490172113, 157], [4, 5.678841375526133, 188],
                              [32, 5.088425730322433, 397], [20, 28.57913397571482, 54], [9, 10.410118959896359, 527],
                              [0, 9.19165921637683, 508], [29, 9.9678966332382, 251], [7, 2.3299863215008583, 579],
                              [22, 4.223988714234906, 50], [10, 14.385378755797866, 2], [0, 4.505230937392426, 255],
                              [2, 15.783579820229843, 719], [25, 8.601843432121868, 671], [13, 18.758948696897733, 158],
                              [23, 6.25375953285234, 384], [9, 2.1602610555771693, 577], [14, 8.451735341486538, 475],
                              [31, 6.022540751929228, 167], [1, 7.6761489680924475, 181], [26, 4.670416397876399, 86],
                              [24, 15.105649204508515, 651], [6, 1.5192225824913435, 369], [7, 6.560326898574845, 210],
                              [4, 8.328503735952088, 313], [18, 20.31351963841494, 769], [1, 16.245970464229593, 465],
                              [27, 20.86031943730355, 289], [12, 12.256030674842055, 123],
                              [29, 16.346032122656798, 494], [5, 9.258591824905302, 182], [5, 5.398864146542276, 750],
                              [6, 6.836064923811851, 487], [6, 16.109017862925366, 548], [17, 4.786797564637181, 306],
                              [10, 15.232088264552852, 814], [19, 11.47457259926635, 279], [5, 16.41102412732672, 312],
                              [3, 29.657044781623593, 459], [0, 2.1870420975106026, 658], [5, 12.079338531238879, 257],
                              [11, 8.146300673838697, 117], [29, 11.809076829061823, 195],
                              [33, 1.9590335036077309, 347], [32, 24.75645160589469, 545], [30, 3.67554210704604, 864],
                              [22, 11.590433671131581, 353], [6, 15.302113486995522, 357],
                              [22, 5.0190941066503365, 787], [27, 16.331873154913055, 93],
                              [17, 3.0917161926218903, 500], [16, 6.9862633298117505, 372], [17, 13.671011244734974, 6],
                              [30, 21.56338595657578, 672], [15, 5.10496093345982, 480], [1, 3.35090999184509, 124],
                              [32, 8.378825613354183, 889], [19, 21.680840521700976, 668], [1, 8.779087462258087, 91],
                              [23, 15.446175721554443, 704], [20, 18.817085315559844, 30], [1, 7.355475374264998, 148],
                              [27, 21.259087987749187, 280], [4, 12.486890317846251, 448], [14, 9.403001964718458, 790],
                              [12, 16.28329607360035, 129], [19, 22.697489766434373, 835], [20, 6.67529638313154, 748],
                              [12, 10.729277349477192, 730], [10, 5.464930680429164, 414], [2, 1.2958191906232486, 316],
                              [6, 19.71248952483305, 601], [20, 1.880592951861829, 897], [19, 14.796518600304601, 402],
                              [21, 19.809115828102307, 272], [23, 14.983181916394884, 232],
                              [25, 6.830097052757497, 403], [28, 19.714686938215912, 627], [4, 5.133805559818504, 333],
                              [12, 6.7512028938805955, 310], [9, 12.851568066228523, 478], [4, 12.98185299168751, 340],
                              [0, 4.186623907973076, 797], [33, 6.083182561297734, 227], [31, 17.89754577017655, 749],
                              [12, 12.002857420964979, 381], [5, 5.177510587130849, 490], [10, 32.83695354082263, 29],
                              [29, 12.59042165061187, 283], [18, 21.553170055866048, 409], [8, 19.896999730404556, 109],
                              [16, 22.816215658994043, 360], [4, 19.867757596316142, 406], [29, 18.22981860110308, 44],
                              [8, 23.069785442891074, 566], [6, 13.455726302313396, 597], [4, 19.016250520366214, 587],
                              [16, 2.920280871060908, 642], [4, 15.112387734013554, 140], [5, 19.73007820432963, 552],
                              [9, 9.36289334633687, 31], [4, 13.517599418834665, 635], [20, 20.663174243651667, 620],
                              [4, 5.050746188969361, 608], [3, 2.0542404954907627, 517], [28, 15.635322098368055, 629],
                              [32, 31.684013195098757, 852], [10, 4.104920099313529, 4], [33, 7.884371702714356, 826],
                              [3, 6.586654835351585, 647], [13, 17.510085743503, 836], [19, 4.165513209655268, 596],
                              [19, 26.195571480646105, 305], [21, 27.59692236741894, 249], [4, 19.546653851983994, 92],
                              [10, 11.547343908559316, 614], [23, 10.705864084281837, 349],
                              [24, 13.133535504628565, 675], [13, 3.213846651194195, 213], [4, 3.7522247103876576, 80],
                              [22, 5.944607576303524, 271], [6, 26.73158149645134, 481], [16, 20.761819619833794, 855],
                              [4, 7.005304850767645, 205], [24, 2.537719163837677, 401], [27, 13.93998189059194, 291],
                              [23, 5.303049222378347, 348], [1, 7.210796012378532, 572], [30, 18.411638568001383, 612],
                              [33, 21.265481802732396, 827], [31, 19.392514466000357, 126],
                              [0, 11.261911815214939, 616], [19, 13.942574217186671, 290],
                              [15, 14.542191542893695, 850], [10, 3.8774337722794483, 722],
                              [24, 2.615454850977071, 893], [3, 12.215295990138566, 520], [25, 7.09789838074715, 829],
                              [13, 29.099327235687333, 649], [20, 21.287829011577053, 879],
                              [24, 3.1421127616075997, 662], [0, 9.764465250967381, 707], [2, 7.774143670535715, 823],
                              [15, 23.463177284017565, 546], [4, 12.203625033596827, 729], [5, 7.3014840694749665, 700],
                              [31, 19.12401817829996, 236], [16, 10.228462902332868, 445], [1, 8.592661202251842, 817],
                              [7, 18.60485389012803, 816], [24, 10.65615268551893, 528], [1, 29.289665260958618, 877],
                              [29, 12.645787301451396, 375], [5, 2.66300192679401, 874], [4, 6.215512301888602, 284],
                              [31, 23.32688475486505, 153], [5, 23.38489411085679, 728], [12, 19.56715030752305, 10],
                              [31, 12.438680953143207, 38], [8, 20.17789174653966, 849], [19, 21.019998190124273, 222],
                              [13, 25.752311758079045, 139], [30, 6.710144165020778, 851], [8, 8.663709350554614, 807],
                              [9, 25.249301934819172, 680], [15, 18.44362087783747, 417],
                              [30, 0.17659125796474862, 330], [14, 24.348910213562746, 443],
                              [17, 14.056891250521126, 101], [6, 8.09552919336303, 549], [11, 6.836101724417922, 383],
                              [1, 6.000340387788617, 425], [10, 23.872174095083988, 757], [17, 12.937217364290127, 154],
                              [25, 18.575321271786873, 408], [5, 10.859438313646665, 335], [20, 7.627626054365918, 208],
                              [20, 5.96227255171479, 214], [2, 26.180967343005264, 533], [25, 10.081075010226533, 798],
                              [24, 16.48093720841181, 881], [10, 23.369611569354657, 512],
                              [32, 26.689981791478232, 462], [14, 12.01861741408477, 363], [17, 9.780119123002969, 175],
                              [30, 10.588631799509447, 713], [29, 12.31573465770413, 438],
                              [16, 15.949625690072853, 725], [31, 21.306146814689328, 780], [4, 13.12733235624102, 320],
                              [1, 5.274784668537927, 225], [5, 5.913228787905685, 132], [17, 18.331781303325634, 8],
                              [3, 12.424601036503619, 486], [3, 1.9579237239349656, 455], [16, 15.702937074122742, 121],
                              [19, 7.177097386985964, 806], [27, 2.0075113435384973, 95], [21, 3.315552619015818, 366],
                              [3, 10.918373396192784, 334], [2, 10.62550090921586, 515], [30, 18.297162533741126, 697],
                              [12, 18.91750401663683, 130], [5, 1.9423447752412228, 156], [0, 6.646574388866718, 420],
                              [27, 13.37370804189306, 18], [4, 10.794122179054488, 690], [8, 2.706565653106049, 565],
                              [27, 13.001112460373351, 65], [13, 14.763351626801368, 796],
                              [14, 12.138028113586891, 258], [32, 8.585441402532275, 606], [21, 7.169545380526764, 775],
                              [27, 13.370242853494611, 820], [21, 18.47363532463789, 760],
                              [12, 12.795293117090717, 479], [2, 10.459337132141503, 756],
                              [23, 18.156107963734392, 177], [2, 13.499247217948254, 263], [28, 18.63251134791822, 495],
                              [3, 17.13578764766172, 5], [27, 17.625795387494342, 513], [28, 15.123930192617353, 332],
                              [5, 3.400875796200744, 529], [8, 25.78082519169049, 793], [23, 11.52483037236002, 803],
                              [18, 13.028443031971008, 598], [28, 8.403848052566529, 666],
                              [29, 19.193337544849395, 265], [31, 9.015645897547106, 0], [3, 29.135439915331947, 367],
                              [1, 12.554302584884923, 737], [10, 4.256861662324506, 292], [23, 4.483862614669463, 687],
                              [4, 5.352215050805391, 768], [11, 7.1755589320428665, 463], [18, 14.006828051498356, 111],
                              [18, 6.235931096313666, 862], [31, 3.682491591325501, 474], [29, 13.484167786801088, 503],
                              [1, 10.504586920884483, 858], [9, 4.522691982615911, 423], [4, 20.66483830250875, 235],
                              [33, 10.1070728589038, 108], [8, 25.287920640034415, 259], [4, 2.556112193013977, 39],
                              [11, 6.283875303601205, 113], [26, 5.586949890057854, 476], [18, 18.417639056773815, 492],
                              [4, 14.000051835268687, 389], [1, 17.9435447193429, 501], [25, 12.706229597042798, 727],
                              [7, 1.3412321518119104, 61], [7, 10.62038131352441, 491], [1, 10.100247269958986, 664],
                              [14, 12.836852375303483, 169], [10, 9.690073925443924, 673],
                              [25, 24.650482169520618, 183], [22, 7.24678555493941, 837], [26, 24.845628555470935, 694],
                              [31, 19.3825665464977, 25], [5, 8.353773385868172, 26], [25, 19.85513515834414, 392],
                              [31, 8.804927210900782, 764], [31, 5.125555239214928, 70], [14, 6.542732489044941, 539],
                              [6, 16.63184289237231, 788], [21, 6.30740209997019, 344], [3, 16.210462903251763, 299],
                              [13, 27.936992669100967, 714], [32, 10.784518469142476, 371],
                              [6, 26.059400723569393, 422], [1, 26.49472778462686, 327], [17, 19.34217493423038, 145],
                              [3, 1.679436761570663, 532], [13, 16.940338025934576, 11], [31, 14.203487182751463, 34],
                              [5, 6.79479980106914, 531], [31, 6.6363121662266344, 209], [2, 5.765686487295914, 695],
                              [6, 16.2982622581144, 710], [30, 19.404942442487183, 355], [8, 20.200014176667725, 51],
                              [31, 10.831622317281935, 152], [28, 25.100866535907663, 176],
                              [29, 3.411909781849632, 745], [9, 8.011959789926168, 738], [4, 4.164935991495509, 398],
                              [12, 16.15147676057913, 197], [18, 7.546047396550724, 116], [7, 8.262720134559261, 450],
                              [9, 5.886483984986953, 12], [22, 25.777031418376154, 190], [6, 23.331799023597295, 755],
                              [23, 4.689995339797285, 239], [6, 3.746548777710432, 260], [23, 3.895084031121679, 472],
                              [23, 0.7809933699085436, 21], [26, 24.16337885688468, 137], [27, 14.60467441100434, 866],
                              [12, 30.325315466308975, 245], [22, 20.11245678653905, 73], [29, 18.6516096070866, 688],
                              [7, 7.136268985408038, 633], [31, 12.61949252052002, 507], [5, 7.413378911878928, 415],
                              [1, 17.647750646509095, 638], [28, 16.213432641507644, 418], [8, 7.669899792510531, 444],
                              [29, 18.66273980941624, 138], [31, 14.171097806721324, 618], [5, 3.3148487415864905, 315],
                              [20, 13.907494260017259, 521], [27, 20.465427571761193, 499],
                              [11, 20.608305207837297, 582], [26, 6.444595550407951, 75], [23, 11.544719736908283, 339],
                              [11, 4.406024704030021, 652], [29, 6.341455073267685, 377], [30, 10.4633054607301, 511],
                              [33, 18.289698222311923, 322], [3, 7.699478942240233, 136], [27, 4.774571190358121, 670],
                              [23, 24.3464884839648, 47], [4, 9.356538616870317, 433], [19, 18.72865369214142, 554],
                              [7, 12.251917051565654, 896], [31, 4.787079905747379, 386], [33, 21.53796781765335, 682],
                              [33, 17.954207851828773, 744], [9, 7.805842165646144, 628], [15, 8.295282361691688, 624],
                              [0, 13.634470279622347, 58], [3, 6.919239712340408, 242], [33, 5.336031425256655, 161],
                              [32, 11.066811530293885, 426], [12, 27.712598157500906, 270],
                              [12, 8.608668608792309, 637], [24, 30.78953724782535, 7], [14, 2.455410337415641, 358],
                              [6, 16.48781955783652, 593], [13, 10.774292867780176, 773], [20, 9.537945073661323, 28],
                              [33, 19.036940543413408, 184], [25, 2.1150717697553594, 556],
                              [17, 3.667484805682826, 160], [21, 16.196863718537234, 678], [5, 5.479998145235349, 752],
                              [10, 3.914257295676875, 765], [1, 10.00573234409074, 641], [22, 11.256835168327004, 180],
                              [12, 7.343981923606121, 395], [0, 13.201362188299676, 834], [31, 8.06753385340213, 782],
                              [28, 3.2988942317078935, 676], [13, 15.85588779867687, 743], [24, 16.9657218097976, 792],
                              [25, 22.870220618212635, 424], [7, 24.291590297041438, 681], [4, 5.604176219084018, 464],
                              [4, 11.683093451948597, 860], [5, 2.6597586030068525, 644], [10, 15.865345749045424, 33],
                              [21, 5.322300350006995, 856], [10, 23.47330588588027, 288], [24, 12.8966653012797, 689],
                              [26, 11.386632526870796, 839], [32, 10.198888878783112, 368], [9, 5.779979484710894, 102],
                              [4, 15.678899167374144, 244], [31, 18.469903653634827, 870], [2, 15.975356131577074, 844],
                              [2, 7.959926042411558, 253], [26, 20.436656007186986, 396], [4, 12.60215475050443, 800],
                              [8, 33.10596748362097, 319], [31, 17.11422894998875, 702], [12, 5.427982803895146, 563],
                              [22, 6.670221219743672, 754], [31, 8.16901739465478, 336], [28, 20.94378366309398, 761],
                              [3, 4.338950928956332, 107], [23, 11.762327510494176, 709], [32, 9.133671275271968, 564],
                              [30, 8.92120335831509, 603], [32, 17.255718084810244, 880], [6, 18.099184552430632, 526],
                              [6, 20.500038801382182, 74], [5, 4.347493122075731, 703], [23, 24.5098843544053, 711],
                              [23, 12.500051360674291, 57], [16, 10.026297242170052, 105], [2, 34.21636794141784, 269],
                              [12, 11.013060415149093, 309], [4, 8.692076454938656, 212], [14, 5.683700075156771, 204],
                              [5, 23.00092212012877, 98], [8, 12.544018030860252, 589], [9, 6.157789573426107, 811],
                              [13, 9.45257917051111, 133], [21, 26.432332410696926, 504], [30, 17.14014040229045, 567],
                              [31, 10.823458468035092, 135], [13, 5.031176240606747, 72], [28, 21.406148901684187, 374],
                              [32, 13.412225855226989, 562], [15, 20.088979587742568, 705], [9, 4.805361377347117, 22],
                              [24, 11.748406847557456, 46], [11, 3.6209677725910687, 619],
                              [30, 19.448076455638464, 318], [4, 6.092742693337868, 665], [16, 10.588221429381566, 382],
                              [23, 1.6557428088681954, 352], [17, 25.242041370065216, 734], [4, 18.702367348904858, 96],
                              [26, 1.621435559826183, 482], [2, 15.673142713907865, 329], [9, 10.124141132479567, 547],
                              [26, 4.192570230631793, 48], [7, 13.7053433951331, 88], [3, 19.821736667329237, 538],
                              [8, 4.629837301293277, 557], [17, 15.004429481268865, 808], [23, 24.12394508007935, 114],
                              [28, 10.56287881588806, 186], [29, 19.27149061474587, 89], [14, 12.237865467539473, 337],
                              [28, 22.660151031567636, 391], [26, 8.853243749670165, 720],
                              [18, 21.682024783691745, 430], [29, 19.39782917151128, 170], [10, 6.92986394402357, 82],
                              [6, 22.148706334934882, 60], [33, 26.848163180024656, 869], [15, 23.887384111613912, 733],
                              [23, 8.034542750898558, 55], [29, 21.949169132704867, 456], [1, 7.527619951079904, 784],
                              [26, 17.582201779086365, 3], [31, 9.081759094634283, 894], [33, 7.7531399568683295, 262],
                              [2, 27.30579324420042, 859], [24, 15.756688003529499, 804], [20, 12.742796554108955, 701],
                              [26, 3.225494246397286, 240], [22, 21.043024350397747, 731], [4, 20.54370430760733, 467],
                              [4, 1.2359567107722258, 303], [3, 19.948772651089723, 62], [3, 3.9029306142091658, 69],
                              [26, 15.678658776482997, 473], [18, 16.3600432059975, 323], [25, 6.827839866065059, 178],
                              [5, 3.0953033486454533, 85], [28, 18.739324909962598, 281], [21, 14.784758413778802, 483],
                              [3, 5.501505418624675, 1], [31, 11.330945063725979, 741], [19, 9.48954734364232, 699],
                              [8, 11.838577438660561, 544], [10, 14.010729987456614, 118], [30, 21.326677021206116, 67],
                              [27, 17.469081301023206, 819], [19, 19.39621091431186, 630],
                              [18, 11.356861991233979, 505], [28, 14.65885853728707, 813], [7, 8.116374482182538, 559],
                              [20, 33.740112021809445, 211], [3, 12.67989330159025, 898], [21, 14.416688491569674, 509],
                              [4, 7.839528773280979, 468], [22, 8.672989577717008, 87], [14, 15.497540410977624, 405],
                              [0, 12.174640912947853, 298], [20, 17.33876496077469, 770], [19, 10.875933025817213, 364],
                              [20, 11.032427427139968, 643], [16, 18.28774287970837, 653], [0, 15.975603435747812, 16],
                              [7, 6.59586468269231, 168], [25, 12.326374462083617, 571], [30, 12.117790977308816, 477],
                              [20, 12.795410449923832, 692], [16, 17.73403536016741, 825],
                              [13, 19.796792720032737, 650], [14, 14.233205449720332, 385],
                              [14, 20.88512497710359, 810], [9, 20.252844685692725, 736], [10, 3.1880549381625953, 484],
                              [8, 6.9955395729810945, 421], [14, 19.349601540623976, 328],
                              [21, 11.486853816961128, 103], [18, 15.778571121453204, 52], [33, 3.076229127515621, 892],
                              [19, 9.300701403441014, 832], [30, 4.01740050078381, 882], [24, 13.475464483722448, 865],
                              [16, 24.3942129140952, 553], [26, 20.65839101576437, 805], [32, 3.4547334082064824, 112],
                              [15, 10.857911446744124, 854], [32, 4.804416073073767, 659], [0, 6.0898491983, 735],
                              [27, 9.680292333373501, 833], [16, 13.580930860394774, 569], [15, 15.74151628726016, 863],
                              [26, 3.4355641276377913, 570], [18, 20.495860278102878, 220],
                              [0, 11.518428770784066, 887], [16, 14.708689115739707, 502],
                              [15, 15.017181280230512, 679], [29, 24.58414941238153, 457],
                              [23, 2.2651514773002304, 669], [33, 20.647817949575103, 873],
                              [14, 3.4289180764263736, 801], [17, 35.67797677055685, 626], [28, 9.586121669781884, 343],
                              [12, 15.97903366621772, 576], [0, 29.985784143164494, 525], [0, 24.73227707877023, 413],
                              [22, 11.039263280180826, 592], [24, 16.13158501412615, 871], [19, 7.19245048851273, 661],
                              [7, 22.449564841230774, 772], [11, 9.183306206176882, 370], [6, 7.018772385689429, 674],
                              [28, 22.489962000965857, 431], [4, 5.260501297334461, 654], [15, 3.28098200839631, 496],
                              [25, 9.759125208150806, 809], [13, 1.5809473742083608, 221], [9, 4.978522984458418, 202],
                              [18, 1.2353930303808072, 331], [19, 8.317469411688336, 781], [5, 7.695751308829434, 581],
                              [16, 18.60122491088071, 146], [3, 5.4227936710228155, 848], [3, 13.035607974637928, 712],
                              [29, 15.720323199754013, 721], [8, 15.945522136648256, 751],
                              [24, 22.306135475422195, 677], [1, 10.758522466737892, 268], [4, 13.322539435157823, 254],
                              [11, 19.26212711145427, 141], [3, 16.45497682615771, 43], [32, 13.976629367484794, 134],
                              [32, 13.92874451728922, 196], [16, 22.17452824347746, 434], [8, 8.72405503144532, 104],
                              [6, 14.35219727594378, 841], [29, 13.64421666977618, 83], [27, 11.17046556998621, 884],
                              [2, 14.157005082911994, 165], [28, 24.898562691348825, 338],
                              [18, 13.134937181284556, 201], [16, 10.964214392164005, 68],
                              [14, 21.807184306057653, 812], [10, 3.2755707398217115, 575],
                              [14, 21.914026857305736, 314], [5, 5.176601329979234, 611], [25, 6.290937703446456, 125],
                              [22, 10.553011084443654, 174], [31, 25.674518460481597, 617],
                              [22, 14.79396502253051, 540], [5, 2.7017844363997585, 708], [24, 20.645703736110633, 432],
                              [21, 21.828959574126035, 293], [4, 10.339938057319499, 753], [2, 10.044953195296415, 228],
                              [12, 18.622234240872604, 610], [5, 20.922125200569134, 716],
                              [12, 22.822095709445033, 691], [17, 12.819079226071821, 523], [4, 1.74214844316713, 437],
                              [19, 10.380010351035738, 247], [5, 3.6203618341193926, 497], [8, 11.75681098260316, 106],
                              [13, 26.5329009225156, 356], [6, 23.40434791607464, 248], [7, 8.991839200090483, 143],
                              [14, 18.214711856742685, 295], [0, 20.97266244129565, 541], [14, 2.7056176596754686, 519],
                              [4, 3.4764517024142676, 583], [30, 22.016355797360777, 847],
                              [28, 17.138150977198322, 724], [9, 5.511803206263356, 127], [28, 27.403513520133398, 198],
                              [16, 19.562330064473706, 419], [19, 13.137865044192301, 609], [9, 3.963889176364802, 802],
                              [11, 8.60611778714189, 460], [20, 9.419177057417969, 410], [11, 13.019970253334789, 206],
                              [10, 6.304176398024983, 231], [22, 13.610244233273903, 142],
                              [17, 18.613276784852175, 774], [21, 14.134647903546783, 66],
                              [14, 17.710514629179897, 815], [21, 8.528126427153078, 867], [26, 5.007673975892933, 427],
                              [27, 13.532451813244036, 229], [2, 29.25216496236692, 307], [29, 21.82001408113594, 868],
                              [0, 30.44645747406437, 166], [20, 14.961845635896626, 766], [23, 14.831392699295703, 399],
                              [5, 2.8808120703999465, 771], [1, 8.340001653088956, 346], [14, 10.95236636916003, 365],
                              [29, 21.00772967261706, 739], [18, 32.00811171912047, 795], [32, 25.7644601376103, 317],
                              [24, 16.121866925475167, 120], [14, 12.96663759158966, 776], [30, 22.82407762940398, 17],
                              [30, 7.773305207627351, 274], [32, 2.780956825377602, 171], [4, 13.799884516528522, 199],
                              [3, 1.0839381785149347, 543], [11, 2.749941222086659, 224], [7, 9.660424639379832, 278],
                              [22, 5.598474434335951, 530], [24, 15.258058133812181, 412], [2, 12.673331754468022, 466],
                              [33, 26.980278498495316, 586], [31, 15.224126672504843, 594], [3, 9.918244356716807, 485],
                              [22, 28.75470069029521, 522], [24, 9.910976046250939, 631], [4, 23.52625891778167, 470],
                              [12, 13.811576204717998, 326], [13, 11.703495606955853, 49],
                              [13, 14.737943734811362, 623], [13, 23.767975959801312, 740],
                              [32, 8.211998263419742, 656], [4, 11.563714068111645, 324], [32, 3.873896772125033, 551],
                              [3, 8.001159495583996, 300], [23, 11.330146868039588, 622], [16, 0.2717738810381847, 285],
                              [22, 9.126899469445735, 173], [12, 10.49037972242391, 321], [2, 1.6087643777243839, 758],
                              [20, 24.107695524235144, 376], [6, 4.4276268737640665, 207], [4, 8.764919174081554, 100],
                              [19, 11.092639511956644, 301], [26, 9.46429171925418, 872], [7, 9.398350346850625, 875],
                              [12, 1.7238539079925541, 799], [22, 15.97900015236291, 516], [33, 7.160217784893816, 203],
                              [19, 9.132663694148814, 537], [16, 4.996440536960173, 588], [14, 5.477303394771647, 890],
                              [31, 16.133214927135196, 449], [9, 13.390004220363867, 261], [8, 18.766812318508613, 241],
                              [3, 16.270325975706502, 853], [2, 10.534524037573535, 783], [10, 0.3819050598334596, 891],
                              [26, 20.97358205560397, 302], [21, 23.37184872085507, 42], [9, 22.63655579421501, 845],
                              [31, 8.536717159217726, 122], [26, 8.130161035148287, 110], [6, 23.188315733058207, 172],
                              [25, 4.679772882593173, 359], [19, 13.360281149537292, 250], [4, 6.937211614055662, 767],
                              [5, 1.2291040033470175, 458], [30, 6.582494362122076, 286], [22, 5.083931243070187, 604],
                              [29, 16.65548905880972, 53], [5, 3.389713249504466, 435], [14, 3.643926402551208, 442],
                              [30, 10.837951604772028, 555], [1, 7.7033698533985175, 821],
                              [16, 25.917214792220996, 550], [4, 16.15744850618215, 818], [26, 32.57251408767877, 778],
                              [6, 21.69449690207629, 645], [4, 6.436420863232319, 99], [4, 6.287891190598109, 452],
                              [14, 16.012517938001853, 35], [17, 10.605458392358653, 615], [19, 6.693832904711367, 830],
                              [25, 9.724756004756246, 831], [1, 17.6274787378117, 599], [27, 10.656805452858087, 380],
                              [7, 27.19993755160861, 663], [13, 3.8274958536875756, 20], [11, 4.913924164647402, 192],
                              [4, 2.322076547654568, 488], [19, 9.111506826212267, 266], [11, 11.676147862502294, 390],
                              [19, 2.91703798913488, 144], [18, 11.247933971896833, 233], [4, 5.662558619767932, 223],
                              [28, 19.840089614584652, 246], [1, 33.44832668474526, 498], [30, 21.63845126438539, 683],
                              [30, 11.149900364665053, 439], [6, 20.98748251753597, 151], [29, 17.766603749001384, 294],
                              [5, 5.059248804480454, 746], [2, 31.74678255959445, 304], [15, 4.258040168215197, 660],
                              [31, 7.775354861453729, 824], [2, 10.661711525061486, 15], [16, 14.367699656910913, 393],
                              [33, 13.444181939218197, 895], [9, 33.69506621107405, 162], [25, 14.348318241448712, 81],
                              [19, 15.813961249144088, 625], [28, 20.566229746000317, 574], [26, 21.02292871334653, 24],
                              [9, 26.57281006216384, 838], [29, 7.560546724433182, 187], [30, 18.356079643285277, 45],
                              [19, 3.909608773583265, 453], [33, 16.950522160768855, 354], [4, 12.43119164298272, 605],
                              [19, 9.427895927137532, 387], [7, 4.344756311621524, 350], [8, 9.322095699472714, 534],
                              [7, 1.447629956021802, 36], [6, 6.576693848014175, 786], [27, 8.029469615649923, 685],
                              [13, 22.272843540809802, 667], [1, 4.388864798586297, 621], [28, 13.085796034505764, 128],
                              [12, 15.105460434419413, 234], [21, 5.249835451235063, 79], [28, 14.778692786055775, 763],
                              [3, 10.032914636883069, 506], [4, 1.7918948371775574, 194], [28, 28.751507964276072, 13],
                              [26, 24.575800514202967, 440], [21, 8.91832701739871, 584], [27, 10.377787348682956, 32],
                              [1, 7.201332691177858, 861], [14, 8.728977790422789, 243], [15, 11.150019104258075, 275],
                              [4, 2.988937365123516, 646], [12, 10.230281713982846, 351], [19, 4.027773529374801, 394],
                              [25, 3.605936714634759, 37], [21, 25.00871361783043, 308], [19, 4.825294672522959, 131],
                              [29, 11.4712208457283, 536], [22, 21.9662944444937, 717], [8, 3.1121932627130544, 759],
                              [18, 25.385548232544167, 273], [28, 18.462803704125573, 451], [7, 8.352038784728009, 590],
                              [30, 4.578021299561984, 40], [27, 17.015977151589325, 573], [24, 24.035834325967365, 59],
                              [17, 5.0647834302733985, 518], [5, 5.624610467965313, 777], [32, 9.761261069409471, 535],
                              [10, 14.540964048681401, 230], [26, 5.3179460276953, 843], [5, 3.300483393907268, 454],
                              [26, 21.083883067753764, 715], [17, 10.173402762590806, 311], [17, 9.77585902382209, 76],
                              [26, 19.131842175376537, 115], [13, 15.281779055111759, 602], [16, 20.7723552667938, 785],
                              [7, 9.302726888546076, 489], [21, 11.818365659683005, 404], [4, 5.949471046551301, 191],
                              [20, 18.74029924446589, 828], [2, 4.42274471537374, 411], [3, 9.87009232520681, 441],
                              [18, 9.667523752783444, 447], [13, 23.328548495857927, 640], [1, 9.202133876702812, 822],
                              [32, 21.32851095803753, 560], [27, 7.825597857432514, 9], [28, 12.113031418626214, 276],
                              [10, 29.899983991325847, 429], [31, 6.845893631001814, 789],
                              [25, 30.861258894213233, 686], [3, 14.223216959698064, 568], [25, 4.208394104485931, 345],
                              [29, 26.314460205869555, 296], [18, 16.836568114498522, 657], [7, 5.283161260693153, 155],
                              [0, 8.229077615160836, 842], [10, 5.502090981652262, 639], [14, 17.53607355977392, 216],
                              [9, 15.067091586124498, 147], [20, 6.516414221291224, 446], [8, 20.85335247718928, 297],
                              [30, 20.75729502152303, 264], [29, 18.467201367806542, 362], [4, 6.665957306858304, 41],
                              [25, 10.112966734125493, 469], [10, 4.105832200368531, 561],
                              [28, 12.281491898589833, 150], [6, 13.521892498965958, 524], [11, 9.97653948479341, 514],
                              [4, 5.309454152929756, 378], [27, 13.380277330086257, 613], [28, 17.573860318211516, 252],
                              [4, 2.165300847919102, 379], [27, 1.461755648535874, 64], [2, 7.894532212685311, 886],
                              [31, 10.05797919472342, 747], [30, 17.18908060923558, 189], [13, 9.048827697893266, 471],
                              [24, 24.86042172613015, 97], [32, 10.374517146958217, 94], [25, 10.456895225034117, 706],
                              [22, 19.39167847685914, 407], [15, 14.106577322570468, 779], [28, 20.421380509015176, 63],
                              [4, 7.020052447118294, 342], [4, 7.418670177992703, 846], [15, 4.8643797993427595, 185],
                              [31, 7.705340591385496, 436], [27, 26.389549128543127, 542], [6, 10.437794222026112, 218],
                              [8, 7.47832103568801, 732], [9, 6.498043809344309, 595], [21, 11.7957012605866, 226],
                              [21, 4.22890756006264, 591], [13, 12.76045178910023, 648], [5, 11.763361813501954, 726],
                              [5, 4.20521777073762, 607], [1, 20.36981498483537, 888], [30, 4.989485580998836, 742],
                              [32, 12.946374867351796, 632], [21, 21.297115449753175, 219],
                              [13, 4.568106242058939, 373], [0, 6.495559062328688, 200], [6, 11.208163193397587, 149],
                              [19, 4.543030499345051, 684], [4, 5.4774210153128315, 179], [6, 17.471321512818843, 237],
                              [11, 18.679336210451222, 600], [5, 3.1685782751200473, 341], [23, 8.27285483097959, 461],
                              [18, 16.92828912667335, 256], [26, 24.827185643661714, 762],
                              [16, 29.423137426266187, 857], [11, 7.380836418162799, 23], [27, 12.419370821606927, 878],
                              [16, 12.590164796517229, 77], [14, 28.841481935455228, 119],
                              [30, 20.240584393767648, 416], [6, 10.90176029057699, 883], [14, 8.346982520027181, 693],
                              [5, 5.109367834507216, 27], [12, 3.8515957275330086, 696], [25, 19.39818701980321, 585],
                              [30, 4.3319306442192955, 159], [19, 16.90088630308527, 287], [29, 5.666635464480432, 19],
                              [22, 5.566588404239195, 282], [1, 11.550325292917199, 718], [6, 21.269963527026082, 277],
                              [9, 2.6138660428293767, 193], [6, 4.838088707800778, 791], [21, 12.884443579469409, 78],
                              [25, 10.547857372353064, 215], [18, 18.48863606478077, 90], [30, 8.990768585496737, 361],
                              [31, 24.574639216091988, 655], [6, 15.13670331492913, 238], [0, 22.645645875197346, 56],
                              [27, 16.19616573103519, 428], [23, 10.98680316202156, 885], [18, 21.369972332079588, 325],
                              [15, 3.7784166293669665, 163], [11, 22.143876101740034, 14], [27, 7.149910007535562, 840],
                              [15, 13.762827283856288, 267], [25, 1.3274889739250797, 510],
                              [5, 12.726377216454402, 794], [19, 16.2649843556777, 580], [5, 6.724968933819453, 71],
                              [29, 20.66150904819859, 493], [6, 25.826022825244614, 634], [10, 11.369731315977528, 558],
                              [15, 12.646321624107223, 578], [25, 6.329516742194804, 217],
                              [25, 20.189194011136493, 400], [25, 29.46139986810681, 876], [3, 24.016281378547358, 698],
                              [5, 4.273163955647827, 723]]
        qt_s = 35.65772198166811
        st_s = 13.164126714530921
        # verify my version
        self.reset()
        for i in range(len(student_experience)):
            ev_number = self.__transfer_o_n[student_experience[i][2]]
            cs_number = student_experience[i][0]
            sl_number = self.get_best_slot(cs_number)
            self.step(ev_number, cs_number, sl_number)
        if self.get_average_time()[0] - st_s < 0.01 and self.get_average_time()[1] - qt_s < 0.01:
            print("Student Calculation Verified, Correct!!")
        else:
            print("Student Calculation Verified, Error!!")
        # verify teacher's version
        self.reset()
        for i in range(len(teacher_experience)):
            ev_number = self.__transfer_o_n[teacher_experience[i][2]]
            cs_number = teacher_experience[i][0]
            sl_number = self.get_best_slot(cs_number)
            self.step(ev_number, cs_number, sl_number)
        if self.get_average_time()[0] - st_t < 0.01 and self.get_average_time()[1] - qt_t < 0.01:
            print("Teacher Calculation Verified, Correct!!")
        else:
            print("Teacher Calculation Verified, Error!!")
        self.reset()
        return None

    # 268 µs ± 4.19 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    def find_position(self,
                      ev_number: int) -> Tuple:
        """
        return this scheduled ev position
        :param ev_number: ev number
        :return: position tuple as (scheduling_order, ev_number, cs_number, slot_number, scheduling order)
        """
        assert ev_number in self.__scheduled_ev
        cs_number = None
        slot_number = None
        __index = None
        charging_order = None
        for i in range(len(self.__scheduled_ev)):
            if self.__trace[i][0] == ev_number:
                # if position of ev was founded
                __index = i
                cs_number = self.__trace[i][1]
                slot_number = self.__trace[i][2]
        for i in range(int(self.__SIP[int(cs_number)][int(slot_number)])):
            if self.__SI[int(cs_number)][int(slot_number)][i] == ev_number:
                charging_order = i
        return int(__index), int(ev_number), int(cs_number), int(slot_number), int(charging_order)

    # 3.35 µs ± 67.2 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    def get_average_charging_time(self,
                                  cs_number: int) -> float:
        """
        get average charging time of cs with cs_number
        :param cs_number: number of charging station
        :return: average charging of cs with cs_number
        """
        assert cs_number in range(len(self.__cs_data))
        c_time = 0
        num = 0
        for i in range(self.__slot_number):
            c_time += self.__time[cs_number][i][2]
            num += self.__SIP[cs_number][i]
        average_charging_time = c_time / num
        return average_charging_time

    # unable to calculate
    def get_average_distance_of_ev_to_cs(self,
                                         ev_number: int) -> float:
        """
        get average distance of ev to cs
        :param ev_number: ev number
        :return: a float number that indicates average distance from current position to all other charging stations
        """
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
        """
        get average first k distance of ev to cs
        :param ev_number: ev number
        :param k: number of charging station
        :return: a float number that indicates average distance from current position to (nearest) k charging stations
        """
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

    # 3.37 µs ± 72.7 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    def get_average_idling_time(self,
                                cs_number: int) -> float:
        """
        get average idling time of cs with cs_number
        :param cs_number: number of charging station
        :return: average idling of cs with cs_number
        """
        assert cs_number in range(len(self.__cs_data))
        i_time = 0
        num = 0
        for i in range(self.__slot_number):
            i_time += self.__time[cs_number][i][1]
            num += self.__SIP[cs_number][i]
        average_idle_time = i_time / num
        return average_idle_time

    # 3.34 µs ± 80.9 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    def get_average_queueing_time(self,
                                  cs_number: int) -> float:
        """
        get average queueing time of cs with cs_number
        :param cs_number: number of charging station
        :return: average queueing of cs with cs_number
        """
        assert cs_number in range(len(self.__cs_data))
        q_time = 0
        num = 0
        for i in range(self.__slot_number):
            q_time += self.__time[cs_number][i][0]
            num += self.__SIP[cs_number][i]
        average_queueing_time = q_time / num
        return average_queueing_time

    # 101 µs ± 4.49 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    def get_average_time(self) -> Tuple:
        """
        get current average idle time  average queueing time and average charging as scheduling consequence
        :return: ave_idle, ave_queue, ave_charging
        """
        ave_queue = self.__qt / len(self.__scheduled_ev)
        ave_idle = self.__st / (len(self.__cs_data) * self.__slot_number)
        ave_charging = 0
        for i in range(len(self.__scheduled_ev)):
            ave_charging += self.__ev_ct[self.__scheduled_ev[i]]
        ave_charging /= len(self.__scheduled_ev)
        return ave_idle, ave_queue, ave_charging

    # 177 µs ± 2.22 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    def get_average_time_for_cs(self) -> Tuple:
        """
        # get average time for every charging station
        :return: average_queueing_time, average_idling_time, average_charging_time
        """
        average_charging_time = np.zeros(len(self.__cs_data))
        average_idling_time = np.zeros(len(self.__cs_data))
        average_queueing_time = np.zeros(len(self.__cs_data))
        for i in range(len(self.__cs_data)):
            # for every charging station ,we calculate as below
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

    # 1.58 µs ± 14.4 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    def get_best_slot(self,
                      cs_number: int) -> int:
        """
        get slot number of slot that has minimal queueing time
        :param cs_number: charging station number
        :return: slot number of slot that has minimal queueing time
        """
        assert cs_number in range(len(self.__cs_data))
        best_slot = int(sum(self.__SIP[cs_number]) % self.__slot_number)
        return best_slot

    # 94.8 ns ± 3.96 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_brief_time_matrix(self) -> np.ndarray:
        """
        return self.__time as a brief time imformation collection
        :return: self.__time
        """
        ret = self.__time
        return ret

    # 1.41 µs ± 104 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    def get_charging_ev_number_for_concrete_cs(self,
                                               cs_number: int) -> int:
        """
        get charging ev quantity of concrete charging station
        :param cs_number: charging station number
        :return: charging ev quantity of concrete charging station
        """
        sum_ = 0
        for i in range(self.__slot_number):
            sum_ += self.__SIP[cs_number][i]
        return sum_

    # 115 ns ± 6.69 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_charging_time(self,
                          ev_number: int) -> float:
        """
        return charging time that ev numbered with ev_number needed
        :param ev_number: ev number
        :return: charging time that ev numbered with ev_number needed
        """
        return self.__ev_ct[ev_number]

    # 125 ns ± 15 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_cs_cp_coordination(self,
                               cs_number: int):
        """
        return charging price of charging station with number cs_number
        :param cs_number: charging station number
        :return: charging price of charging station with number cs_number
        """
        price = self.__cp[cs_number]
        return price

    # 112 ns ± 1.65 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_cs_x_coordination(self,
                              cs_number: int):
        """
        return x-coordination of charging station cs_number
        :param cs_number: charging station number
        :return: x coordiantion of charging station numbered with cs_number
        """
        x = self.__cs_x[cs_number]
        return x

    # 112 ns ± 1.79 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_cs_y_coordination(self,
                              cs_number: int):
        """
        return y-coordination of charging station cs_number
        :param cs_number: charging station number
        :return: y coordiantion of charging station numbered with cs_number
        """
        y = self.__cs_y[cs_number]
        return y

    # 6.51 ms ± 76.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    def get_current_cs_state(self,
                             sel_ev_number: int) -> np.ndarray:
        """
        # return current cs state matrix that get from environment
        :return: current cs state matrix that get from environment
        """
        assert sel_ev_number in self.__schedulable_ev_list
        ev_left_travel_distance = np.array([self.__ev_ld[i] for i in self.__backup_for_schedulable_ev_number])
        ev_expecting_charging_time = np.array([self.__ev_ct[i] for i in self.__backup_for_schedulable_ev_number])
        distance_between_ev_and_cs = np.array(self.__distance)[self.__backup_for_schedulable_ev_number]
        for i in range(len(self.__schedulable_ev_list)):
            if self.__transfer_o_n[i] in self.__scheduled_ev:
                # if any ev was scheduled ,the set line it matched zero
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
        return cs_state  # (1, 5, 36)

    # 6.49 ms ± 24 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    def get_current_ev_state(self) -> np.ndarray:
        """
        # return current ev state matrix that get from environment
        :return: current ev state matrix that get from environment
        """
        ev_left_travel_distance = np.array([self.__ev_ld[i] for i in self.__backup_for_schedulable_ev_number])
        ev_expecting_charging_time = np.array([self.__ev_ct[i] for i in self.__backup_for_schedulable_ev_number])
        distance_between_ev_and_cs = np.array(self.__distance)[self.__backup_for_schedulable_ev_number]
        for i in range(len(self.__schedulable_ev_list)):
            if self.__transfer_o_n[i] in self.__scheduled_ev:
                # if any ev was scheduled ,the set line it matched zero
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
        )  # (899, 36)
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
        ev_state = np.concatenate(
            (
                ev_part,
                cs_part
            ),
            axis=0
        )[np.newaxis, :]
        return ev_state  # (1, 903, 36)

    # 96.9 ns ± 1.46 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_detailed_time_matrix(self) -> np.ndarray:
        """
        return self.__time as a detailed time information collection
        :return: self.__time_for_ev
        """
        ret = self.__time_for_ev
        return ret

    # unable to access
    def get_distance(self,
                     ev_number: int,
                     cs_number: int) -> float:
        """
        return the distance from EV to CS
        :param ev_number: ev number
        :param cs_number: cs number
        :return: distance
        """
        distance = self.__distance[ev_number][cs_number]
        return distance

    # 109 ns ± 1.56 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_ev_charging_time(self,
                             ev_number: int) -> float:
        """
        # return charging time that ev numbered with ev_number need
        :param ev_number: electrical vihrcle number
        :return: charging time that ev numbered with ev_number need
        """
        return self.__ev_ct[ev_number]

    # 112 ns ± 1.6 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_ev_left_travel_distance(self,
                                    ev_number: int) -> float:
        """
        return  left travel distance of ev numbered with ev_number
        :param ev_number: electrical vehicle number
        :return: left travel distance of ev numbered with ev_number
        """
        return self.__ev_ld[ev_number]

    # 197 ns ± 8.67 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_ev_position(self,
                        ev_number: int) -> Tuple:
        """
        return ev's position whose number is ev_number
        :param ev_number: electrical vehicle
        :return: pair(x, y)
        """
        x = self.__ev_x[ev_number]
        y = self.__ev_y[ev_number]
        return x, y

    # 123 ns ± 4.98 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_ev_x_coordination(self,
                              ev_number: int) -> float:
        """
        return x coordiantion of ev numbered with ev_number
        :param ev_number:
        :return: x-axis of ev numbered with ev_number
        """
        x = self.__ev_x[ev_number]
        return x

    # 116 ns ± 1.15 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_ev_y_coordination(self,
                              ev_number: int) -> float:
        """
        return y coordiantion of ev numbered with ev_number
        :param ev_number:
        :return: y-axis of ev numbered with ev_number
        """
        y = self.__ev_y[ev_number]
        return y

    # it could be run in an extremely short time
    def get_file(self) -> None:
        """
        get file name list used to calculate line numbers
        :param self: this pointer of class
        :return: None
        """
        for parent, dirnames, filenames in os.walk(self.__basedir):
            for filename in filenames:
                ext = filename.split('.')[-1]
                if ext in self.whitelist:
                    self.filelists.append(os.path.join(parent, filename))
        return None

    # 112 ns ± 1.25 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_how_many_ev_need_to_be_scheduled(self) -> int:
        """
        return the number that how many ev need to be scheduled
        :return: number that how many ev need to be scheduled
        """
        return len(self.__not_scheduled_ev)

    # 135 ns ± 12.2 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_how_many_ev_have_been_scheduled(self) -> int:
        """
        return the number that how many ev have been scheduled
        :return: the number that how many ev have been scheduled
        """
        return len(self.__scheduled_ev)

    # 92.8 ns ± 3.14 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_idling_time(self) -> float:
        """
        :return idling time
        :return: idling time
        """
        return self.__st

    # 101 ns ± 9 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_not_scheduled_ev(self) -> list:
        """
        return a list which stores all not-scheduled-ev number
        :return: a list which stores all not-scheduled-ev number
        """
        return self.__not_scheduled_ev

    # 90.6 ns ± 0.88 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_queueing_time(self):
        """
        return queueing time
        :return: queueing time
        """
        return self.__qt

    # 93.5 ns ± 0.505 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_raw_state_matrix(self) -> np.ndarray:
        """
        return self.__state
        :return: self.__state
        """
        return self.__state

    # 217 µs ± 6.97 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    def get_reachable_cs_list_for_ev(self,
                                     ev_number: int) -> np.ndarray:
        """
        get reachable cs list for ev with ev number ev_number
        :param ev_number: ev number
        :return: reachable cs list for ev with ev number ev_number
        """
        reachable_cs_list_for_ev = np.array(self.__reachable_cs_for_ev)[ev_number]
        return reachable_cs_list_for_ev

    # 487 ns ± 6.04 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    def get_reachability_of_an_ev(self,
                                  ev_number: int) -> np.ndarray:
        """
        return reachability list for ev numbered with ev_number
        :param ev_number: ev number
        :return: a list or array that stores the information of reachability about ev numbered with ev_number
        """
        reachability = np.array(self.__reachability[ev_number])
        return reachability

    # 93.5 ns ± 2.87 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_reachability_matrix(self) -> np.ndarray:
        """
        return reachability matrix
        :return: reachability matrix
        """
        return self.__reachability

    # 706 ns ± 10.2 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    def get_reward(self,
                   q: float = 0.8,
                   i: float = 0.2
                   ) -> float:
        """
        return weighted reward from environment
        :param q: weight for queueing time
        :param i: weight for idling time
        :return: weighted reward from environment
        """
        reward = self.__qt / len(self.__scheduled_ev) * q + self.__st / (len(self.__cs_data) * self.__slot_number) * i
        return reward

    # unable to access
    def get_reward_for_cs(self,
                          cs_number: int,
                          ev_number: int) -> float:
        """
        get reward for cs
        :param cs_number: charging station number
        :param ev_number: electric vehicle number
        :return: values return from environment
        """
        distance = self.__distance[ev_number][cs_number]
        occupied = np.sum(self.__cd_wt[cs_number]) / self.__slot_number
        reward = occupied - distance
        return reward

    # 1.83 µs ± 16.3 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    def get_reward_for_ev(self,
                          ev_number: int,
                          cs_number: int,
                          slot_number: int,
                          q: float = 0.8,
                          i: float = 0.2
                          ) -> float:
        """
        get reward for an ev that queueing before a slot of charging slot
        :param ev_number: electrical vehicle number
        :param cs_number: charging station number
        :param slot_number: slot number that ev scheduled at
        :param q: weight for queueing time
        :param i: weight for idling time
        :return: weighted reward for target ev
        """
        idle = 0
        queue = 0
        for pointer in range(int(self.__SIP[cs_number][slot_number])):
            if self.__SI[cs_number][slot_number][pointer] == ev_number:
                # if target ev information was found
                queue = self.__time_for_ev[cs_number][slot_number][pointer][0]
                idle = self.__time_for_ev[cs_number][slot_number][pointer][1]
                break
        weighted_reward = idle * i + queue * q
        return weighted_reward

    # 32.8 µs ± 630 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    def get_schedulable_ev(self) -> np.ndarray:
        """
        get all schedulable ev number whether it was scheduled or not
        :return: all schedulable ev number array whether it was scheduled or not
        """
        result = np.array(self.__schedulable_ev_list)
        return result

    # 103 ns ± 1.03 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_scheduled_ev(self) -> list:
        """
        return a list which stores all scheduled-ev number
        :return: a list which stores all scheduled-ev number
        """
        return self.__scheduled_ev

    def get_sip(self) -> np.ndarray:
        """
        get sip
        :return: sip
        """
        return self.__SIP

    # 95.8 ns ± 1.74 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def get_slot_count_on_one_charging_station(self) -> int:
        """
        get slot count on one charging station
        :return: slot count on one charging station
        """
        return self.__slot_number

    # 255 ns ± 0.994 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    def get_travel_time_from_ev_to_cs(self,
                                      ev_number: int,
                                      cs_numer: int) -> float:
        """
        get travel time from position of an ev numbered with ev_number to chargingstation cs_number
        :param ev_number: number of lectric vehicle
        :param cs_numer: number of charging station
        :return: travel time from position of an ev numbered with ev_number to chargingstation cs_number
        """
        return self.__distance[ev_number][cs_numer]

    # 134 ns ± 1.8 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def is_done(self) -> bool:
        """
        return whether this scheduling process was done
        :return: a boolean value
        """
        boolean_value = len(self.__not_scheduled_ev) == 0
        return boolean_value

    def optimize(self) -> List:
        """
        优化调度结果
        :return:
        """
        # 针对所有充电桩下的电车进行顺序重排序并进行二次调度
        # 首先复制一份调度结果
        old_result = []  # 这里应当是三维数组，用来保存每一个充电桩下的调度策略
        new_result = []
        for i in range(len(self.__cs_x)):
            sub_ = []  # 用来保存每一个充电站下的结果
            for j in range(self.__slot_number):
                if self.__SIP[i][j] == 0:
                    pass
                if self.__SIP[i][j] != 0:
                    r_in_slot_ = self.optimize_(i, j)
                    sub_.append(r_in_slot_)
            new_result.append(sub_)
        # 根据结果进行重新调度
        self.reset()
        for i in range(len(new_result)):
            for j in range(len(new_result[i])):
                for k in range(len(new_result[i][j])):
                    self.step(new_result[i][j][k], i, j)
        return new_result

    def optimize_(self,
                  cs_number: int,
                  sl_number: int) -> List:  ## old
        """
        optimize scheduling result
        :return: None
        """
        assert self.__SIP[cs_number][sl_number] != 0
        SI = self.__SI[cs_number][sl_number]
        SIP = self.__SIP[cs_number][sl_number]
        distance = []
        for i in range(SIP):
            distance.append(self.get_distance(SI[i], cs_number))
        # 找出当前所有的电车占用充电桩的时间
        ev_no_list = []
        occup_list = []
        final = []
        # 定义充电桩被占用的时间
        cs_occup = 0
        for i in range(SIP):
            charging_time = self.get_ev_charging_time(SI[i])
            travel_time = self.get_distance(SI[i], cs_number)
            """ ev_no_list 与 occup_list 在顺序上是一一对应的关系 """
            ev_no_list.append(SI[i])
            occup_list.append(charging_time + 2*travel_time)
        first_ev = True
        # 如果当前是空闲的，那么就找到距离充电桩最近的
        for i in range(SIP):
            # 对所有的电车,找出最短占用时间的电车
            if cs_occup == 0:
                # 此时找到一个距离充电桩最近的电车
                ev_index = distance.index(min(distance))
                ev_no = ev_no_list[ev_index]
            else:
                ev_index = occup_list.index(min(occup_list))
                ev_no = ev_no_list[ev_index]
            # 将该电车从备选列表中删除
            ev_no_list.remove(ev_no)
            occup_list.remove(occup_list[ev_index])
            # 将该电车加入到最终结果列表
            final.append(ev_no)
            # 更新当前充电桩被占用的时间
            if first_ev:
                cs_occup += self.__distance[ev_no][cs_number] + self.get_ev_charging_time(ev_no)
                first_ev = False
            else:
                if cs_occup >= self.__distance[ev_no][cs_number]:
                    cs_occup += self.get_ev_charging_time(ev_no)
                else:
                    cs_occup = self.__distance[ev_no][cs_number] + self.get_ev_charging_time(ev_no)
            # 更新剩余的每一辆电车对当前充电桩的占用时间
            for i_ in range(len(occup_list)):
                ev = ev_no_list[i_]
                cs = cs_number
                if cs_occup >= self.__distance[ev][cs]:
                    # 将对充电桩的占用时间缩减为充电时间
                    occup_list[ev_no_list.index(ev)] = self.get_ev_charging_time(ev)
                else:
                    # 在原有占用时间的基础上减去当前充电桩被占用的时间
                    occup_list[ev_no_list.index(ev)] = (self.get_ev_charging_time(ev) + 2 * (self.__distance[ev][cs] - cs_occup))
        return final

    # unable to access
    def print_scheduling_info_for_concrete_slot(self,
                                                cs_number: int,
                                                slot_number: int) -> None:
        """
        print scheduling details after all scheduling process in one trajectory was done
        :return: None
        """
        print("CS--{:5s},SLOT--{:5s}--({:})---->".format(str(cs_number),
                                                         str(slot_number),
                                                         int(self.__SIP[cs_number][slot_number])), end=" ")
        for k in range(int(self.__SIP[cs_number][slot_number])):
            print(int(self.__SI[cs_number][slot_number][k]))
        return None

    # unable to access
    def print_scheduling_info(self) -> None:
        """
        print scheduling details after all scheduling process in one trajectory was done
        :return: None
        """
        for i in range(len(self.__cs_data)):
            for j in range(self.__slot_number):
                print("CS--{:5s},SLOT--{:5s}--({:})---->".format(str(i),
                                                                 str(j),
                                                                 int(self.__SIP[i][j])), end=" ")
                for k in range(int(self.__SIP[i][j])):
                    print(int(self.__SI[i][j][k]), end="\t")
                print("")
        return None

    # unable to access
    def print_scheduling_info_for_one_cs(self,
                                         cs_number: int) -> None:
        """
        print scheduling details after all scheduling process in one trajectory was done
        :return: None
        """
        for j in range(self.__slot_number):
            print("CS--{:5s},SLOT--{:5s}--({:})---->".format(str(cs_number),
                                                             str(j),
                                                             int(self.__SIP[cs_number][j])), end=" ")
            for k in range(int(self.__SIP[cs_number][j])):
                print(int(self.__SI[cs_number][j][k]), end="\t")
            print("")
        return None

    # unable to access
    def print_scheduling_consequence_info(self) -> None:
        """
        print average idle time and average queueing time after scheduling
        :return: None
        """
        print("|||",
              "average idling time = {:.3f}".format(self.__st / 170),
              "|||",
              "average queueing time = {:.3f}".format(self.__qt / 899),
              "|||", end="\t")
        return None

    def print_scheduling_consequence_list(self) -> list:
        """
        print scheduling consequence list
        :return: None
        """
        assert self.is_done() == True
        result = []  # ev cs sl
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

    # unable to access
    def print_state_of_cs(self,
                          sel_ev_number: int) -> None:
        """
        print state of cs
        :return: None
        """
        pd.set_option('display.max_columns',
                      None)
        pd.set_option('display.max_rows',
                      None)
        print(pd.DataFrame(data=self.get_current_cs_state(sel_ev_number=sel_ev_number)))
        return None

    # unable to access
    def print_state_of_ev(self) -> None:
        """
        print state of ev
        :return: None
        """
        pd.set_option('display.max_columns',
                      None)
        pd.set_option('display.max_rows',
                      None)
        print(pd.DataFrame(data=self.get_current_ev_state()))
        return None

    # 90.4 ms ± 1.97 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    def reset(self) -> None:
        """
        reset all state variables etc. back to initial state
        :return: None
        """
        self.__init__(self.__filename_of_ev_information,
                      self.__filename_of_cs_information,
                      self.__filename_of_cp_information)
        return None

    def ret_scheduling_consequence_list(self) -> list:
        """
        print scheduling consequence list
        :return: None
        """
        assert self.is_done() == True
        result = []  # ev cs sl
        for i in range(34):
            for j in range(5):
                for k in range(self.__SIP[i][j]):
                    temp = [self.__SI[i][j][k], i, j]
                    result.append(temp)
        return result

    # 6.55 s ± 12.3 ms per 899 loop (mean ± std. dev. of 7 runs, 1 loop each) at around 7.29ms every call
    def step(self,
             ev_number: int,
             cs_number: int,
             slot_number: int) -> None:
        """
        dispatch a tram to the designed location
        :param ev_number: electrical number
        :param cs_number: charging station number
        :param slot_number: slot number
        :return: None
        """
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

        # 增补内容
        if self.__cd_wt[cs_number][slot_number] < 60:
            self.hour_charging_ev += 1

        if not self.__teacher_version:
            self.calculate()
        else:
            self.calculate_teacher_version()
        return None

    def get_hour_charing_ev(self):
        return self.hour_charging_ev

    # 151 ns ± 1.34 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def transfer_ev_no_to_order(self,
                                ev_number: int) -> int:
        """
        transfer ev_no to order
        :param ev_number: real ev number
        :return: relative order of ev numbered with ev_number in dataset
        """
        assert ev_number in self.__schedulable_ev_list
        return self.__transfer_n_o[ev_number]

    # 108 ns ± 0.297 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def transfer_ev_order_to_no(self,
                                ev_order: int) -> int:
        """
        transfer ev_order to real ev number
        :param ev_order: relative ev_order
        :return: real order of ev numbered with ev_number in dataset
        """
        return self.__transfer_o_n[ev_order]

    # 6.55 s ± 12.3 ms per 899 loop (mean ± std. dev. of 7 runs, 1 loop each) at around 7.29ms every call
    def unstep(self,
               ev_number: int) -> None:
        """
        undo pair number is ev_ No. tram dispatching
        :param ev_number: electric vihecle
        :return: None
        """
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
