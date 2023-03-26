
from Environment.environment import *


def squeeze_pre_trained_data() -> List:
    pre_trained_data = []
    with open("../Data/trimmed datas/Trimmed_Raw_Experience_1000.txt") as file:
        for line in file:
            pre_trained_data.append(eval(line))
    return pre_trained_data
