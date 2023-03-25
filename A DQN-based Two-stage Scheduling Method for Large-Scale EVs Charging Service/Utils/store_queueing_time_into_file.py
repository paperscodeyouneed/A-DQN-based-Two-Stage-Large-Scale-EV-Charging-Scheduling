

from Environment.environment import *


class Calculate(object):
    """
        将最后1000条经验的排队时间计算出来存入文件
    """
    def __init__(self):
        """
            初始化
        """
        self.env = Environment()
        self.file_ = open("../Data/trimmed datas/Trimmed_Raw_Experience_1000.txt", "r")

    def cal(self):
        """
        calcuolate
        :return: None
        """
        exp = self.file_.readlines()
        for k in range(1000):
            print(k)
            episode = eval(exp[k])
            for i in range(899):
                ev_number_ = self.env.transfer_ev_order_to_no(episode[i][2])
                cs_number_ = episode[i][0]
                self.env.step(ev_number_, cs_number_, self.env.get_best_slot(cs_number_))
            self.env.calculate()
            res = self.env.get_average_time()
            with open("../Data/pre-trained data/pre_train_value_list", "a+") as file:
                file.write(str(res))
                file.write("\n")
            self.env.reset()


if __name__ == '__main__':
    c = Calculate()
    c.cal()
