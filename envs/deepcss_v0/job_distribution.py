import numpy as np


class Dist:

    def __init__(self, max_job_len):
        self.max_job_len = max_job_len

        self.job_small_chance = 0.8

        self.job_len_big_lower = max_job_len * 2 / 3
        self.job_len_big_upper = max_job_len

        self.job_len_small_lower = 1
        self.job_len_small_upper = max_job_len / 5

    def normal_dist(self):
        # TODO complete here
        pass

    def bi_model_dist(self):

        # -- job length --
        if np.random.rand() < self.job_small_chance:  # small job
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)

        return nw_len
