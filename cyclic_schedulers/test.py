from cyclic_schedulers.reversed_cos_annealing_scheduler import ReversedCosAnnealingScheduler
import matplotlib.pyplot as plt
if __name__ == "__main__":
    rc = ReversedCosAnnealingScheduler(0.0, 1.0, 10, 2.0)
    values = []
    for i in range(100):
        values.append(rc.step())

    plt.plot(values)
    plt.show()