import numpy as np
import matplotlib
from matplotlib import pyplot
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import perceptron

cwa = 28285714.0
cwb = 28342286.0
cma = 16438.0
cmb = 24658.0
ca = 1869229.0
clca = 2479843.0

density = 1000


def delta3(p_of_ah_scrap, p_of_ah):
    result = (2 * cwa + ca) * (p_of_ah_scrap - 1) * p_of_ah + 2 * cwa + ca + cma + cmb
    return result


def delta4(p_of_ah_scrap, p_of_ah):
    result = 2 * cwa * p_of_ah * (p_of_ah_scrap - 1) + 2 * cwa + clca * (1 - p_of_ah) + cma + cmb
    return result


def delta8(p_of_ah_scrap, p_of_ah):
    result = (2 * cwb * p_of_ah_scrap + ca * p_of_ah_scrap - 2 * cwb - ca) * p_of_ah + 2 * cwb + ca + cmb
    return result


def delta16(p_of_ah_scrap, p_of_ah):
    result = cwb * (p_of_ah_scrap * p_of_ah + (1 - p_of_ah) + (p_of_ah_scrap * p_of_ah + (1 - p_of_ah))) + clca * (
    1 - p_of_ah) + cmb
    return result


def clf(X, Y, Z):
    fig, ax = pyplot.subplots()
    cs = ax.contourf(X, Y, Z, levels=[-1, 0, 1, 2, 3])
    # cset = ax.contour(X, Y, Z, zdir='x', offset=120, cmap=cm.coolwarm)
    ax.axis('on')

    proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
             for pc in cs.collections]

    plt.legend(proxy, ["delta3(independent)", "delta4(independent)", "delta8(dependent)", "delta16(dependent)"])
    plt.ylabel("prior probability of etching machine")
    plt.xlabel("likelihood probabilty of etching machine")


if __name__ == "__main__":
    test = np.arange(0, 1, 1.0/density)
    funcs = [delta3, delta4, delta8, delta16]

    result = []
    for y in test:  # row y
        result.append(list())
        for x in test:  # col x
            minimum = None
            minimum_index = None
            for index, func in enumerate(funcs):
                value = func(x, y)
                if minimum is None or value < minimum:
                    minimum = value
                    minimum_index = index
            result[-1].append(minimum_index)

    clf(X=test, Y=test, Z=result)

    pyplot.show()