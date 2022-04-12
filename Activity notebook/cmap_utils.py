import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def generate_custom_cmap():
    blue = np.array([0.2, 0.2, 0.9])
    black = np.array([0.0, 0.0, 0.0])
    red = np.array([1.0, 0.3, 0.2])

    green = np.array([0.0, 181 / 255, 26 / 255])
    yellow = np.array([1.0, 1.0, 0.0])
    blue2 = np.array([0.0, 56 / 255, 123 / 255])
    orange_fluo = np.array([1.0, 77 / 255, 6 / 255])

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    cms = [

            mpl.colors.LinearSegmentedColormap.from_list(
                name="", colors=[[0.0, 0.0, 0.0], [1.0, 0.2, 0.2], [1.0, 1.0, 1.0]]
            ),
            mpl.colors.LinearSegmentedColormap.from_list(
                name="",
                colors=[
                    (0, [0.0, 0.0, 0.0]),
                    (0.70, [1.0, 0.2, 0.2]),
                    (0.71, [1.0, 1.0, 1.0]),
                    (1.00, [1.0, 1.0, 1.0]),
                ],
            ),
            mpl.colors.LinearSegmentedColormap.from_list(
                name="", colors=[[1.0, 0.2, 0.2], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
            ),
            mpl.colors.LinearSegmentedColormap.from_list(
                name="", colors=[(0, red), (0.50, black), (0.51, black), (1.00, blue)]
            ),
            mpl.colors.LinearSegmentedColormap.from_list(
                name="", colors=[(0, red), (0.30, black), (0.71, black), (1.00, blue)]
            ),
            mpl.colors.LinearSegmentedColormap.from_list(
                name="",
                colors=[(0, black), (0.50, blue * 0.5), (0.51, red * 0.6), (1.00, red)],
            ),
            mpl.colors.LinearSegmentedColormap.from_list(
                name="",
                colors=[(0, green), (0.30, black), (0.71, black), (1.00, yellow)],
            ),
            mpl.colors.LinearSegmentedColormap.from_list(
                name="",
                colors=[(0, blue2), (0.30, black), (0.71, black), (1.00, orange_fluo)],
            ),
            mpl.colors.LinearSegmentedColormap.from_list(
                name="",
                colors=[(0, green), (0.30, black), (0.71, black), (1.00, orange_fluo)],
            ),
        ]
    cmap_names = []
    for i in range(0, len(cms)):
        plt.register_cmap(name="AACustom" + str(i), cmap=cms[i])
        cmap_names.append("AACustom" + str(i))
    return cmap_names
