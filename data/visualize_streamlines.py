#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from synthetic_streamlines import SyntheticStreamlineGenerator

if __name__ == "__main__":
    generator = SyntheticStreamlineGenerator()

    fig = plt.figure(figsize=(12, 8))

    tract_names = [
        "Straight",
        "Curved",
        "Fan",
        "Crossing"
    ]

    for i in range(4):

        ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        for _ in range(10): # 10 streamlines

            streamline = generator.generate_streamline(i)

            ax.plot(streamline[:, 0], streamline[:, 1], streamline[:, 2])

        ax.set_title(tract_names[i])

    plt.tight_layout()
    plt.show()