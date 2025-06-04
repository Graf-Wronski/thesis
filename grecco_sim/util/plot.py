import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def set_two_hours_x_axis(ax: plt.axis) -> plt.axis:
    """ Two hour time step on x-axis is well readable. """

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    return ax