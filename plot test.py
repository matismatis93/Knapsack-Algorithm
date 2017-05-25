import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    xar = 10
    yar = [5] * 10
    ax1.clear()
    ax1.plot(xar,yar)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()