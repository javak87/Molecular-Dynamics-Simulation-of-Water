import numpy as np
import matplotlib.pyplot as plt

O2_Count = np.array([64, 125, 216, 343])
consumed_time_naive = np.array ([189, 602, 1625, 3930])
consumed_time_linked_cell = np.array ([207, 651, 1742, 4115])
consumed_time_PD_Linked_cell = np.array ([265, 662, 1747, 4103])

# plot Convergence Analysis
fig, ax = plt.subplots()
fig.suptitle("Benchmark of LJ Force", fontsize=20)

ax.set_xlabel('The number of oxyegen atoms', fontsize=15)
ax.set_ylabel('Consumed time (second)', fontsize=15)

ax.plot (O2_Count, consumed_time_naive, color='red', label='Naive')
ax.plot (O2_Count, consumed_time_linked_cell, color='blue', label='Linked cell List')
ax.plot (O2_Count, consumed_time_PD_Linked_cell, color='green', label='Periodic Linked cell List')
plt.legend(loc="left")
plt.legend(prop={"size":14})
plt.tick_params(axis='x', labelsize=13)
plt.tick_params(axis='y', labelsize=13)
plt.show()