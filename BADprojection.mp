'''
Puloma Bishnu and Manu Amundsen
Back projection for PulsON440
'''
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
dist_vals = []

def dist_form(x, y, z):
    value = math.sqrt(x**2 + y**2 + z**2) 
    return value

with open('2Points_1way_data.pkl', 'rb') as f:
    data = pickle.load(f)
'''
abs_val_img = np.abs(data['scan_data'])
plt.imshow(abs_val_img)
'''
def test_dist(platform_pos):
    for trip in platform_pos:
        dist = dist_form(trip[0], trip[1], trip[2])
        dist_vals.append(dist)
    return dist_vals

final = test_dist(data['platform_pos'])
print(final)

final_len = len(final)
slow_time = range(1, final_len)

plt.plot(final[:99], slow_time)
plt.axis([15, 17, 0, 100])
plt.show()
    
'''
pyg_thrm = 
d = distance between platform (radar) and point

PSEUDOCODE ---
def backprojection:
for each x in grid:
        for each y in grid:
            sum = 0
            for each scan:
                use pythagorean theorem to find distance from radar to the x, y point
                get scan value, add it to sum
            set the value of x,y on the output to be equal to sum
        output the entire matric of sums
        
def back_projection(x-input vector, y-platform position, z-pulse time): 
  Use pythagorean theorum to find distance between the platform and the point 
  Use the range to time function to find the time shift between the pulse and the return
  Shift the return pulses by the time that the signal takes to travel to line up portions of the waves
  set value equal to matched_filtering (shifted pulses, original pulse, t)
  add value to sigma dictionary


matched_filtering(pulses-list of all the return pulses, org-outputted pulse, t-return time):
  Multiply the original pulse by each of the other pulses
  Sum each equivalent pulse on the seperate waveforms 
  return the value at t

In order to iterate over all of the x and y pairs in the system, use two nested for loops with a back projection call inside of the second

back_projection():
  For each x in the grid:
    for each y in the grid:
      sum = 0
      for each scan:
        use pythagorean theorum to find distance from radar to the x, y point
        get scan value, add it to sum
      set the value of (x, y) on the output to be equal to sum
  output the entire matrix of sums
'''
