'''
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
numpy.inter can help collapse the for loops so it can be faster

for x pixels:
    for y pixels:
        for scans: (figure out how to move this for loop outside and eliminate the first two)
            computer-range
            find_nearest_range
            imgt = scan_nearest_range
'''


import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
with open('Mandrill_1way_data.pkl', 'rb') as f:
    data=pickle.load(f)

def backprojection(pulses, range_bins, platform_pos, x_vec, y_vec, z_offset=0):
    num_pulses=pulses.shape[0]
    num_x_coords=len(x_vec)
    num_y_coords=len(y_vec)
    complex_image=np.zeros((num_y_coords, num_x_coords), dtype=np.complex)

    for i in range(num_x_coords):

        for j in range(num_y_coords):

            for k in range(num_pulses):
                one_way_range=np.sqrt((x_vec[i]-platform_pos[k,0])**2 +
                (y_vec[j]-platform_pos[k,1])**2 +
                (z_offset-platform_pos[k,2])**2)

                closest_idx=np.argmin(np.abs(one_way_range-range_bins))
                complex_image[j,i] += pulses[k, closest_idx]

    return complex_image

yeet=backprojection(data['scan_data'], data['range_bins'], data['platform_pos'], (np.array(list(range(101)))-50) * 0.05 , (np.array(list(range(101)))-50) * 0.05)
yeet = np.abs(yeet)
plt.imshow(yeet)
plt.show()






PSEUDOCODE ---
def backprojection:
for each x in grid:
        for each y in grid:
            sum = 0
            for each scan:
                use pythagorean theorem to find distance from radar to the x, y point
                get scan value, add it to sum
            set the value of x,y on the output to be equal to sum
        output the entire matrix of sums
        
In order to iterate over all of the x and y pairs in the system, use two nested for loops with a back projection call inside of the second

