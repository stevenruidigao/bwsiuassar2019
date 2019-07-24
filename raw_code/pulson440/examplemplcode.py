plt.imshow(np.abs(data["scan_data"]),
           extent=(data["range_bins"][0, 0], data["range_bins"][0, -1]
                   data["scan_timestamps"][-1] - data["scan_timestamps"][0], 0), vmin=400, vmax=800)
plt.xlabel("Range (m)")
plt.ylabel("Time Elapsed (s)")
plt.title("Alignment View")
plt.colorbar()
