import numpy as np

# src_size = (640, 480) # (horizontal size, vertical size)
# picam_fov = np.array((54,41)) # degrees

# mid_pt = np.array(src_size)/2
# track_bbs_ids = np.array([[ 111.,   2., 472., 192.,   1.],[0., 20., 100., 50., 2.],[0., 10., 90., 50., 3.]])
# center_pts = np.transpose(
#     np.array([np.mean(track_bbs_ids[:,[1, 3]], axis = 1), 
#               np.mean(track_bbs_ids[:,[0, 2]], axis = 1)])) # (+right, +down)
# dist_to_center = np.linalg.norm(center_pts-mid_pt,axis=1) 
# min_dist_idx = np.argmin(dist_to_center)

# delta_az_el_pixels = -(center_pts[min_dist_idx]-mid_pt) # (+right, +down)
# delta_az_el_degrees = picam_fov/src_size*delta_az_el_pixels
# delta_az_el_arcsecs = delta_az_el_degrees*3600

# delta_az_el_str = f"a{delta_az_el_arcsecs[0]:.0f}e{delta_az_el_arcsecs[1]:.0f}"

# print(center_pts-mid_pt) 
# print(dist_to_center) 
# print(min_dist_idx)
# print(delta_az_el_pixels)
# print(delta_az_el_degrees)
# print(delta_az_el_arcsecs)
# print(delta_az_el_str)

# FRIEND_HUE_BOUNDS = np.concatenate((np.arange(0, 16, 1, dtype=int), np.arange(165, 181, 1, dtype=int)), axis = 0)
FRIEND_HUE = lambda x : (0<=x<=15) | (165<=x<=180)
FRIEND_SAT = lambda x : (50 <= x)

img = np.random.randint(0, 255, (50,50))
print(np.sum(FRIEND_SAT(img))/np.size(img))
