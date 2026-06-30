load 3q2x_A.pdb, 3q2x_A
hide line,3q2x_A
unset dynamic_measures
show cartoon,3q2x_A
color grey,3q2x_A
run draw_links.py
draw_links resi 1 and name CA and Chain A and 3q2x_A, resi 6 and name CA and Chain A and 3q2x_A, color=red, color2=red, radius=0.05, object_name=1:6_red_3q2x_A
draw_links resi 4 and name CA and Chain A and 3q2x_A, resi 6 and name CA and Chain A and 3q2x_A, color=green, color2=green, radius=0.05, object_name=4:6_green_3q2x_A
zoom all
hide labels
color red, max_frst_wm_3q2x_A
color green, min_frst_wm_3q2x_A