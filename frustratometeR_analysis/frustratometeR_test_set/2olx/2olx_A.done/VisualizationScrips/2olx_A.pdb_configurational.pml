load 2olx_A.pdb, 2olx_A
hide line,2olx_A
unset dynamic_measures
show cartoon,2olx_A
color grey,2olx_A
run draw_links.py
distance min_frst_wm_2olx_A= (2olx_A//A/1/CA),(2olx_A//A/3/CA)
zoom all
hide labels
color red, max_frst_wm_2olx_A
color green, min_frst_wm_2olx_A