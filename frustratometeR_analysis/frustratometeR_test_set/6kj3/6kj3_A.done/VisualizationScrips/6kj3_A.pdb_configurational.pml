load 6kj3_A.pdb, 6kj3_A
hide line,6kj3_A
unset dynamic_measures
show cartoon,6kj3_A
color grey,6kj3_A
run draw_links.py
distance min_frst_wm_6kj3_A= (6kj3_A//A/1/CA),(6kj3_A//A/3/CA)
distance min_frst_wm_6kj3_A= (6kj3_A//A/4/CA),(6kj3_A//A/6/CA)
zoom all
hide labels
color red, max_frst_wm_6kj3_A
color green, min_frst_wm_6kj3_A