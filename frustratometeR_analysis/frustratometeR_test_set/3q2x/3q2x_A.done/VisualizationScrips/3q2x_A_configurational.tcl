set sel1 [atomselect top "resid 1 and name CA and chain A"]
set sel6 [atomselect top "resid 6 and name CA and chain A"]
# get the coordinates
lassign [atomselect0 get {x y z}] pos1
lassign [atomselect1 get {x y z}] pos2
# draw a green line between the two atoms
draw color red
draw line $pos1 $pos2 style solid width 2
set sel4 [atomselect top "resid 4 and name CA and chain A"]
set sel6 [atomselect top "resid 6 and name CA and chain A"]
# get the coordinates
lassign [atomselect2 get {x y z}] pos1
lassign [atomselect3 get {x y z}] pos2
# draw a green line between the two atoms
draw color green
draw line $pos1 $pos2 style solid width 2

mol modselect 0 top all
mol modstyle 0 top newcartoon
mol modcolor 0 top colorid 15
