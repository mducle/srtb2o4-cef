set dgrid3d 500,50 qnorm 2; set view map;
set term png
set output 'results/mcphase_lowE.png'
set zrange[0:2]; splot 'results/powdermagnon_le.dat' u 5:6:7 w pm3d

set output 'results/mcphase_highE.png'
set zrange[0:10]; splot 'results/powdermagnon_lr.dat' u 5:6:7 w pm3d
