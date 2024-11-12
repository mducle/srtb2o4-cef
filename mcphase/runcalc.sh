mkdir -p results
#cif2mcphas SrTb2O4_30341-ICSD.cif
#makenn 10 -rkky 0.4 0.5
#mcphasit >& results/mcphasit.out
powdermagnon 0.1 3 0.1  0.1  0 40
mcdispit -minE 0 -maxE 40
powdermagnon -r results/mcdisp.qei 0 40 1 > results/powdermagnon_lr.dat
powdermagnon -r results/mcdisp.qei 0 5 0.05 > results/powdermagnon_le.dat
