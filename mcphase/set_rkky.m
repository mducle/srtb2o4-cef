function set_rkky(A,k)
    global mcphasedir 
    if ispc; sp = '\'; else; sp = '/'; end
    [rs,rv]=perl([mcphasedir sp 'bin' sp 'makenn.pl'],'10','-rkky',num2str(A),num2str(k));
    if(length(rs)>8e3); rs=rs((end-8e3):end); end
    if rv~=0; error('Failed to run makenn program to set exchange. Output is:\n%s',rs); end
    copyfile('results/makenn.j', 'mcphas.j');
end
