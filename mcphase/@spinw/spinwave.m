function spectra = spinwave(obj, hkl, varargin)
    global mcphasedir; mcphasedir = '~/src/mcphase';
    global numproc; numproc = 6;
    spectra = mcphase_sqw(hkl(1,:), hkl(2,:), hkl(3,:), obj);
end
