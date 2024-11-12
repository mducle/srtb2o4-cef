function set_exchange(parfunc)
    fid = fopen('makenn.d');
    xx = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    xx = xx{1};
    iscomment = cellfun(@(x)x(1)=='#', xx);
    ijkdata = cell2mat(cellfun(@(x)sscanf(x,'%f')', xx(~iscomment), 'UniformOutput', false));
    %0.38./(ijkdata(:,7).^3)
    ijkdata(:,4:6) = repmat(parfunc(ijkdata(:,7)), 1, 3);
    ijkdata(:,7) = [];
    xx(~iscomment) = cellfun(@(x)sprintf('%12g ',x), mat2cell(ijkdata, ones(1,616), 6), 'UniformOutput', false);
    fid = fopen('mcphas.j', 'w');
    fprintf(fid, '%s', cell2mat(cellfun(@(x)sprintf('%s\n',x), xx, 'UniformOutput', false)'));
    fclose(fid);
end

