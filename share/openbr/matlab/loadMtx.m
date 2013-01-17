function [x2] = loadMtx(filename,isRowMajor)
% [x2] = loadMtx(filename,reverse)
%
% Loads a *.mtx file into a Matlab matrix. 
%       'filename' - the name of the mtx file
%       'isRowMajor' - determines whether the 
%                   matrix is read in row major or
%                   column major order (optional, default is false)
%
% This file can likely be improved in terms of effeciency. 

   
    if nargin < 2,
        reverse = false;
    end

    z = fopen(filename,'r');
    if z == -1,
        fprintf('Error opening file %s\n',filename);
        x2 = 0;
        return
    end

    buf = zeros(100,1);
    str_list = cell(4,1);
    for i = 1:4,
        cnt = 0;
        while true
            cnt = cnt + 1;
            [a ] = fread(z,1,'uchar');
            if a == 10
                break;
            end
            buf(cnt) = a;
        end
        str_list{i} = char(buf(1:cnt-1)');
    end

    s = strsplit(' ',str_list{4});
    x = str2num(s{2});
    x1 = str2num(s{3});
    typ = s{1}(2);

    %x2 = fread(z,[x1  x],'float32');
    if strcmp(typ,'F')
        x2 = fread(z,[x1  x],'single');
    elseif strcmp(typ,'B')
        x2 = fread(z,[x1  x],'int8');
    else
        assert(0);
    end
    x2 = x2';
    fclose(z);

    if reverse,
        sz = size(x2);
        x2 = x2';
        x2 = reshape(x2,sz);
    end