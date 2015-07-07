function [eigenvalues, eigenvectors, meanVector, V]=PCA(X,varargin)
%function [eigenvalues, eigenvectors, meanVector, V]=PCA(X)
    cnt = 1;
    doVar = false;
    doEigs = false;

    if nargin < 2,
    end
    while cnt < length(varargin)
        switch varargin{cnt}
            case 'VarEnergy'
                doVar = true;
                varPercent = varargin{cnt+1};
            case 'nEigs'
                doEigs = true;
                eigKeep = varargin{cnt+1};
            otherwise
                fprintf('Error, unknown argument %s\n',varargin{cnt});
                return
        end
        cnt = cnt + 2;
    end


    [Row Column]=size(X);

    %Mean center X
    meanVector = mean(X,2);  meanVector = meanVector(:);
    M=repmat(meanVector,1,Column);
    X=X-M;

    C=X'*X./Column;
    [V,D]=eig(C);
    eigenvalues=diag(D);
    
    %Ordered by eigenvalues%
    [eigenvalues,Index]=sort(eigenvalues,'descend');
    
    V=V(:,Index) ;   %V1 is the the eigenvectors got from X'X;
    eigenvectors=X*V;%eigenvectors is the eigenvectors for XX';
    
    %normalize%
    NV=sum(eigenvectors.^2);
    NV=NV.^(1/2);
    
    %normalize eigenvectors
    NM=repmat(NV,Row,1);
    eigenvectors=eigenvectors./NM;

    if doVar
        d = cumsum(eigenvalues)/ sum(eigenvalues);
        [a1 a2] = max(d > varPercent);
        eigenvalues = eigenvalues(1:a2);
        eigenvectors = eigenvectors(:,1:a2);
    end

    if doEigs
        eigenvalues = eigenvalues(1:eigKeep);
        eigenvectors = eigenvectors(:,1:eigKeep);
    end

    %normalize V1;
    NN=repmat(NV,[Column,1]);
    V=V./NN;



end
