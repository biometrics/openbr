function [subspaceData]=LDA(X,classNo,varargin)
% [subspaceData]=LDA(X,classNo)
%
%   LDA method for learning a subspace that seeks to maximize the Fisher 
%       seperability measure. 'X' is d x n matrix, where d is the feature
%       vector size, and n is the number of instaces. 'classNo' is a n x 1
%       vector that indicates which class/subject each instance belongs to.
%
%   Optional parameters:

%       'EnergyRetain' - percentage of variance (0.0 to 1.0) to retain in the initial  
%                           PCA step. (default = 0.98)
%
%       'do_direct' - Whether or not to perform Direct LDA (default = false)
%
%       'FixedRetain' - Keep a fixed number of eigenvectors in the initial PCA.
%                           If used, then specify number to keep (e.g. 100)
%
%
%       'DominantEig' - whether or not to use the dominant eigenvector method. If 
%                           the d >> n,  then this should be set to true. 
%                           (default = true)
%
%       'SkipPCA'     - whether of not to skip the PCA step. If Sw is believed to 
%                           be non-singular then PCA step can be safely skipped. 
%                           (default = false)                
%
%       'ScaleW'      - the factor by which to scale the within-class scatter matrix. 
%                           This controls the importance of the between and within 
%                           class scatter to each other in the fisher criterion
%
%   The output 'subspaceData' is a struct that contains the following important fields:
%       subspaceData.mean - d x 1 vector containing the mean of the training data 
%       subspaceData.W    - d x k LDA projection matrix, where k is the number of 
%                                subspace dimensions
%
%
%   Algorithm based on Fukanaga's LDA method. This code was orginally written by Zhifeng Li,
%       and has since been modified and improved by Brendan Klare.
%
%

    useFixedEnergy = false;
    doDominantEig = true;
    doPCA = true;
    useFixedEig = false;
    energyPercentage = .98;
    ScaleW = 1;
    doDLDA = false;
    do_null = false;

    cnt = 1;
    while cnt < length(varargin)
        switch varargin{cnt}
            case 'EnergyRetain'
                useFixedEnergy = true;
                energyPercentage = varargin{cnt+1};
            case 'FixedRetain'
                useFixedEig = true;
                fixedEig = varargin{cnt+1};
            case 'DominantEig'
                doDominantEig = varargin{cnt+1};
            case 'SkipPCA'
                doPCA = ~varargin{cnt+1};
            case 'ScaleW'
                ScaleW = varargin{cnt+1};
            case 'do_direct'
                do_null = varargin{cnt+1};
            otherwise
                fprintf('Error, unknown argument %s\n',varargin{cnt});
                return
        end
        cnt = cnt + 2;
    end


    [FeatureLength SampleNumber]=size(X);
    % ClassNum=round(SampleNumber/2);

    [a1 a2 classNo] = unique(classNo); 
    ClassNum = max(classNo);

    %Calculate eigenspace from X
    if doPCA
        if ~doDominantEig
            [eigenvalues, eigenvectors, Mean_Vector]=PCA2(X);
        else
            [eigenvalues, eigenvectors, Mean_Vector, V1]=PCA(X);
        end

        if useFixedEnergy
            d1 = cumsum(eigenvalues)./sum(eigenvalues);
            [a nEigs] = max(d1 > energyPercentage);   
        elseif useFixedEig
            nEigs = fixedEig;
        else
            nEigs = min(FeatureLength,SampleNumber - 1);
        end

        %Select eigenvectors
        Select_eigenvectors=eigenvectors(:,1:nEigs);
        eigenvalues = eigenvalues(1:nEigs);

        %Project the sample data on to the eigenvectors
        W=Select_eigenvectors'*(X-repmat(Mean_Vector,1,SampleNumber));
    else
        W = X;
        Mean_Vector = zeros(size(X,1),1);
        nEigs = size(X,1);
        Select_eigenvectors = eye(size(X,1));
        eigenvalues = ones(size(X,1),1);
    end  

    %Caculate the centers for each class
    ClassCenters = zeros(nEigs,ClassNum);
    for i = 1:ClassNum
        ClassCenters(:,i) = mean(W(:,classNo == i),2);
    end

    for i = 1:ClassNum,
        W(:,classNo == i) = W(:,classNo == i) - repmat(ClassCenters(:,i),1,sum(classNo == i));
    end

    if ScaleW ~= 1
        W = ScaleW .* W;
    end

    [W_val, W_vec, W_m]=PCA2(W);

    if ~do_null
        nDim2 = min(nEigs,SampleNumber - ClassNum);
        SW_val=W_val(1:nDim2);
        SW_vec=W_vec(:,1:nDim2);
        SW_vec=SW_vec./(repmat(SW_val',[size(SW_vec,1) 1]).^0.5);
    else
        nDim2 = nEigs;
        SW_val = W_val;
        SW_vec = W_vec;
        if nEigs > SampleNumber - ClassNum
            SW_val(SampleNumber-ClassNum+1:end) = SW_val(SampleNumber-ClassNum)/2;
        end

        d1 = cumsum(W_val)/sum(W_val);
        [d1 start_idx] = max(d1 > .1);

        SW_vec = SW_vec(:,start_idx:end);
        SW_val = SW_val(start_idx:end);
        nDim2 = size(SW_vec,2);
        SW_vec=SW_vec./(repmat(SW_val',[size(SW_vec,1) 1]).^0.15);
    end


    m = mean(W,2);
    M=repmat(m(:),1,ClassNum);
    mean2 = m;
    B=SW_vec'*(ClassCenters-M);
    % Between_Class_Matrix=B*B';


    [B_val,B_vec,B_m]=PCA2(B);

    nDim3 = min(ClassNum-1,nDim2);
    SB_vec=B_vec(:,1:nDim3);

    subspaceData.mean = Mean_Vector(:);    
    subspaceData.mean2 = mean2(:);
    subspaceData.W1 = Select_eigenvectors;
    subspaceData.D1 = eigenvalues;
    subspaceData.W2 = SW_vec;
    subspaceData.W3 = SB_vec;
    subspaceData.W = (subspaceData.W3' * subspaceData.W2' * subspaceData.W1')';
