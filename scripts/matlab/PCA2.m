function [eigenvalues, eigenvectors, Mean_Vector]=PCA2(X,varRetain)
% [eigenvalues, eigenvectors, Mean_Vector]=PCA2(X)
%
%Compute the eienvectors of X when the sample number is larger than the feature lenght 

[Row Column]=size(X);
Mean_Vector=mean(X,2);
m=repmat(Mean_Vector(:),1,Column);
X=X-m;
C=X*X'./Column;
[V,D]=eig(C);
eigenvalues=diag(D);
%Ordered by eigenvalues%
[eigenvalues,Index]=sort(eigenvalues);
eigenvalues=eigenvalues(end:-1:1);
Index=Index(end:-1:1);
eigenvectors=V(:,Index);

if nargin == 2,
    d = cumsum(eigenvalues)/ sum(eigenvalues);
    [a1 a2] = max(d > varRetain);
    eigenvalues = eigenvalues(1:a2);
    eigenvectors = eigenvectors(:,1:a2);
end
