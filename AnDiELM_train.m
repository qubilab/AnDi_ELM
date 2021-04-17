function [metrics, elapsed] = AnDiELM_train(X,T, task, NHidNeur, activ, model)
%
% Input:
% X      - features matrix
% T      - ground truth
% task   - 1 for regression; 2 for classification
% NHidNeur - Number of hidden neurons assigned to the ELM
% activ    - Type of activation function:
%                           'lin' for linear
%                           'relu' for ReLU
%                           'sig' for Sigmoi
% model    - name and path of the model to save
%
% Output:
% metrics - MAE for regression, accuracy for classification
%
%
% Carlo Manzo, UVic-UCC, July 2020 - carlo.manzo@uvic.cat


NData=size(X,1);
NInpNeur=size(X,2);

switch task
    case 1 %regression
        T=2*((T-min(T))./(max(T)-min(T)))-1;
    case 2 %classification
        T=T+1;
        T = 2*[(T==1:max(T))]-1;
    case 3 %classification
        T=(T==1:199);
        T = 2*[T]-1;
end
T=T';



tic

% input weights
w=rand(NHidNeur,NInpNeur)*2-1;
H=w*X';
% bias
bias=rand(NHidNeur,1);
H=H+repmat(bias,1,NData);

% Calculate hidden neuron output matrix H
switch activ
    case 'sig'
        %Sigmoid
        H = 1 ./ (1 + exp(-H));
    case 'lin'
        %Linear
    case 'relu'
        %ReLU
        H(find(H<0))=0;
        %H(find(H>1))=1;
end


% % dropout
% p=1; % if p=1, no dropout
% mask=zeros(NData,NHidNeur);
% loc=find(mean(H')<median(H'));
% loc(2,:)=binornd(1,p,size(loc,2),1);
% mask(:,loc(1,:))=repmat(loc(2,:),NData,1);
% H=H.*mask';

% output weights
beta=pinv(H') * T';

%   elapsed time for training
elapsed=toc;

%   predictions
Y=(H' * beta)';



switch task
    case {1}% regression
        %   MAE
        metrics=mean(abs(T - Y));
        
    case {2}
        [~,gt]=max(T,[],1);
        [~,pred]=max(Y,[],1);
        %   accuracy
        metrics=length(find(gt-pred==0))/NData;
    
    case {3}% 
        [~,gt]=max(T,[],1);
        [~,pred]=max(Y,[],1);
        %   RMSE
        metrics=sqrt(mean((gt - (pred)).^2));    
        
end
mod=struct('task',task, 'NInpNeur', NInpNeur, 'w', w, 'bias', bias,'activ', activ, 'beta', beta);
save(model, 'mod');


