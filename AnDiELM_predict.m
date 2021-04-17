function [metrics, output] = AnDiELM_predict(X,T, model)

%
% Input:
% X      - features matrix
% T      - ground truth
% model  - path to saved model
%
% Output:
% metrics - MAE for regression, accuracy for classification
% output - exponents for regression, model scores for classification
%
%
% Carlo Manzo, UVic-UCC, July 2020 - carlo.manzo@uvic.cat


load(model);
NData=size(X,1);
NInpNeur=mod.NInpNeur;

switch mod.task
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


H=mod.w*X';
H=H+repmat(mod.bias,1,NData);

% Calculate hidden neuron output matrix H
switch mod.activ
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


%   predictions
Y=(H' * mod.beta)';

switch mod.task
    case {1} % regression
        %   MAE
        metrics=mean(abs(T - Y));
        output=Y;
    case {2} % classification
        [~,gt]=max(T,[],1);
        [~,pred]=max(Y,[],1);
        %   accuracy
        metrics=length(find(gt-pred==0))/NData;
        output=Y-repmat(min(Y,[],1),size(Y,1),1);
        output=output./repmat(sum(output,1),size(Y,1),1);
        
    case {3}%
        [~,gt]=max(T,[],1);
        [~,pred]=max(Y,[],1);
        %   RMSE
        metrics=sqrt(mean((gt - (pred)).^2));
        output=Y;%-repmat(min(Y,[],1),size(Y,1),1);
        
end


