% Carlo Manzo, UVic-UCC, July 2020 - carlo.manzo@uvic.cat
%
% example to use AnDi-ELM developed for the AnDi challenge
%
clear all
close all
clc
%
task=2; % 1 inference of alpha, 2 classification of model, 3 trajectory segmentation
dimen=3; % number of dimension of trajectory 
%
% features parameters
Mtlag=7; % maximum tlag (<10)
ff=2; % number of features for each tlag (2 defined per time lag)
%
saveflag=1; % if 1 saves preprocessed data
%
% data preprocessing and feature calculations
pathname_train='../../data/development for training';
pathname_test='../../data/challenge for scoring';

switch task
    case{1,2}
    train=read_data(task,dimen, Mtlag, ff, pathname_train, 1);
    test=read_data(task,dimen, Mtlag, ff, pathname_test, 1);
    case {3}
    train=read_data_T3(task,dimen, pathname_train, 1);
    test=read_data_T3(task,dimen, pathname_test, 1);
end
if saveflag==1
save(['datasets_task',num2str(task),'_',num2str(dimen),'D.mat'], 'train', 'test')
end
%
%
%% train and test
%
task=2; % 1 inference of alpha, 2 classification of model, 3 trajectory segmentation
dimen=3; % number of dimension of trajectory 
%
m=1000; % number of hidden layers for the ELM
%
load(['datasets_task',num2str(task),'_',num2str(dimen),'D.mat'])
savepath=['./model_task',num2str(task),'_',num2str(dimen),'D.mat'];
% train/test
switch task
    case {1,2}
        [train_data,train_mu,train_sigma]=zscore(train.data);
        train_gt=train.gt;
        metr = AnDiELM_train(train_data, train_gt,task, m, 'sig',savepath);
        test_data=(test.data-repmat(train_mu,size(test.data,1),1))./repmat(train_sigma,size(test.data,1),1);
        test_gt=test.gt;
        [metr2,out] = AnDiELM_predict(valid_data, valid_gt, savepath);
        metr2
    case {3}
        train_data=cumsum(train.data,2);
        train_data=2*(train_data./repmat(max(train_data,[],2),1,size(train.data,2)))-1;
        train_gt=train.gt(:,1);
        metr = AnDiELM_train(train_data, train_gt,task, m, 'sig',savepath);
        test_data=cumsum(test.data,2);
        test_data=2*(test_data./repmat(max(test_data,[],2),1,size(test.data,2)))-1;
        test_gt=test.gt(:,1);
        [metr2,out] = AnDiELM_predict(test_data, test_gt, savepath);
end
%
% plots
switch task
    case {1}
        out=out+1;
        MAE=mean(abs(test_gt' - out)) %TestingAccuracy
        
        figure(1)
        X = [test_gt, out'];
        hist3(X,'edges',{-0.025:0.05:2 -0.025:0.05:2});
        xlabel('gt'); ylabel('pred');
        set(gcf,'renderer','opengl');
        set(get(gca,'child'),'FaceColor','interp','EdgeColor','interp','CDataMode',...
            'auto');
        cmap = getPyPlot_cMap('OrRd', 128);
        colormap(cmap)
        colorbar
        view(2)
        xlim([0 2])
        ylim([0 2])
    case {2} 
        models={'ATTM'; 'CTRW'; 'FBM'; 'LW'; 'SBM'};
        [~,pred]=max(out,[],1);
        pred=pred-1;        
        [ micro, macro ] = micro_macro_PR( test_gt , pred);
        micro.fscore
        figure(1)
        D = confusionmat(test_gt,pred);
        confusionchart(D,models);
      case {3} 
        [~,pred]=max(out,[],1);
        RMSE=sqrt(mean((test_gt' - pred).^2)) 
         figure(1)
         X = [test_gt, pred'];
         hist3(X,'edges',{0.5:4:200.5 0.5:4:200.5});
         xlabel('gt'); ylabel('pred');
         set(gcf,'renderer','opengl');
         set(get(gca,'child'),'FaceColor','interp','EdgeColor','interp','CDataMode','auto');
         cmap = getPyPlot_cMap('OrRd', 128);
         colormap(cmap)
         colorbar
         view(2)
         xlim([1 199])
         ylim([1 199])
end