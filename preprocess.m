function [d,n]=preprocess(D, Mtlag, ff)

%pretreatment
traj=reshape(D(2:end),[],D(1));
n=size(traj,1);

displa=traj(:,:)-traj(1,:); % set initial point to 0
displa0=displa(1+1:end,:) - displa(1:end-1,:);  % calculate displacement (tlag=1)
%
displa0=displa0./repmat(std(displa0),size(displa0,1),1); % standardize displacements (tlag=1)
displat=cumsum([ zeros(1, D(1)); displa0],1); % (re)make trajectories
%


for tlag=1:Mtlag
    displa2=displat(tlag+1:end,:) - displat(1:end-tlag,:);  % calculate displacement (tlag>=1)
    displa2=abs(displa2(:));   % calculate absolute displacement (tlag>=1)
    d(1,1+(tlag-1)*ff)=log(mean(displa2))/log(tlag+1); % feature #1 log of mean displacement over log of tlag
    d(1,2+(tlag-1)*ff)=log(mean(displa2.^2))/log(tlag+1); % feature #2 log of mean squared displacement over log of tlag
    clear displa2
end

% extra feature, only for last tlag

plag=1;
displa0=abs(displa0);
ddispla0=displa0(plag+1:end,:).*displa0(1:end-plag,:);
d(1,2+(tlag-1)*ff+1)=mean(ddispla0(:))./mean(displa0(:).^2); % correlation of displacement
%
