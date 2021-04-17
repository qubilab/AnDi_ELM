function out=read_data_T3(task,dimen, pathname, gt_flag)

fid=fopen(fullfile(pathname,['task',num2str(task),'.txt']));

data=[];

if gt_flag==1
    fid2=fopen(fullfile(pathname,['ref',num2str(task),'.txt']));
    gt=[];
else fid2=fid1;
end

i=0;
while ~feof(fid) && ~feof(fid2) 
    line=fgetl(fid);
    D=str2num(line);
    if gt_flag==1
        line2=fgetl(fid2);
        F=str2num(line2);
    end
    if D(1)==dimen
        i=i+1;
        traj=reshape(D(2:end),[],D(1));
        displa=traj(:,:)-traj(1,:); % set initial point to 0
        displa0=displa(1+1:end,:) - displa(1:end-1,:);  % calculate displacement (tlag=1)
        displa0=displa0./repmat(std(displa0),size(displa0,1),1); % standardize displacements (tlag=1)
        d=sum((displa0).^2,2);
        %        d=sum(diff(reshape(D(2:end),[],D(1))).^2,2);
        data=[data; d'];
        if gt_flag==1
            gt=[gt; F(2:end)'];
        end
    end
    clear  d traj displa displa0 line D line2 F
end
fclose(fid);
fclose(fid2);
if gt_flag==1
out.gt=gt;
end

out.data=data;
out.traj_numb=i;
