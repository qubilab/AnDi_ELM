function out=read_data(task,dimen, Mtlag, ff, pathname, gt_flag)

fid=fopen(fullfile(pathname,['task',num2str(task),'.txt']));
num_features=ff*Mtlag+1;
data=[];
N=[];
if gt_flag==1
    fid2=fopen(fullfile(pathname,['ref',num2str(task),'.txt']));
    gt=[];
else fid2=fid1;
end

i=0;
while ~feof(fid) && ~feof(fid2) %i<traj_numb
    line=fgetl(fid);
    D=str2num(line);
    if gt_flag==1
        line2=fgetl(fid2);
        F=str2num(line2);
    end
    if D(1)==dimen
        i=i+1;
        [d,n]=preprocess(D, Mtlag, ff);
        data=[data; d];
        if gt_flag==1
            gt=[gt; F(2)];
        end
        N=[N; n];
    end
    clear g d n line D line2 F
end
fclose(fid);
fclose(fid2);
if gt_flag==1
out.gt=gt;
end
out.N=N;
out.data=data;
out.traj_numb=i;
