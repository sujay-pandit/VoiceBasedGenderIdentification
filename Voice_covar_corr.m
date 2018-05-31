%% Import Data
clear all;
run('loadData1.m');
run('loadLabels.m');
[r,c]=size(voice_data);
data_cov=cov(voice_data);
data_corr=corrcov(data_cov);
data_pca=pca(voice_data);
for j=1:c
    mn=min(voice_data(:,j));
    mx=max(voice_data(:,j));
    for i=1:r
        voice_data(i,j)=(voice_data(i,j)-mn)/(mx-mn);
    end
end
for i=1:r
    if(voice_label(i,1)=="male")
        voice_data(i,c+1)=1;
    else
        voice_data(i,c+1)=2;
    end
end
data_covn=cov(voice_data);
data_corrn=corrcov(data_covn);
data_pcan=pca(voice_data);
max_corr=zeros(length(data_corr));
max_corrsum=zeros(length(data_corr),1);
for i=1:length(data_corr)
    count=1;
    for j=1:length(data_corr)
        if(abs(data_corr(i,j))>0.7)
            max_corr(i,count)=j;
            max_corrsum(i,1)=max_corrsum(i,1)+1;
            count=count+1;
        end
    end
end
freq_ele=zeros(20,1);
for i=1:20
    for j=1:14
        if(max_corr(i,j)~=0)
        freq_ele(max_corr(i,j),1)=freq_ele(max_corr(i,j),1)+1;
        end
    end
end
count=1;
for i=1:length(max_corr)
    if((max_corr(i,1)<5)||(max_corr(i,1)==0))
        D(:,count)=voice_data(:,i);
        feature_list(count,1)=i;
        count=count+1;
    end
end
D_cov=cov(D);
D_corr=corrcov(D_cov);
%%
stem(1:20,max_corrsum);
title('Correlation Analysis')
ylabel('Number of features with correlation greater than 0.5')
xlabel('Features')

