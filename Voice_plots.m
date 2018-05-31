clear all;
%% Import Data
run('loadData1.m');
run('loadLabels.m');
load('roc_nb.mat')
load('roc_LR.mat')
load('roc_LR_target.mat')
load('roc_knn3.mat')
load('roc_knn5.mat')

%% User Defined Functions
%[accuracy_fitcknn,accuracy_mynb]=NB_KNN(voice_data,voice_label);
%% Scaling data using min-max method
[r,c]=size(voice_data);

for j=1:c
    mn=min(voice_data(:,j));
    mx=max(voice_data(:,j));
    for i=1:r
        voice_data(i,j)=(voice_data(i,j)-mn)/(mx-mn);
    end
end

%% Dividing data into Experimental Data and Test Data
% Class-1 for Male Class-2 for Female
for i=1:r
    if(voice_label(i,1)=="male")
        voice_data(i,c+1)=1;
    else
        voice_data(i,c+1)=2;
    end
end
trainData(1:1056,:,1)=voice_data(1:1056,:);
trainData(1057:2112,:,1)=voice_data(1585:2640,:);
testData(1:528,:,1)=voice_data(1057:1584,:);
testData(529:1056,:,1)=voice_data(2641:3168,:);

trainData(1:1056,:,2)=voice_data(1585:2640,:);
trainData(1057:2112,:,2)=voice_data(1:1056,:);
testData(1:528,:,2)=voice_data(2641:3168,:);
testData(529:1056,:,2)=voice_data(1057:1584,:);

trainData(1:1056,:,3)=voice_data(529:1584,:);
trainData(1057:2112,:,3)=voice_data(2113:3168,:);
testData(1:528,:,3)=voice_data(1:528,:);
testData(529:1056,:,3)=voice_data(1585:2112,:);


trainData(1:1056,:,4)=voice_data(2113:3168,:);
trainData(1057:2112,:,4)=voice_data(529:1584,:);
testData(1:528,:,4)=voice_data(1585:2112,:);
testData(529:1056,:,4)=voice_data(1:528,:);


trainData(1:528,:,5)=voice_data(1:528,:);
trainData(529:1056,:,5)=voice_data(1057:1584,:);
trainData(1057:1584,:,5)=voice_data(1585:2112,:);
trainData(1585:2112,:,5)=voice_data(2641:3168,:);
testData(1:528,:,5)=voice_data(529:1056,:);
testData(529:1056,:,5)=voice_data(2113:2640,:);

trainData(1:528,:,6)=voice_data(1057:1584,:);
trainData(529:1056,:,6)=voice_data(1:528,:);
trainData(1057:1584,:,6)=voice_data(2641:3168,:);
trainData(1585:2112,:,6)=voice_data(1585:2112,:);
testData(1:528,:,6)=voice_data(529:1056,:);
testData(529:1056,:,6)=voice_data(2113:2640,:);


trainData(1:528,:,7)=voice_data(2641:3168,:);
trainData(529:1056,:,7)=voice_data(1:528,:);
trainData(1057:1584,:,7)=voice_data(1057:1584,:);
trainData(1585:2112,:,7)=voice_data(1585:2112,:);
testData(1:528,:,7)=voice_data(2113:2640,:);
testData(529:1056,:,7)=voice_data(529:1056,:);

trainData(1:528,:,8)=voice_data(529:1056,:);
trainData(529:1056,:,8)=voice_data(1585:2112,:);
trainData(1057:1584,:,8)=voice_data(2641:3168,:);
trainData(1585:2112,:,8)=voice_data(1:528,:);
testData(1:528,:,8)=voice_data(2113:2640,:);
testData(529:1056,:,8)=voice_data(1057:1584,:);

trainData(1:528,:,9)=voice_data(2641:3168,:);
trainData(529:1056,:,9)=voice_data(2113:2640,:);
trainData(1057:1584,:,9)=voice_data(1057:1584,:);
trainData(1585:2112,:,9)=voice_data(1:528,:);
testData(1:528,:,9)=voice_data(1585:2112,:);
testData(529:1056,:,9)=voice_data(529:1056,:);

trainData(1:528,:,10)=voice_data(1585:2112,:);
trainData(529:1056,:,10)=voice_data(2113:2640,:);
trainData(1057:1584,:,10)=voice_data(529:1056,:);
trainData(1585:2112,:,10)=voice_data(1057:1584,:);
testData(1:528,:,10)=voice_data(2641:3168,:);
testData(529:1056,:,10)=voice_data(1:528,:);
%% PLOTTING
tData=testData(:,21,8);
% for i=1:20
%     scatter([1:1584],voice_data(1:1584,i),'s');
%     hold on
%     scatter([1:1584],voice_data(1585:3168,i),'o')
%     figure
% end
for i=1:1056
    if(tData(i,1)==1)
        teData(:,i)=[1;0];
    else
        teData(:,i)=[0;1];
    end
    if(predict_class_knn5(i,1)==1)
        knn5(:,i)=[1;0];
    else
        knn5(:,i)=[0;1];
    end
    if(predict_class_nb(i,1)==1)
        nb(:,i)=[1;0];
    else
        nb(:,i)=[0;1];
    end
    if(roc_knn3(i,1)==1)
        knn3(:,i)=[1;0];
    else
        knn3(:,i)=[0;1];
    end
end
[Xlogknn3,Ylogknn3,Tlogknn3,AUClogknn3] = perfcurve(teData(1,:),knn3(1,:),'1');
[Xlogknn5,Ylogknn5,Tlogknn5,AUClogknn5] =  perfcurve(teData(1,:),knn5(1,:),'1');
[Xlognb,Ylognb,Tlognb,AUClognb] = perfcurve(teData(1,:),nb(1,:),'1');
[Xloglr,Yloglr,Tloglr,AUCloglr] = perfcurve(roc_LR_target',roc_LR','1');
plot(Xlogknn3,Ylogknn3);
hold on;
plot(Xlogknn5,Ylogknn5);
hold on;
% plotroc(teData(1,:),knn3(1,:),'KNN3');
% figure 
% plotroc(teData(1,:),knn5(1,:),'KNN5');
% figure 
% plotroc(teData(1,:),nb(1,:),'NB');
% figure 
% plotroc(roc_LR_target',roc_LR','LR');

