%% Import Data
clear all;
run('loadData1.m');
run('loadLabels.m');
%% Scaling data using min-max method
[r,c]=size(voice_data);

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
%Creating testing and Training data
trainData(1:1056,:)=voice_data(1:1056,:);
trainData(1057:2112,:)=voice_data(1585:2640,:);
testData(1:528,:)=voice_data(1057:1584,:);
testData(529:1056,:)=voice_data(2641:3168,:);
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

%% Training on Inbuilt SVM

featurelist=[1:20];
for j=1:10
    for i=1:length(featurelist)
        trData(:,i)=trainData(:,featurelist(i),j);
        teData(:,i)=testData(:,featurelist(i),j);
    end
    SVM_model = fitcsvm(trData,trainData(:,21,j));

    %% Testing on generated SVM model

    [testData(:,22,j)] = predict(SVM_model,teData);

    accuracy_fitcSVM(1,j)=calc_accuracy(testData(:,21,j),testData(:,22,j));
    
end
acc_svm=mean(accuracy_fitcSVM);

%% User-Defined Functions
function accuracy=calc_accuracy(arr,arr2)
c=0;
    for i=1:length(arr)
        if arr(i,1)==arr2(i,1)
            c=c+1;
        end
    end
    accuracy=(c/length(arr))*100;
end

