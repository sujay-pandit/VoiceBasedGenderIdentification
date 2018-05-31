clear all;
%% Import Data
run('loadData1.m');
run('loadLabels.m');

[r,c]=size(voice_data);

for j=1:c
    mn=min(voice_data(:,j));
    mx=max(voice_data(:,j));
    men=mean(voice_data(:,j));
    for i=1:r
        voice_data(i,j)=(voice_data(i,j)-men)/(mx-mn);
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

%% 

for j=1:10
    v=cov(trainData(:,1:20,1));
    [V,U]=eig(v);
    for f=1:20
    pcalist=[1:f];
    [coeff,score,latent,tsquared,explained]=pca(trainData(:,1:length(pcalist),j));

    %scre=trainData(:,1:20,j)*coeff;
    score(:,length(pcalist)+1)=trainData(:,21,j);
    test_score=testData(:,1:length(pcalist),j)*coeff;
    test_score(:,length(pcalist)+1)=testData(:,21,j);
    %pcalist=[1,2,4,6];
    
    Hypothesis_func_knn = fitcknn(score(:,1:length(pcalist)),score(:,length(pcalist)+1),'NumNeighbors',5);%KNN Classifier
    for i=1:1056
    predicted_class=predict(Hypothesis_func_knn,test_score(i,1:length(pcalist)));
    test_score(i,length(pcalist)+2)=predicted_class;
    end
    accuracy_fitcknn5(1,f,j)=calc_accuracy(test_score(:,length(pcalist)+1),test_score(:,length(pcalist)+2));
    Hypothesis_func_knn = fitcknn(score(:,1:length(pcalist)),score(:,length(pcalist)+1),'NumNeighbors',3);%KNN Classifier
    for i=1:1056
    predicted_class=predict(Hypothesis_func_knn,test_score(i,1:length(pcalist)));
    test_score(i,length(pcalist)+2)=predicted_class;
    end
    accuracy_fitcknn3(1,f,j)=calc_accuracy(test_score(:,length(pcalist)+1),test_score(:,length(pcalist)+2));
    end
end
%% Mean ACCuracy
for i=1:20
    acc_knn3(i,1)=mean(accuracy_fitcknn3(1,i,:));
    acc_knn5(i,1)=mean(accuracy_fitcknn5(1,i,:));
end
%% PLotting


%     for i=1:2112
%         if(score(i,21)==1)
%             tr_score_male(i,:)=score(i,1:2);
%         
%         elseif(score(i,21)==2)
%             tr_score_female(i,:)=score(i,1:2);
%         end
%     end
%     for i=1:1056
%         if(test_score(i,21)==1)
%             te_score_male(i,:)=test_score(i,1:2)
%         
%         elseif(test_score(i,21)==2)
%             te_score_female(i,:)=test_score(i,1:2);
%         end
%     end
% tr_score_male=score(1:1056,1:20);
% tr_score_female=score(1057:2112,1:20);
% te_score_male=test_score(1:528,1:20);
% te_score_female=test_score(529:1056,1:20);



    %%
%     scatter(tr_score_male(:,1),tr_score_male(:,2),'r');
%     hold on;
%     scatter(tr_score_female(:,1),tr_score_female(:,2),'y');
%     hold on;
%     scatter(te_score_male(:,1),te_score_male(:,2),'g');
%     hold on;
%     scatter(te_score_female(:,1),te_score_female(:,2),'b');
    
    
function accuracy=calc_accuracy(arr,arr2)
c=0;
    for i=1:length(arr)
        if arr(i,1)==arr2(i,1)
            c=c+1;
        end
    end
    accuracy=(c/length(arr))*100;
end
