clear all;
%% Import Data
run('loadData1.m');
run('loadLabels.m');

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
% for i=1:20
%     plot(voice_data(1:1584,i),'r');
%     hold on
%     plot(voice_data(1585:3168,i),'b')
%     figure
% end
%% Cross Validation Accuracy
for i=1:10
    [a_mynb(1,i),a_fitcknn3(1,i),a_fitcknn5(1,i),predicted_classnb(:,i),predicted_class5(:,i),predicted_class3(:,i)]=NB_KNN(trainData(:,:,i),testData(:,:,i),[1,6,13,15]);
end
    accuracy_mynb=mean(a_mynb);
    accuracy_fitcknn3=mean(a_fitcknn3);
    accuracy_fitcknn5=mean(a_fitcknn5);
%%


%% Feature Wise
% for j=1:20
%     for i=1:10
%     [a_mynb(1,i),a_fitcknn(1,i)]=NB_KNN(trainData(:,:,i),testData(:,:,i),j);
%     end
%     accuracy_mynb(1,j)=mean(a_mynb);
%     accuracy_fitcknn(1,j)=mean(a_fitcknn);
% end
%% FUNCTIONs
function [accuracy_mynb,accuracy_fitcknn3,accuracy_fitcknn5,predicted_classnb,predicted_class5,predicted_class3]=NB_KNN(trainData,testData,featurelist);
trainData_b=bin(trainData);
testData_b=bin(testData);
idx=1;
for i=1:length(featurelist)
    trData_b(:,idx)=trainData_b(:,featurelist(1,i));
    teData_b(:,idx)=testData_b(:,featurelist(1,i));
    trData(:,idx)=trainData(:,featurelist(1,i));
    teData(:,idx)=testData(:,featurelist(1,i));
    idx=idx+1;
end

%% KNN: Using In-Built functions
Hypothesis_func_knn = fitcknn(trData,trainData(:,21),'NumNeighbors',5);%KNN Classifier
for i=1:1056
predicted_class=predict(Hypothesis_func_knn,teData(i,:));
testData(i,22)=predicted_class;
predicted_class5(i,1)=predicted_class;
end
accuracy_fitcknn5=calc_accuracy(testData(:,21),testData(:,22));
Hypothesis_func_knn = fitcknn(trData,trainData(:,21),'NumNeighbors',3);%KNN Classifier
for i=1:1056
predicted_class=predict(Hypothesis_func_knn,teData(i,:));
testData(i,22)=predicted_class;
predicted_class3(i,1)=predicted_class;
end
accuracy_fitcknn3=calc_accuracy(testData(:,21),testData(:,22));
%% Naive Bayes: Self Defined 
% elementcount counts frequency of every element in every class in the
% feature
% calc_prob calculates probability of a value of a feature in a particular
% class
for i=1:length(featurelist)
x(:,:,i)=calc_prob(elementcount(trData_b(:,i),trainData_b(:,21))); 
end

% testing
final_prob=ones(length(testData),2);
for j=1:length(testData)
   % [final_prob,predicted_class]=test_model(x,testData(i,1:20)');
    for class=1:2
        for k=1:length(featurelist)
            for i=1:length(x(:,:,k))
                if(x(i,1,k)==teData_b(j,k))
                    final_prob(j,class)=final_prob(j,class)*x(i,class+1,k);
                end
            end
        end
    end
    [prob,predicted_class] = max(final_prob(j,:));
    testData_b(j,22)=predicted_class;
    testData_b(j,23)=final_prob(j,1);
    testData_b(j,24)=final_prob(j,2);
    predicted_classnb(j,1)=predicted_class;
end
accuracy_mynb=calc_accuracy(testData_b(:,21),testData_b(:,22));
end

function accuracy=calc_accuracy(arr,arr2)
c=0;
    for i=1:length(arr)
        if arr(i,1)==arr2(i,1)
            c=c+1;
        end
    end
    accuracy=(c/length(arr))*100;
end

function data=bin(D) % Into 8 bins 
[r,c]=size(D);
for j=1:c-1
    for i=1:r
        if(D(i,j)<=0.25)
            data(i,j)=0.25;
        elseif(D(i,j)>0.25 && D(i,j)<=0.5)
            data(i,j)=0.5;
        elseif(D(i,j)>0.5 && D(i,j)<=0.75)
            data(i,j)=0.75;
        elseif(D(i,j)>0.75 && D(i,j)<=1)
            data(i,j)=1;
        end
%         if(D(i,j)<=0.125)
%             data(i,j)=0.125;
%         elseif(D(i,j)>0.125 && D(i,j)<=0.25)
%             data(i,j)=0.25;
%         elseif(D(i,j)>0.25 && D(i,j)<=0.375)
%             data(i,j)=0.375;
%         elseif(D(i,j)>0.375 && D(i,j)<=0.5)
%             data(i,j)=0.5;
%         elseif(D(i,j)>0.5 && D(i,j)<=0.625)
%             data(i,j)=0.625;
%         elseif(D(i,j)>0.625 && D(i,j)<=0.75)
%             data(i,j)=0.75;
%         elseif(D(i,j)>0.75 && D(i,j)<=0.875)
%             data(i,j)=0.875;
%         elseif(D(i,j)>0.875)
%             data(i,j)=1;
%         end
    end
end
data(:,c)=D(:,c);
end
function count=elementcount(arr,arr2)
bins=4;
count=zeros(bins,3);
% count(:,1)=0.125:1/8:1;
count(:,1)=0.25:1/4:1;
[r,c]=size(count);
for class=1:2
    for j=1:r 
     for i=1:length(arr)
            if arr(i,1)==count(j,1) && arr2(i,1)==class
            count(j,class+1)=count(j,class+1)+1;
            end
     end
    end
end
end
function tmp=calc_prob(x)
[r,c]=size(x);
tmp=zeros(r,3);
tmp(:,1)=x(:,1);
for j=1:2
    for i=1:r
     tmp(i,j+1)=x(i,j+1)/sum(x(:,j+1));
    end
end
end
function [final_prob,predicted_class]=test_model(x,tData)
final_prob=ones(1,2);
    for class=1:2
        for k=1:20
            for i=1:length(x(:,:,k))
                if(x(i,1,k)==tData(k,1))
                    final_prob(1,class)=final_prob(1,class)*x(i,class+1,k);
                end
            end
        end
    end
[prob,predicted_class] = max(final_prob(:));
end
function d=calc_distance(test_pt,data_pt)
sq_d=0;
    for i=1:4
        sq_d=sq_d+(test_pt(1,i)-data_pt(1,i))^2;
    end
    d=sqrt(sq_d);
end
function [unq,prob,predicted_class]=knn_class(x)
x=x';
unq=zeros(length(unique(x)),3);
unq(:,1)=unique(x);
[r,c]=size(unique(x));
    for i=1:r
        for j=1:length(x)
            if(x(j,1)==unq(i,1))
                unq(i,2)=unq(i,2)+1;
            end
        end
    end
    for i=1:length(unique(x))
        unq(i,3)=unq(i,2)/sum(unq(:,2));
    end
    [prob,index]=max(unq(:,3));
    predicted_class=unq(index,1);
end