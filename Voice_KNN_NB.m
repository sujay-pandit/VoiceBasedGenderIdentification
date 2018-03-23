%% Import Data
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

%% Dividing data into Experimental Data and Test Data
% Class-1 for Male Class-2 for Female
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

trainData_b=bin(trainData);
testData_b=bin(testData);

%% Naive Bayes: Using In-built functions
Hypothesis_func_nb = fitcnb(trainData_b(:,1:20),trainData_b(:,21));%NB Classifier

for i=1:1056
predicted_class=predict(Hypothesis_func_nb,testData_b(i,1:20));
testData_b(i,22)=predicted_class;
end
accuracy_fitcnb=calc_accuracy(testData_b(:,21),testData_b(:,22));
%% KNN: Using In-Built functions
Hypothesis_func_knn = fitcknn(trainData_b(:,1:20),trainData_b(:,21),'NumNeighbors',5);%KNN Classifier
for i=1:1056
predicted_class=predict(Hypothesis_func_knn,testData_b(i,1:20));
testData_b(i,22)=predicted_class;
end
accuracy_fitcknn=calc_accuracy(testData_b(:,21),testData_b(:,22));
%% Naive Bayes: Self Defined 
% elementcount counts frequency of every element in every class in the
% feature
% calc_prob calculates probability of a value of a feature in a particular
% class
for i=1:c
x(:,:,i)=calc_prob(elementcount(trainData_b(:,i),trainData_b(:,21))); 
end

% testing
final_prob=ones(length(testData),2);
for j=1:length(testData)
   % [final_prob,predicted_class]=test_model(x,testData(i,1:20)');
    for class=1:2
        for k=1:20
            for i=1:length(x(:,:,k))
                if(x(i,1,k)==testData_b(j,k))
                    final_prob(j,class)=final_prob(j,class)*x(i,class+1,k);
                end
            end
        end
    end
    [prob,predicted_class] = max(final_prob(j,:));
    testData_b(j,22)=predicted_class;
    testData_b(j,23)=final_prob(j,1);
    testData_b(j,24)=final_prob(j,2);
end
accuracy_mynb=calc_accuracy(testData_b(:,21),testData_b(:,22));
%% KNN Self defined
for i=1:length(testData_b)
    for j=1:length(trainData_b)
            d(i,j)=calc_distance(testData_b(i,1:20),trainData_b(j,1:20));
    end
end
for k=1:5
    for i=1:length(testData_b)
           [data_val,data_index] = min(d(i,:)');
           d(i,data_index)=100000000;
           k_nearest(i,k)=trainData_b(data_index,21);
    end
end
for i=1:length(k_nearest)
    [u,prob,predicted_class]=knn_class(k_nearest(i,:));
    testData_b(i,25)=predicted_class;
    testData_b(i,26)=prob;
end
accuracy_myknn=calc_accuracy(testData_b(:,21),testData_b(:,25));

%% USER-DEFINED Functions 
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
        if(D(i,j)<=0.125)
            data(i,j)=0.125;
        elseif(D(i,j)>0.125 && D(i,j)<=0.25)
            data(i,j)=0.25;
        elseif(D(i,j)>0.25 && D(i,j)<=0.375)
            data(i,j)=0.375;
        elseif(D(i,j)>0.375 && D(i,j)<=0.5)
            data(i,j)=0.5;
        elseif(D(i,j)>0.5 && D(i,j)<=0.625)
            data(i,j)=0.625;
        elseif(D(i,j)>0.625 && D(i,j)<=0.75)
            data(i,j)=0.75;
        elseif(D(i,j)>0.75 && D(i,j)<=0.875)
            data(i,j)=0.875;
        elseif(D(i,j)>0.875)
            data(i,j)=1;
        end
    end
end
data(:,c)=D(:,c);
end
function count=elementcount(arr,arr2)
bins=8;
count=zeros(bins,3);
count(:,1)=0.125:1/8:1;
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

