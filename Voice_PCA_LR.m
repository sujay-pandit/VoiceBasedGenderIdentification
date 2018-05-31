%% Import Data
clear all;
clc;
run('loadData2.m');
run('loadLabels.m');
%feature_list=[1,6,7,10,13,14,17,20];
feature_list=[1:20];
idx=1;
for i=1:length(feature_list)
    voice_data(:,idx)=voice_data_o(:,feature_list(1,i));
    idx=idx+1;
end

%% Scaling data using min-max method
[r,c]=size(voice_data);

for j=1:c
    mn=min(voice_data(:,j));
    men=mean(voice_data(:,j));
    mx=max(voice_data(:,j));
    for i=1:r
        voice_data(i,j)=(voice_data(i,j)-men)/(mx-mn);
    end
end
for i=1:r
    if(voice_label(i,1)=="male")
        voice_data(i,c+1)=1;
    else
        voice_data(i,c+1)=0;
    end
end
% 1056:2112 CROSS VALIDATION DATA
XTrain(1:1056,:,1)=voice_data(1:1056,1:c);
XTrain(1057:2112,:,1)=voice_data(1585:2640,1:c);
XTest(1:528,:,1)=voice_data(1057:1584,1:c);
XTest(529:1056,:,1)=voice_data(2641:3168,1:c);
yTrain(1:1056,:,1)=voice_data(1:1056,c+1);
yTrain(1057:2112,:,1)=voice_data(1585:2640,c+1);
yTest(1:528,:,1)=voice_data(1057:1584,c+1);
yTest(529:1056,:,1)=voice_data(2641:3168,c+1);

XTrain(1:1056,:,2)=voice_data(1585:2640,1:c);
XTrain(1057:2112,:,2)=voice_data(1:1056,1:c);
XTest(1:528,:,2)=voice_data(2641:3168,1:c);
XTest(529:1056,:,2)=voice_data(1057:1584,1:c);
yTrain(1:1056,:,2)=voice_data(1585:2640,c+1);
yTrain(1057:2112,:,2)=voice_data(1:1056,c+1);
yTest(1:528,:,2)=voice_data(2641:3168,c+1);
yTest(529:1056,:,2)=voice_data(1057:1584,c+1);

XTrain(1:1056,:,3)=voice_data(529:1584,1:c);
XTrain(1057:2112,:,3)=voice_data(2113:3168,1:c);
XTest(1:528,:,3)=voice_data(1:528,1:c);
XTest(529:1056,:,3)=voice_data(1585:2112,1:c);
yTrain(1:1056,:,3)=voice_data(529:1584,c+1);
yTrain(1057:2112,:,3)=voice_data(2113:3168,c+1);
yTest(1:528,:,3)=voice_data(1:528,c+1);
yTest(529:1056,:,3)=voice_data(1585:2112,c+1);


XTrain(1:1056,:,4)=voice_data(2113:3168,1:c);
XTrain(1057:2112,:,4)=voice_data(529:1584,1:c);
XTest(1:528,:,4)=voice_data(1585:2112,1:c);
XTest(529:1056,:,4)=voice_data(1:528,1:c);
yTrain(1:1056,:,4)=voice_data(2113:3168,c+1);
yTrain(1057:2112,:,4)=voice_data(529:1584,c+1);
yTest(1:528,:,4)=voice_data(1585:2112,c+1);
yTest(529:1056,:,4)=voice_data(1:528,c+1);

XTrain(1:528,:,5)=voice_data(1:528,1:c);
XTrain(529:1056,:,5)=voice_data(1057:1584,1:c);
XTrain(1057:1584,:,5)=voice_data(1585:2112,1:c);
XTrain(1585:2112,:,5)=voice_data(2641:3168,1:c);
XTest(1:528,:,5)=voice_data(529:1056,1:c);
XTest(529:1056,:,5)=voice_data(2113:2640,1:c);
yTrain(1:528,:,5)=voice_data(1:528,c+1);
yTrain(529:1056,:,5)=voice_data(1057:1584,c+1);
yTrain(1057:1584,:,5)=voice_data(1585:2112,c+1);
yTrain(1585:2112,:,5)=voice_data(2641:3168,c+1);
yTest(1:528,:,5)=voice_data(529:1056,c+1);
yTest(529:1056,:,5)=voice_data(2113:2640,c+1);

XTrain(1:528,:,6)=voice_data(1057:1584,1:c);
XTrain(529:1056,:,6)=voice_data(1:528,1:c);
XTrain(1057:1584,:,6)=voice_data(2641:3168,1:c);
XTrain(1585:2112,:,6)=voice_data(1585:2112,1:c);
XTest(1:528,:,6)=voice_data(529:1056,1:c);
XTest(529:1056,:,6)=voice_data(2113:2640,1:c);
yTrain(1:528,:,6)=voice_data(1057:1584,c+1);
yTrain(529:1056,:,6)=voice_data(1:528,c+1);
yTrain(1057:1584,:,6)=voice_data(2641:3168,c+1);
yTrain(1585:2112,:,6)=voice_data(1585:2112,c+1);
yTest(1:528,:,6)=voice_data(529:1056,c+1);
yTest(529:1056,:,6)=voice_data(2113:2640,c+1);


XTrain(1:528,:,7)=voice_data(2641:3168,1:c);
XTrain(529:1056,:,7)=voice_data(1:528,1:c);
XTrain(1057:1584,:,7)=voice_data(1057:1584,1:c);
XTrain(1585:2112,:,7)=voice_data(1585:2112,1:c);
XTest(1:528,:,7)=voice_data(2113:2640,1:c);
XTest(529:1056,:,7)=voice_data(529:1056,1:c);
yTrain(1:528,:,7)=voice_data(2641:3168,c+1);
yTrain(529:1056,:,7)=voice_data(1:528,c+1);
yTrain(1057:1584,:,7)=voice_data(1057:1584,c+1);
yTrain(1585:2112,:,7)=voice_data(1585:2112,c+1);
yTest(1:528,:,7)=voice_data(2113:2640,c+1);
yTest(529:1056,:,7)=voice_data(529:1056,c+1);

XTrain(1:528,:,8)=voice_data(529:1056,1:c);
XTrain(529:1056,:,8)=voice_data(1585:2112,1:c);
XTrain(1057:1584,:,8)=voice_data(2641:3168,1:c);
XTrain(1585:2112,:,8)=voice_data(1:528,1:c);
XTest(1:528,:,8)=voice_data(2113:2640,1:c);
XTest(529:1056,:,8)=voice_data(1057:1584,1:c);
yTrain(1:528,:,8)=voice_data(529:1056,c+1);
yTrain(529:1056,:,8)=voice_data(1585:2112,c+1);
yTrain(1057:1584,:,8)=voice_data(2641:3168,c+1);
yTrain(1585:2112,:,8)=voice_data(1:528,c+1);
yTest(1:528,:,8)=voice_data(2113:2640,c+1);
yTest(529:1056,:,8)=voice_data(1057:1584,c+1);

XTrain(1:528,:,9)=voice_data(2641:3168,1:c);
XTrain(529:1056,:,9)=voice_data(2113:2640,1:c);
XTrain(1057:1584,:,9)=voice_data(1057:1584,1:c);
XTrain(1585:2112,:,9)=voice_data(1:528,1:c);
XTest(1:528,:,9)=voice_data(1585:2112,1:c);
XTest(529:1056,:,9)=voice_data(529:1056,1:c);
yTrain(1:528,:,9)=voice_data(2641:3168,1+c);
yTrain(529:1056,:,9)=voice_data(2113:2640,1+c);
yTrain(1057:1584,:,9)=voice_data(1057:1584,1+c);
yTrain(1585:2112,:,9)=voice_data(1:528,1+c);
yTest(1:528,:,9)=voice_data(1585:2112,1+c);
yTest(529:1056,:,9)=voice_data(529:1056,1+c);

XTrain(1:528,:,10)=voice_data(1585:2112,1:c);
XTrain(529:1056,:,10)=voice_data(2113:2640,1:c);
XTrain(1057:1584,:,10)=voice_data(529:1056,1:c);
XTrain(1585:2112,:,10)=voice_data(1057:1584,1:c);
XTest(1:528,:,10)=voice_data(2641:3168,1:c);
XTest(529:1056,:,10)=voice_data(1:528,1:c);
yTrain(1:528,:,10)=voice_data(1585:2112,c+1);
yTrain(529:1056,:,10)=voice_data(2113:2640,c+1);
yTrain(1057:1584,:,10)=voice_data(529:1056,c+1);
yTrain(1585:2112,:,10)=voice_data(1057:1584,c+1);
yTest(1:528,:,10)=voice_data(2641:3168,c+1);
yTest(529:1056,:,10)=voice_data(1:528,c+1);

%% Logistic Regression Accuracy

for j=1:10
        v=cov(XTrain(:,1:20,1));
        [V,U]=eig(v);

        [coeff,score,latent,tsquared,explained(:,j)]=pca(XTrain(:,1:20,j));
        for f=1:20
        %scre=XTrain(:,1:f,j)*coeff(:,1:f);
        
        test_score=XTest(:,:,j)*coeff(:,1:f);
        
        [accuracy_test_LR(1,f,j)]=LogisticRegression(score(:,1:f),test_score(:,1:f),yTrain(:,:,j),yTest(:,:,j));

        end
end
%%  MEAN ACCURACY
for i=1:20%length(feature_list)
         accuracy_LR(i,1)=mean(accuracy_test_LR(1,i,:));
end

%% VARIANCE PLOT
for i=1:20
    average_var(i,1)=mean(explained(i,:));
end
stem([1:20],average_var);
%% Feature-Wise Accuracy
% for i=1:10
%      for j=1:c
%          [accuracy_test_LR(1,j,i)]=LogisticRegression(XTrain(:,j,i),XTest(:,j,i),yTrain(:,:,i),yTest(:,:,i));
%      end
% end
% for i=1:c
%          accuracy_features(1,i,1)=mean(accuracy_test_LR(1,i,1:10));
% end

%% USER DEFINED FUNCTIONS
function [accuracy_test_LR]=LogisticRegression(XTrain,XTest,yTrain,yTest)
    [nsamples, nfeatures] = size(XTrain);
    w0 = rand(nfeatures + 1, 1);
    [iteration,weight,tmp] = logisticRegressionWeight( XTrain, yTrain, w0, 100, 0.01); %Maximum value for accuracy comes b/w 90-100 for MAX_ITR
    predicted_class = logisticRegressionClassify( XTrain, weight );
    count=0;
    for k=1:2112
        if(yTrain(k,1)==predicted_class(k,1))
            count=count+1;
        end
    end
    accuracy_train_LR=100*(count/2112);
    predicted_class = logisticRegressionClassify( XTest, weight );

    count=0;
    for k=1:1056
        if(yTest(k,1)==predicted_class(k,1))
            count=count+1;
        end
    end
    accuracy_test_LR(1)=100*(count/1056);
end



function [iteration,w,temp] = logisticRegressionWeight( XTrain, yTrain, w0, maxIter, learningRate)

    [nSamples, nFeature] = size(XTrain);
    w = w0;
    precost = 0;
    iteration=0;
    for j = 1:maxIter
        temp = zeros(nFeature + 1,1);
        for k = 1:nSamples
            temp = temp + (yTrain(k)-sigmoid([1.0 XTrain(k,:)] * w) ) * [1.0 XTrain(k,:)]';
        end
        w = w + learningRate * temp;
        cost = CostFunc(XTrain, yTrain, w);
        if j~=0 && abs(cost - precost)<= 0.0001
            break;
        end
        precost = cost;
        iteration=iteration+1;
    end

end
function [ predict ] = logisticRegressionClassify( XTest, w )

    nTest = size(XTest,1);
    predict = zeros(nTest,1);
    for i = 1:nTest
        sigm = sigmoid([1.0 XTest(i,:)] * w);
        if sigm >= 0.5
            predict(i) = 1;
        else
            predict(i) = 0;
        end
    end

end
function [ output ] = sigmoid( input )

    %output = tanh(input);
    output = 1 / (1 + exp(- input));

end

function [ J ] = CostFunc( XTrain, yTrain, w )

    [nSamples, nFeature] = size(XTrain);
    temp = 0.0;
    for m = 1:nSamples
        hx = sigmoid([1.0 XTrain(m,:)] * w);
        if yTrain(m) == 1
            temp = temp + log(hx);
        else
            temp = temp + log(1 - hx);
        end
    end
    J = temp / (-nSamples);

end

