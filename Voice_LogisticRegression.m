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
for i=1:r
    if(voice_label(i,1)=="male")
        y_data(i,1)=1;
    else
        y_data(i,1)=0;
    end
end
XTrain(1:1056,:)=voice_data(1:1056,:);
XTrain(1057:2112,:)=voice_data(1585:2640,:);
XTest(1:528,:)=voice_data(1057:1584,:);
XTest(529:1056,:)=voice_data(2641:3168,:);
yTrain(1:1056,1)=y_data(1:1056,1);
yTrain(1057:2112,1)=y_data(1585:2640,1);
yTest(1:528,1)=y_data(1057:1584,1);
yTest(529:1056,1)=y_data(2641:3168,1);

[nsamples, nfeatures] = size(XTrain);
w0 = rand(nfeatures + 1, 1);
[weight,tmp] = logisticRegressionWeight( XTrain, yTrain, w0, 90, 0.01); %Maximum value for accuracy comes b/w 90-100 for MAX_ITR
res = logisticRegressionClassify( XTest, weight );

count=0;
for k=1:1056
    if(yTest(k,1)==res(k,1))
        count=count+1;
    end
end
accuracy_LR=100*(count/1056);



function [w,temp] = logisticRegressionWeight( XTrain, yTrain, w0, maxIter, learningRate)

    [nSamples, nFeature] = size(XTrain);
    w = w0;
    precost = 0;
    for j = 1:maxIter
        temp = zeros(nFeature + 1,1);
        for k = 1:nSamples
            temp = temp + (yTrain(k)-sigmoid([1.0 XTrain(k,:)] * w) ) * [1.0 XTrain(k,:)]';
        end
        w = w + learningRate * temp;
        cost = CostFunc(XTrain, yTrain, w);
        if j~=0 && abs(cost - precost) / cost <= 0.0001
            break;
        end
        precost = cost;
    end

end
function [ res ] = logisticRegressionClassify( XTest, w )

    nTest = size(XTest,1);
    res = zeros(nTest,1);
    for i = 1:nTest
        sigm = sigmoid([1.0 XTest(i,:)] * w);
        if sigm >= 0.5
            res(i) = 1;
        else
            res(i) = 0;
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

