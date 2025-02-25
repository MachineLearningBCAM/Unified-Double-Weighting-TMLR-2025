function [xTrain,yTrain,xTest,yTest,name] = generate_source(idx)
load('20Newsgroups_processed.mat')
if idx == 1
name = 'comp-vs-rec';
xTrain{1,1} = [xTrainData1{1};xTrainData2{1}];
yTrain{1,1} = [ones(size(xTrainData1{1},1),1);2*ones(size(xTrainData2{1},1),1)];
xTrain{2,1} = [xTrainData1{2};xTrainData2{2}];
yTrain{2,1} = [ones(size(xTrainData1{2},1),1);2*ones(size(xTrainData2{2},1),1)];
xTrain{3,1} = [xTrainData1{3};xTrainData2{3}];
yTrain{3,1} = [ones(size(xTrainData1{3},1),1);2*ones(size(xTrainData2{3},1),1)];
xTrain{4,1} = [xTrainData1{4};xTrainData2{4}];
yTrain{4,1} = [ones(size(xTrainData1{4},1),1);2*ones(size(xTrainData2{4},1),1)];
xTest{1,1} = [xTestData1{1};xTestData2{1}];
xTest{2,1} = [xTestData1{2};xTestData2{2}];
xTest{3,1} = [xTestData1{3};xTestData2{3}];
xTest{4,1} = [xTestData1{4};xTestData2{4}];
yTest{1,1} = [ones(size(xTestData1{1},1),1);2*ones(size(xTestData2{1},1),1);];
yTest{2,1} = [ones(size(xTestData1{2},1),1);2*ones(size(xTestData2{2},1),1)];
yTest{3,1} = [ones(size(xTestData1{3},1),1);2*ones(size(xTestData2{3},1),1)];
yTest{4,1} = [ones(size(xTestData1{4},1),1);2*ones(size(xTestData2{4},1),1)];
end
if idx == 2
name = 'comp-vs-sci';
xTrain{1,1} = [xTrainData1{1};xTrainData3{1}];
yTrain{1,1} = [ones(size(xTrainData1{1},1),1);2*ones(size(xTrainData3{1},1),1)];
xTrain{2,1} = [xTrainData1{2};xTrainData3{2}];
yTrain{2,1} = [ones(size(xTrainData1{2},1),1);2*ones(size(xTrainData3{2},1),1)];
xTrain{3,1} = [xTrainData1{3};xTrainData3{3}];
yTrain{3,1} = [ones(size(xTrainData1{3},1),1);2*ones(size(xTrainData3{3},1),1)];
xTrain{4,1} = [xTrainData1{4};xTrainData3{4}];
yTrain{4,1} = [ones(size(xTrainData1{4},1),1);2*ones(size(xTrainData3{4},1),1)];
xTest{1,1} = [xTestData1{1};xTestData3{1}];
xTest{2,1} = [xTestData1{2};xTestData3{2}];
xTest{3,1} = [xTestData1{3};xTestData3{3}];
xTest{4,1} = [xTestData1{4};xTestData3{4}];
yTest{1,1} = [ones(size(xTestData1{1},1),1);2*ones(size(xTestData3{1},1),1);];
yTest{2,1} = [ones(size(xTestData1{2},1),1);2*ones(size(xTestData3{2},1),1)];
yTest{3,1} = [ones(size(xTestData1{3},1),1);2*ones(size(xTestData3{3},1),1)];
yTest{4,1} = [ones(size(xTestData1{4},1),1);2*ones(size(xTestData3{4},1),1)];
end
if idx == 3
name = 'comp-vs-talk';
xTrain{1,1} = [xTrainData1{1};xTrainData4{1}];
yTrain{1,1} = [ones(size(xTrainData1{1},1),1);2*ones(size(xTrainData4{1},1),1)];
xTrain{2,1} = [xTrainData1{2};xTrainData4{2}];
yTrain{2,1} = [ones(size(xTrainData1{2},1),1);2*ones(size(xTrainData4{2},1),1)];
xTrain{3,1} = [xTrainData1{3};xTrainData4{3}];
yTrain{3,1} = [ones(size(xTrainData1{3},1),1);2*ones(size(xTrainData4{3},1),1)];
xTrain{4,1} = [xTrainData1{4};xTrainData4{4}];
yTrain{4,1} = [ones(size(xTrainData1{4},1),1);2*ones(size(xTrainData4{4},1),1)];
xTest{1,1} = [xTestData1{1};xTestData4{1}];
xTest{2,1} = [xTestData1{2};xTestData4{2}];
xTest{3,1} = [xTestData1{3};xTestData4{3}];
xTest{4,1} = [xTestData1{4};xTestData4{4}];
yTest{1,1} = [ones(size(xTestData1{1},1),1);2*ones(size(xTestData4{1},1),1);];
yTest{2,1} = [ones(size(xTestData1{2},1),1);2*ones(size(xTestData4{2},1),1)];
yTest{3,1} = [ones(size(xTestData1{3},1),1);2*ones(size(xTestData4{3},1),1)];
yTest{4,1} = [ones(size(xTestData1{4},1),1);2*ones(size(xTestData4{4},1),1)];
end
if idx == 4
name = 'rec-vs-sci';
xTrain{1,1} = [xTrainData2{1};xTrainData3{1}];
yTrain{1,1} = [ones(size(xTrainData2{1},1),1);2*ones(size(xTrainData3{1},1),1)];
xTrain{2,1} = [xTrainData2{2};xTrainData3{2}];
yTrain{2,1} = [ones(size(xTrainData2{2},1),1);2*ones(size(xTrainData3{2},1),1)];
xTrain{3,1} = [xTrainData2{3};xTrainData3{3}];
yTrain{3,1} = [ones(size(xTrainData2{3},1),1);2*ones(size(xTrainData3{3},1),1)];
xTrain{4,1} = [xTrainData2{4};xTrainData3{4}];
yTrain{4,1} = [ones(size(xTrainData2{4},1),1);2*ones(size(xTrainData3{4},1),1)];
xTest{1,1} = [xTestData2{1};xTestData3{1}];
xTest{2,1} = [xTestData2{2};xTestData3{2}];
xTest{3,1} = [xTestData2{3};xTestData3{3}];
xTest{4,1} = [xTestData2{4};xTestData3{4}];
yTest{1,1} = [ones(size(xTestData2{1},1),1);2*ones(size(xTestData3{1},1),1);];
yTest{2,1} = [ones(size(xTestData2{2},1),1);2*ones(size(xTestData3{2},1),1)];
yTest{3,1} = [ones(size(xTestData2{3},1),1);2*ones(size(xTestData3{3},1),1)];
yTest{4,1} = [ones(size(xTestData2{4},1),1);2*ones(size(xTestData3{4},1),1)];
end
if idx == 5
name = 'rec-vs-talk';
xTrain{1,1} = [xTrainData2{1};xTrainData4{1}];
yTrain{1,1} = [ones(size(xTrainData2{1},1),1);2*ones(size(xTrainData4{1},1),1)];
xTrain{2,1} = [xTrainData2{2};xTrainData4{2}];
yTrain{2,1} = [ones(size(xTrainData2{2},1),1);2*ones(size(xTrainData4{2},1),1)];
xTrain{3,1} = [xTrainData2{3};xTrainData4{3}];
yTrain{3,1} = [ones(size(xTrainData2{3},1),1);2*ones(size(xTrainData4{3},1),1)];
xTrain{4,1} = [xTrainData2{4};xTrainData4{4}];
yTrain{4,1} = [ones(size(xTrainData2{4},1),1);2*ones(size(xTrainData4{4},1),1)];
xTest{1,1} = [xTestData2{1};xTestData4{1}];
xTest{2,1} = [xTestData2{2};xTestData4{2}];
xTest{3,1} = [xTestData2{3};xTestData4{3}];
xTest{4,1} = [xTestData2{4};xTestData4{4}];
yTest{1,1} = [ones(size(xTestData2{1},1),1);2*ones(size(xTestData4{1},1),1);];
yTest{2,1} = [ones(size(xTestData2{2},1),1);2*ones(size(xTestData4{2},1),1)];
yTest{3,1} = [ones(size(xTestData2{3},1),1);2*ones(size(xTestData4{3},1),1)];
yTest{4,1} = [ones(size(xTestData2{4},1),1);2*ones(size(xTestData4{4},1),1)];
end
if idx == 6
name = 'sci-vs-talk';
xTrain{1,1} = [xTrainData3{1};xTrainData4{1}];
yTrain{1,1} = [ones(size(xTrainData3{1},1),1);2*ones(size(xTrainData4{1},1),1)];
xTrain{2,1} = [xTrainData3{2};xTrainData4{2}];
yTrain{2,1} = [ones(size(xTrainData3{2},1),1);2*ones(size(xTrainData4{2},1),1)];
xTrain{3,1} = [xTrainData3{3};xTrainData4{3}];
yTrain{3,1} = [ones(size(xTrainData3{3},1),1);2*ones(size(xTrainData4{3},1),1)];
xTrain{4,1} = [xTrainData3{4};xTrainData4{4}];
yTrain{4,1} = [ones(size(xTrainData3{4},1),1);2*ones(size(xTrainData4{4},1),1)];
xTest{1,1} = [xTestData3{1};xTestData4{1}];
xTest{2,1} = [xTestData3{2};xTestData4{2}];
xTest{3,1} = [xTestData3{3};xTestData4{3}];
xTest{4,1} = [xTestData3{4};xTestData4{4}];
yTest{1,1} = [ones(size(xTestData3{1},1),1);2*ones(size(xTestData4{1},1),1);];
yTest{2,1} = [ones(size(xTestData3{2},1),1);2*ones(size(xTestData4{2},1),1)];
yTest{3,1} = [ones(size(xTestData3{3},1),1);2*ones(size(xTestData4{3},1),1)];
yTest{4,1} = [ones(size(xTestData3{4},1),1);2*ones(size(xTestData4{4},1),1)];
end


end