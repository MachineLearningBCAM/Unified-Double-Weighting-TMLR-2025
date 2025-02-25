function [xTr,yTr,xTe,yTe] = generate_train_test(xTrain,yTrain,xTest,yTest)
major = 100;
minor1 = 50;
minor2 = 50;
a = randsample(4,4);
for i = 1:length(a)-1
    idx_tr = randsample(size(xTrain{a(i),1},1),major);
    xTr{i,1} = xTrain{a(i),1}(idx_tr,:);
    yTr{i,1} = yTrain{a(i),1}(idx_tr);
    if i == 3
        idx_tr = randsample(size(xTrain{a(1),1},1),minor1);
        xTr{i,1} = [xTr{i,1};xTrain{a(1),1}(idx_tr,:)];
        yTr{i,1} = [yTr{i,1};yTrain{a(1),1}(idx_tr)];
        idx_tr = randsample(size(xTrain{a(3),1},1),minor2);
        xTr{i,1} = [xTr{i,1};xTrain{a(3),1}(idx_tr,:)];
        yTr{i,1} = [yTr{i,1};yTrain{a(3),1}(idx_tr)];
    end
    if i == 2
        idx_tr = randsample(size(xTrain{a(3),1},1),minor1);
        xTr{i,1} = [xTr{i,1};xTrain{a(3),1}(idx_tr,:)];
        yTr{i,1} = [yTr{i,1};yTrain{a(3),1}(idx_tr)];
        idx_tr = randsample(size(xTrain{a(1),1},1),minor2);
        xTr{i,1} = [xTr{i,1};xTrain{a(1),1}(idx_tr,:)];
        yTr{i,1} = [yTr{i,1};yTrain{a(1),1}(idx_tr)];
    end
    if i == 1
        idx_tr = randsample(size(xTrain{a(2),1},1),minor1);
        xTr{i,1} = [xTr{i,1};xTrain{a(2),1}(idx_tr,:)];
        yTr{i,1} = [yTr{i,1};yTrain{a(2),1}(idx_tr)];
        idx_tr = randsample(size(xTrain{a(3),1},1),minor2);
        xTr{i,1} = [xTr{i,1};xTrain{a(3),1}(idx_tr,:)];
        yTr{i,1} = [yTr{i,1};yTrain{a(3),1}(idx_tr)];
    end
end
xTe = [];
yTe = [];
for i = 1:length(a)-1
    idx_te = randsample(size(xTest{a(i),1},1),40);
    xTe = [xTe;xTest{a(i)}(idx_te,:)];
    yTe = [yTe;yTest{a(i)}(idx_te)];
end
idx_te = randsample(size(xTest{a(end),1},1),30);
xTe = [xTe;xTest{a(end),1}(idx_te,:)];
yTe = [yTe;yTest{a(end),1}(idx_te)];
end