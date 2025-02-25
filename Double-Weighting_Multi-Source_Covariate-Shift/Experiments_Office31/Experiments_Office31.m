function Experiments_Office31(idx1)

% Clear all

clearvars -except idx1

% Add paths to CVX

addpath('../cvx')
cvx_setup
cvx_solver mosek
cvx_save_prefs

% Add paths to existing and proposed methods

addpath(genpath('../Datasets'));

addpath('../Functions_Library_MSDW/LR')
addpath('../Functions_Library_MSDW/KMM')
addpath('../Functions_Library_MSDW/DWGCS')
addpath('../Functions_Library_MSDW/2SWMDA')
addpath('../Functions_Library_MSDW/MSDRL')
addpath('../Functions_Library_MSDW/CWKMM')
addpath('../Functions_Library_MSDW/DWMSCS')
addpath('../Functions_Library_MSDW/Auxiliar')

%%

% Initialize the base model

Mdl.intercept = false;
Mdl.fmap = 'linear';
Mdl.loss = 'log';
Mdl.deterministic = true;
Mdl.lambda0 = 0;
Mdl.labels = 5;
Mdl.distribs = 2;

%%

% Generate the initial sources and obtain the 500 * idx1 features
% with highest Pearson's correlation, and normalize the dataset
files = {'Office31_5labels_electronics', 'Office31_5labels_stationery', ...
    'Office31_5labels_organization', 'Office31_5labels_mixed'};
load(strcat(files{idx1},'.mat'));


X = cell2mat(Domain);
X = zscore(X(:,1:end-1));
Y = cell2mat(Domain);
Y = Y(:,end);

top_features = PCC(X, Y, 100);
X = X(:, top_features);

[~, distance] = knnsearch(X, X, 'K', 50);
sigma        = mean(distance(:, 50));
Mdl.sigma = sigma;
XY = [X,Y];

% Compute sigma for the kernel matrix using 50-Nearest Neighbors
for i = 1 : Mdl.distribs + 1
    n(i) = size(Domain{i}, 1);
    if i == 1
        Domain{i, 1} = XY(1:n(i), :);
    else
        Domain{i, 1} = XY(sum(n(1:i - 1)) + 1:sum(n(1:i)), :);
    end
end

sizes = [47 30 34 37];

for rep = 1:20

    % Generate the training sources and testing set for each repetition
    xTr = []; xTest = []; yTr = []; yTest = [];
    for i = 1 : Mdl.distribs + 1
        data = Domain{i,1};
        cv = cvpartition(size(data, 1), 'HoldOut', 0.5);
        train_idx = cv.training;
        test_idx = cv.test;
        xTr{i,1} = data(train_idx, 1:end-1);
        yTr{i,1} = data(train_idx, end);
        xTest{i,1} = data(test_idx, 1:end-1);
        yTest{i,1} = data(test_idx, end);
    end
    xTr(2)=[];
    yTr(2)=[];
    xTe = [];
    yTe = [];
    for i = 1:3
        pos = randperm(length(yTest{i}), sizes(idx0));
        xTe = [xTe;xTest{i}(pos,:)];
        yTe = [yTe;yTest{i}(pos)];
    end

    %% Compute the classification error using ERM LR 

    for d = 1:Mdl.distribs
            LR = Mdl;
            LR.lambda0 = 0;
            LR = LR_parameters(LR, xTr{d}, yTr{d});
            LR = LR_learning(LR, xTr{d});
            LR = LR_prediction(LR, xTe, yTe);
            ErrorLR(rep, d) = LR.error;
    end

    %% Compute the classification error using KMM

        for d = 1:Mdl.distribs
            Kmm = Mdl;
            Kmm.lambda0 = 0;
            Kmm = KMM(Kmm, xTr{d}, xTe);
            Kmm = IWMRC_parameters(Kmm, xTr{d}, yTr{d});
            Kmm = IWMRC_learning(Kmm, xTr{d});
            Kmm = MRC_prediction(Kmm, xTe, yTe);
            ErrorKmm(rep,d) = Kmm.error;
        end

    %% Compute the classification error using DW-GCS

    for d = 1:Mdl.distribs
            D = 1 ./ (1 - (0:0.1:0.9)) .^ 2;
            DWGCS = [];
            for l = 1:length(D)
                DWGCS{l}      = Mdl;
                DWGCS{l}.loss = '0-1';
                DWGCS{l}.D    = D(l);
                DWGCS{l}.B    = 1000;
                DWGCS{l}      = DWKMM(DWGCS{l}, xTr{d}, xTe);
                DWGCS{l}      = DWMRC_parameters(DWGCS{l}, xTr{d}, yTr{d}, xTe);
                DWGCS{l}      = DWMRC_learning(DWGCS{l}, xTe);
                RU(l) = DWGCS{l}.min_MRC;
            end
            [~, position] = min(RU);
            DWGCS = DWGCS{position};
            DWGCS = DWMRC_prediction(DWGCS, xTe, yTe);
            ErrorDWGCS(rep, d) = DWGCS.error;
    end

    %% Compute the classification error using 2SW-MDA

    SWMDA      = Mdl;
        SWMDA.lambda0 = 0;
        SWMDA.H    = zeros(Mdl.distribs, size(xTe, 1));
        for d = 1:Mdl.distribs
            SWMDAd                            = Mdl;
            SWMDAd                            = KMM(SWMDAd, xTr{d}, xTe);
            beta_aux{d,1}                     = SWMDAd.beta;
            SWMDAd                            = IWMRC_parameters(SWMDAd, xTr{d}, yTr{d});
            SWMDAd                            = IWMRC_learning(SWMDAd, xTr{d});
            SWMDAd                            = MRC_prediction(SWMDAd, xTe, yTe);
            SWMDAd.y_pred(SWMDAd.y_pred == 2) = -1;
            SWMDA.H(d,:)                      = SWMDAd.y_pred;
        end
        SWMDA.beta = beta_aux;
        SWMDA      = SWMDA_2weights(SWMDA,xTe);
        for d = 1:Mdl.distribs
            SWMDA.beta{d, 1} = (size(cell2mat(xTr), 1) * SWMDA.w(d) / size(xTr{d}, 1)) * SWMDA.beta{d};
        end
        SWMDA.beta         = cell2mat(SWMDA.beta);
        SWMDA              = IWMRC_parameters(SWMDA, cell2mat(xTr), cell2mat(yTr));
        SWMDA              = IWMRC_learning(SWMDA, cell2mat(xTr));
        SWMDA              = MRC_prediction(SWMDA, xTe, yTe);
        Error2SWMDA(rep,1) = SWMDA.error;

    %% Compute the classification error using MS-DRL

        ErrorMSDRL(rep,1) = MSDRL_learning(Mdl,xTr,yTr,xTe,yTe);

    %% Coompute the classification error using DWC with individual classification rules obtained using KMM

        for d = 1:Mdl.distribs
            clfCWKMM = Mdl;
            clfCWKMM.lambda0 = 0;
            clfCWKMM = KMM(clfCWKMM, xTr{d}, xTe);
            clfCWKMM = IWMRC_parameters(clfCWKMM, xTr{d}, yTr{d});
            clfCWKMM = IWMRC_learning(clfCWKMM, xTr{d});
            clfCWKMM = MRC_prediction(clfCWKMM, xTe, yTe);
            CWKMM_h{d,1} = clfCWKMM.h;
        end
        ClasWeightKMM = KDE_MultiSource(xTr,xTe);
        ErrorCWKMM(rep,1) = CWKMM_prediction(CWKMM_h,ClasWeightKMM,yTe,clfCWKMM.deterministic,clfCWKMM.labels);

    %% Compute the classification error using MS-DW

    D = 1 ./ (1 - (0:0.1:0.9)) .^ 2;
        for d = 1:Mdl.distribs
            psix_tr{d} = psi(Mdl, xTr{d});
        end
        psix_te  = psi(Mdl, xTe);
        old_fmap = Mdl.fmap;
        Mdl.fmap = 'linear';

        for k = 1:length(D)
            MSDW{k} = Mdl;
            MSDW{k}.loss = '0-1';
            MSDW{k}.distribs = Mdl.distribs;
            MSDW{k}.D = D(k);
            MSDW{k}.B = 1000;
            MSDW{k}     = MSKMM(MSDW{k},xTr,xTe);
            MSDW{k}     = MSDW_parameters(MSDW{k},psix_tr,yTr,psix_te);
            MSDW{k}     = MSDW_learning(MSDW{k},psix_te);
            min_MSDW(k) = MSDW{k}.min_MRC;
        end
        [~,position] = min(min_MSDW);
        MSDW{position}     = MSDW_prediction(MSDW{position},psix_te,yTe);
        Mdl.fmap = old_fmap;
        ErrorDWMSCS(rep,1) = MSDW{position}.error;

end
[~,pos] = min(mean(ErrorLR));
ErrorLR = ErrorLR(:,pos);
[~,pos] = min(mean(ErrorKMM));
ErrorKMM = ErrorKMM(:,pos);
[~,pos] = min(mean(ErrorDWGCS));
ErrorDWGCS = ErrorDWGCS(:,pos);
end