function Experiments_MSLS(idx1,alpha_type)
% Add paths to existing and proposed methods

% Clear all

clearvars -except idx1 alpha_type

% Add paths to CVX

addpath('../cvx')
cvx_setup
cvx_solver mosek
cvx_save_prefs

% Add paths to existing and proposed methods

addpath('../Functions_Library_DW-MSLS/LR');
addpath('../Functions_Library_DW-MSLS/DWMSLS');
addpath('../Functions_Library_DW-MSLS/Rw');
addpath('../Functions_Library_DW-MSLS/Auxiliar');

% Initialize the base model

clf.intercept = true;
clf.fmap = 'linear';
clf.loss = '0-1';
clf.deterministic = false;
clf.lambda0 = 0;
clf.labels = 2;
clf.lambda_kernel = 0.001;
n_classes = clf.labels;
clf.distribs = 3;

%%

load('Domains.mat');
Domains{1,1} = full(Domains{1,1});
Domains{2,1} = full(Domains{2,1});
Domains{3,1} = full(Domains{3,1});
Domains{4,1} = full(Domains{4,1});


XY = cell2mat(Domains);
X  = XY(:,1:end-1);

for rep = 1:20

    xTe = [];
    yTe = [];
    cont = 1;

    switch alpha_type
        case 1
            alpha_ = 0.01 * ones(1, 2);
        case 2
            alpha_ = 0.1 * ones(1, 2);
        case 3
            alpha_ = ones(1, 2);
        case 4
            alpha_ = 10 * ones(1, 2);
        case 5
            alpha_ = 100 * ones(1, 2);
        otherwise
            error('Invalid alpha_type');
    end

    for i = 1:4
        % Generate Dirichlet distribution proportions for the current domain
        label_proportions = dirrnd(alpha_, 1);
        if label_proportions(1) > 0.99
            label_proportions = [0.99, 0.01];
        elseif label_proportions(1) < 0.01
            label_proportions = [0.01, 0.99];
        end

        % Subsample 500 samples for the test set with the generated proportions
        xTe_i = [];
        yTe_i = [];

        Domains{i} = Domains{i}(randperm(2000), :);
        for j = 1:n_classes
            class_indices = find(Domains{i}(:, end) == j);
            num_samples = round(150 * 0.5);
            if num_samples > length(class_indices)
                num_samples = length(class_indices);
            end
            selected_indices = randsample(class_indices, num_samples);
            xTe_i = [xTe_i; Domains{i}(selected_indices, 1:end-1)];
            yTe_i = [yTe_i; Domains{i}(selected_indices, end)];
        end
        xTe = [xTe; xTe_i];
        yTe = [yTe; yTe_i];

        % Subsample 1000 samples for the training set from remaining 1000 samples
        if i ~= idx1
            xTr{cont,1} = [];
            yTr{cont,1} = [];
            for j = 1:n_classes
                class_indices = find(Domains{i}(:, end) == j);
                num_samples = round(1000 * label_proportions(j));
                if num_samples > length(class_indices)
                    num_samples = length(class_indices);
                end
                selected_indices = randsample(class_indices, num_samples);
                xTr{cont,1} = [xTr{cont,1}; Domains{i}(selected_indices, 1:end-1)];
                yTr{cont,1} = [yTr{cont,1}; Domains{i}(selected_indices, end)];
            end
            cont = cont + 1;
        end
    end

    r1 = randperm(length(yTe));
    xTe = xTe(r1,:);
    yTe = yTe(r1);
    for s = 1 : clf.distribs
        r1 = randperm(length(yTr{s}));
        xTr{s} = xTr{s}(r1,:);
        yTr{s} = yTr{s}(r1);
    end

    %% Compute the classification error using ERM LR


    for s = 1:clf.distribs
        clf1            = clf;
        clf1.deterministic = true;
        clf1.loss = 'log';
        clf1            = LR_parameters(clf1, xTr{s}, yTr{s});
        clf1            = LR_learning(clf1, xTr{s});
        clf1            = LR_prediction(clf1, xTe, yTe);
        Error_LR(rep, s) = clf1.error;
    end

%% Compute the classification error using Reweighted

    h_mix = zeros(clf.labels, length(yTe));
    beta_mix = zeros(1, clf.labels);
    clf1 = clf;
    clf1 = MSLS_KMM(clf1, xTr, yTr, xTe);
    all_betas = clf1.pte_ptr;
    for s = 1: clf.distribs
        clf1 = clf;
        clf1.loss = 'log';
        clf1.distribs = 1;
        clf1.beta = all_betas{s};
        clf1 = MSLSrw_parameters(clf1, xTr{s}, yTr{s});
        clf1 = MRC_prediction(clf1, xTe, yTe);
        ptr = histcounts(yTr{s}, [1,2,3])/1000;
        h_mix = h_mix + ptr' .* clf1.h;
        beta_mix = beta_mix + ptr;
    end
    h_mix = h_mix ./ beta_mix';
    h_mix = h_mix ./ sum(h_mix, 1);
    y_aprox = zeros(size(yTe));
    fails = 0;
    for i = 1 : length(yTe)
        [~, y_aprox(i)] = max(h_mix(:,i));
        if y_aprox(i) ~= yTe(i)
            fails = fails + 1;
        end
    end
    Error_LWCKMM(rep, 1) = fails / length(yTe);

%% Compute the classification error using Single-Source Reweighted

    clf1 = clf;
    clf1 = MSLS_KMM(clf1, xTr, yTr, xTe);
    all_betas = clf1.pte_ptr;
    for s = 1:clf.distribs
        clf1            = clf;
        clf1.deterministic = true;
        clf1.loss = 'log';
        clf1.distribs = 1;
        clf1.beta = all_betas{s};
        clf1 = MSLSrw_parameters(clf1, xTr{s}, yTr{s});
        clf1 = MRC_prediction(clf1, xTe, yTe);
        Error_KMM(rep, s) = clf1.error;
    end

%% Compute the classification error using DW-MSLS

    Ds = 1 ./ (1 - [0 : 0.1 : 0.9]) .^ 2;
    beta = [];
    alpha = [];
    clf1=[];
    for i = 1:numel(Ds)
        clf1{i} = clf;
        clf1{i}.deterministic = true;
        clf1{i} = MSLS_KMM(clf1{i}, xTr, yTr, xTe);
        w_aux = zeros(n_classes, 1);
        for y_idx = 1:n_classes
            for s_aux = 1 : clf.distribs
                w_aux(y_idx) = w_aux(y_idx) + 1./clf1{i}.pte_ptr{s_aux}(y_idx);
            end
            w_aux(y_idx) = 1./ w_aux(y_idx);
        end
        for s = 1:clf.distribs
            beta{s}  = min(w_aux, (1 / Ds(i)) * max(w_aux) * ones(clf1{i}.labels, 1));
            alpha{s} = min(w_aux ./ clf1{i}.pte_ptr{s}, (1 / Ds(i)) * max(w_aux) ./ clf1{i}.pte_ptr{s});
        end
        clf1{i}.beta = beta;
        clf1{i}.alpha = alpha;
        clf1{i} = MSLS_parameters(clf1{i}, xTr, yTr, xTe);
        RU(i) = clf1{i}.min_MRC;
        clf1{i} = MSLS_prediction(clf1{i}, xTe, yTe);
        Error_DWMRC(rep, i) = clf1{i}.error;
    end
    [~,min_RU] = min(RU);
    Error_DWMSCS(rep, 1) = Error_DWMRC(rep, min_RU);

%% Compute the classification error using DW-LS

    Ds = 1 ./ (1 - [0 : 0.1 : 0.9]) .^ 2;
    for s = 1:clf.distribs
        clf1 = [];
        for l = 1:length(Ds)
            clf1{l}      = clf;
            clf1{l}.deterministic = true;
            clf1{l} = MSLS_KMM(clf1{l}, xTr, yTr, xTe);
            w_aux = zeros(n_classes, 1);
            for y_idx = 1:n_classes
                for s_aux = 1 : clf.distribs
                    w_aux(y_idx) = w_aux(y_idx) + 1./clf1{l}.pte_ptr{s_aux}(y_idx);
                end
                w_aux(y_idx) = 1./ w_aux(y_idx);
            end
            beta{s}  = min(clf1{l}.pte_ptr{s}, (1 / Ds(l)) * max(clf1{l}.pte_ptr{s}) * ones(clf1{l}.labels, 1));
            alpha{s} = min(ones(clf1{l}.labels, 1), (1 / Ds(l)) * max(clf1{l}.pte_ptr{s}) ./ clf1{l}.pte_ptr{s});
            clf1{l}.beta = beta{s};
            clf1{l}.alpha = alpha{s};
            clf1{l}.distribs = 1;
            clf1{l}      = MSLS_parameters(clf1{l}, xTr{s}, yTr{s}, xTe);
            RU(l) = clf1{l}.min_MRC;
            clf1{l} = MSLS_prediction(clf1{l}, xTe, yTe);
            Error_DWMRC(rep, l) = clf1{l}.error;
        end
        [~,min_RU] = min(RU);
        Error_DWLS(rep, s) = Error_DWMRC(rep, min_RU);
    end

end
[~,pos] = min(mean(Error_LR));
Error_LR = Error_LR(:,pos);
[~,pos] = min(mean(Error_KMM));
Error_KMM = Error_KMM(:,pos);
[~,pos] = min(mean(ErrorDWGCS));
Error_DWLS = Error_DWLS(:,pos);
end