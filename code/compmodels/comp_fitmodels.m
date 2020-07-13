function comp_fitmodels()
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Paths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear;

% Specify paths
% Paths for toolboxes
p.HGF_path = '/home/mp/Documents/MATLAB/HGF';
p.VBA_path = '/home/mp/Documents/MATLAB/VBA-toolbox-master';

% Custom tapas models path
p.customfuncpath = ['/home/mp/gdrive/projects/'...
                    '2020_painlearning/code/compmodels/'];
% Where is the data
p.datafile =  ['/media/mp/lxhdd/2020_painlearning/derivatives'...
                '/task-fearcond_scr_glmresp_processed.mat'];
% Ouput path
m.path = '/media/mp/lxhdd/2020_painlearning/derivatives/compmodels';
if ~exist(m.path, 'dir')
   mkdir(m.path)
end

% Add toolbox and functions to path
addpath(p.HGF_path);
addpath(genpath(p.VBA_path));
addpath(genpath(p.customfuncpath));

% Name to give to comparison file
p.comparison_name = 'comp_all_';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Models to run using names below or 'all'
p.to_run =  {'HGF2_intercue'};


% Models to compare using VBA (if empty compares model that were ran)
% Use this to compare models without running any
p.to_compare =  {};
p.comp_families = {};

% General model parameters
m.optim = 'tapas_quasinewton_optim_config'; % Optimisation function


% Exclude participants
p.substoremove = {'24', '31', '35', '51'};

% Design
m.nofirstblock = 0; % DO not consider first block
m.ignoreshocks = 1; % Ignore the shock trials in the response model

% Transform
m.zscore = 0; % Z score data before running
m.meancenter = 0; % Mean center data before run
m.logtrans = 0; % Log transform before running
m.removeoutliers = 3; % Remove outliers in response data before running (0 or Z score threshold)

% Input
m.uselpp = 0; % Use LPP data instead of SCR

% Use average parameters
m.use_avarage_param = 0;
m.use_bayesian_average = 0;  % Bayesian or regular average

% Output
m.makeplot = 1; % Create plots after fit
m.makeindividualplot = 0; % Create plots for each subject



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load data
d = load(p.datafile);
m.data = d.data;

% Run options
debug = 0; % Just run a couple of participants to test
if debug == 1
    m.data = m.data(1:3);
end

% Exclude participants
keep = ones(1, length(m.data));
for i = 1:length(m.data)
    for rems = 1:length(p.substoremove)
        if strcmp(p.substoremove(rems), num2str(m.data{i}.sub))
            keep(i) = 0;
        end
    end
end
m.data = m.data(logical(keep));

% Add cspplus for numm model
for s = 1:length(m.data)
    m.data{s}.csplus = zeros(length(m.data{s}.trials), 1);
    m.data{s}.csplus([strmatch('CSplusSI', m.data{s}.condition); strmatch('CSplus ', m.data{s}.condition)]) = 1;
end

% Array to collect LME for each model
L_vba = [];
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODELS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
models = {}; % Init empty cell
m.simulate = 0;
m.usehab = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Null models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m.HGF=0; % Not an HGF
m.name = 'null_binary'; % Name to save under
m.prc_model = 'null_binary_config'; % Perceptual model function
m.resp_model = 'resp_RW_vhat_config'; % Response model function
m.tolineplot = {'state'}; % Parameters to plot
m.tolineplot_labels = {'cspplus'}; % Labels for line plots
m.resp_param = {'be0', 'be1', 'ze'}; % Response paramters
m.perc_param = {}; % Perceptual paramters
m.u = "[m.data{s}.shock', m.data{s}.csplus]";  % Data input
models{end+1} = m;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RW no intercue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simplest model
m.HGF=0; % Not an HGF
m.name = 'RW_nointercue'; % Name to save under
m.prc_model = 'RW_nointercue_config'; % Perceptual model function
m.resp_model = 'resp_RW_vhat_config'; % Response model function
m.tolineplot = {'vhat'}; % Parameters to plot
m.tolineplot_labels = {'Expectation'}; % Labels for line plots
m.resp_param = {'be0', 'be1', 'ze'}; % Response paramters
m.perc_param = {'v_0', 'al'}; % Perceptual paramters
m.u = "[m.data{s}.shock', m.data{s}.cuenum']";  % Data input
models{end+1} = m;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RW intercue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m.HGF=0; % Not an HGF
m.name = 'RW_intercue'; % Name to save under
m.prc_model = 'RW_intercue_config'; % Perceptual model function
m.resp_model = 'resp_RW_vhat_config'; % Response model function
m.tolineplot = {'vhat'}; % Parameters to plot
m.tolineplot_labels = {'Expectation'}; % Labels for line plots
m.resp_param = {'be0', 'be1', 'ze'}; % Response paramters
m.perc_param = {'v_0', 'al'}; % Perceptual paramters
m.u = "[m.data{s}.shock', m.data{s}.cuenum']";  % Data input
models{end+1} = m;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PH no intercue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m.HGF = 0;
m.name = 'PH_nointercue';
m.prc_model = 'PH_nointercue_config';
m.resp_model = 'resp_PH_vhat_assoc_config';
m.tolineplot = {'vhat', 'a'};
m.tolineplot_labels = {'Expectation', 'Associativity'};
m.resp_param = {'be0', 'be1', 'be2', 'ze'};
m.perc_param = {'v_0', 'al', 'a_0', 'ga'};
m.u = "[m.data{s}.shock', m.data{s}.cuenum']";
models{end+1} = m;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PH intercue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m.HGF = 0;
m.name = 'PH_intercue';
m.prc_model = 'PH_intercue_config';
m.resp_model = 'resp_PH_vhat_assoc_config';
m.tolineplot = {'vhat', 'a'};
m.tolineplot_labels = {'Expectation', 'Associativity'};
m.resp_param = {'be0', 'be1', 'be2', 'ze'};
m.perc_param = {'v_0', 'al', 'a_0', 'ga'};
m.u = "[m.data{s}.shock', m.data{s}.cuenum']";
models{end+1} = m;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 levels HGF global options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m.HGF = 1;
m.hgflevels = 2;
m.simulate = 0;
m.usehab = 0;
m.tolineplot = {'vhat', 'sa1hat', 'sa2hat'};
m.tolineplot_labels = {'Expectation (vhat)', ...
    'Irreducible uncertainty (sa1hat)', ...
    'Estimation uncertainty (sa2hat)'};
m.sim_prc_model = 'HGF_2levels_config_sim'; % Model for simulation
m.prc_model = 'HGF_2levels_intercue_config';
m.perc_param = {'om2'};
m.u = "[m.data{s}.shock', m.data{s}.cuenum']";


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 levels HGF no interecue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2 levels, uncertainty only, 1 predictor no intercue

%___________________________________________________
% Simulation
m.name = 'HGF2_nointercue_sim';
m.prc_model = 'HGF_2levels_config_sim';
m.resp_model = 'tapas_bayes_optimal_binary_config';
m.simulate = 1;
models{end+1} = m;

m.name = 'HGF2_nointercue';
m.prc_model = 'HGF_2levels_config';
m.resp_model = 'HGF_sa1hat_config';
m.resp_param = {'be0', 'be1'};
m.simulate = 0;
models{end+1} = m;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 levels HGF intercue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulation
m.name = 'HGF2_intercue_sim';
m.prc_model = 'HGF_2levels_intercue_config_sim';
m.resp_model = 'tapas_bayes_optimal_binary_config';
m.simulate = 1;
models{end+1} = m;

m.name = 'HGF2_intercue';
m.prc_model = 'HGF_2levels_intercue_config';
m.resp_model = 'HGF_sa1hat_config';
m.resp_param = {'be0', 'be1'};
m.simulate = 0;
models{end+1} = m;

m.name = 'HGF2_intercue_hab';
m.prc_model = 'HGF_2levels_intercue_config';
m.resp_model = 'HGF_sa1hat_hab_config';
m.resp_param = {'be0', 'be1'};
m.simulate = 0;
models{end+1} = m;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3 levels HGF global options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%___________________________________________________
% HGF 3 levels global options
m.HGF = 1;
m.hgflevels = 3;
m.simulate = 0;
m.usehab = 0;
m.tolineplot = {'vhat', 'sa1hat', 'sa2hat', 'pv', 'vol'};
m.tolineplot_labels = {'Expectation (vhat)', ...
                       'Irreducible uncertainty', ...
                       'Estimation uncertainty',...
                       'Phasic volatility',...
                       'Volatility uncertainty'};
m.sim_prc_model = 'HGF_3levels_config_sim'; % Model for simulation
m.prc_model = 'HGF_3levels_intercue_config';
m.perc_param = {'om2', 'om3'};
m.u = "[m.data{s}.shock', m.data{s}.cuenum']";




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3 levels HGF no intercue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulate
m.name = 'HGF3_nointercue_sim';
m.prc_model = 'HGF_3levels_config_sim';
m.resp_model = 'tapas_bayes_optimal_binary_config';
m.simulate = 1;
models{end+1} = m;


m.name = 'HGF3_nointercue';
m.prc_model = 'HGF_3levels_config';
m.resp_model = 'HGF_sa1hat_config';
m.resp_param = {'be0', 'be1'};
m.simulate = 0;
models{end+1} = m;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3 levels HGF intercue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m.name = 'HGF3_intercue_sim';
m.prc_model = 'HGF_3levels_intercue_config_sim';
m.resp_model = 'tapas_bayes_optimal_binary_config';
m.simulate = 1;
models{end+1} = m;



m.name = 'HGF3_intercue';
m.prc_model = 'HGF_3levels_intercue_config';
m.resp_model = 'HGF_sa1hat_config';
m.resp_param = {'be0', 'be1'};
m.simulate = 0;
models{end+1} = m;



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run/Compare selected models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run
if sum(strcmp(p.to_run, 'all')) || sum(strcmp(p.to_compare, 'all'))
    names = {};
    for mod = 1:length(models) % Loop all models
        names{end+1} = models{mod}.name;
    end
end
for mod = 1:length(models) % Loop all models
    % Run if in torun or 'all
    if sum(strcmp(models{mod}.name, p.to_run)) || sum(strcmp(p.to_run, 'all'))
        m = fearcond_fit_HGF(models{mod});
        % Keep LME if not simulation
        if ~m.simulate
            L_vba = [L_vba, m.LMEs];
        end
    end
end

% Compare
if size(L_vba, 2) > 1
    if sum(strcmp(p.to_run, 'all'))
        fearcond_compare_models(L_vba', m.path, names, p.comp_families)
    else
        fearcond_compare_models(L_vba', m.path, p.to_run, p.comp_families)
    end
end

% Compare without running
if ~isempty(p.to_compare)
    names = {};
    for mod = 1:length(models)
        if sum(strcmp(models{mod}.name, p.to_compare)) || sum(strcmp(p.to_compare, 'all'))
            load(fullfile(m.path, models{mod}.name, [models{mod}.name '_data']))

            L_vba = [L_vba, m.LMEs];
            names{end+1} = models{mod}.name;
        end
    end
    if sum(strcmp(p.to_compare, 'all'))
        fearcond_compare_models(L_vba', m.path, names, p.comp_families)
    else
        fearcond_compare_models(L_vba', m.path, names, p.comp_families)
    end
end

end

%%
function m = fearcond_fit_HGF(m)

% Make output dir
outpath = fullfile(m.path, m.name);
if ~exist(outpath, 'dir')
    mkdir(outpath)
end

% Fit model for each subject
m.subfit = {};

% Collect LMEs, AICs, BICs, in a single column for model comparison
m.LMEs = nan(length(m.data), 1);
m.BICs = nan(length(m.data), 1);
m.AICs = nan(length(m.data), 1);

% Loop subject
for s = 1:length(m.data)

    % Extract inputs and response
    if isfield(m, 'u')
        u = eval(m.u);  % If special input;
    else
        u = m.data{s}.shock';
    end

    % Add number of shocks received to each trial

    if ~m.uselpp
        y = m.data{s}.response';
    else
        y = m.data{s}.lpp;
    end

    if m.nofirstblock
        y(1:36) = [];
        u(1:36, :) = [];
    end

    tic

    % Ignore shock trials in response function
    if m.ignoreshocks
        y(u(:,1) == 1) = nan;
    end

    if m.zscore
        y = (y-nanmean(y))/nanstd(y);
    end

    if m.meancenter
        y = y-nanmean(y);
    end

    if m.logtrans
        y = log(y+2);
    end

    if m.removeoutliers ~= 0
        z = abs((y-nanmean(y))/nanstd(y));
        y(z > m.removeoutliers) = nan;
        %m.data{s}.response = y';
        m.data{s}.nremoved = sum(z > m.removeoutliers);
    else
        m.data{s}.nremoved = 0;
    end

    if ~m.simulate % IF not simulation
        disp(['Fitting model ' m.name ' for sub ' num2str(s) ' out of '...
            num2str(length(m.data))])
        m.subfit{s} = tapas_fitModel(y,...
            u,...  % Inputs
            m.prc_model,... % Perceptual model
            m.resp_model, ... % Response model
            m.optim);

        % GEt model fits
        SSE = nansum(m.subfit{s}.optim.res.^2);
        m.subfit{s}.optim.SSE = SSE;
        m.LMEs(s) = m.subfit{s}.optim.LME;
        m.BICs(s) = m.subfit{s}.optim.BIC;
        m.AICs(s) = m.subfit{s}.optim.AIC;

        %         clc;
        %         disp(m.LMEs(:))

    else %Estimate priors for HGF
        disp(['Simulating model ' m.name ' for sub ' num2str(s) ' out of '...
            num2str(length(m.data))])
        sim{s} = tapas_fitModel([],...
            u,...  % Inputs
            m.prc_model,...
            'tapas_bayes_optimal_binary_config', ...
            m.optim);
        minparam(s, :) = sim{s}.optim.argMin;
        clc;
        %         disp(minparam(:, :))
    end
    toc


end



% Use average parameters
if m.use_avarage_param && m.simulate == 0
    disp('Refitting using average parameters')
    % Get average parameters
    if m.use_bayesian_average
        params = tapas_bayesian_parameter_average(m.subfit{:});
    else

        for s = 1:length(m.data)
            if s == 1
                params.p_prc.p = m.subfit{s}.p_prc.p;
                params.p_obs.ptrans = m.subfit{s}.p_obs.ptrans;

            else
                params.p_prc.p  = [params.p_prc.p; m.subfit{s}.p_prc.p];
                params.p_obs.ptrans = [params.p_obs.ptrans; m.subfit{s}.p_obs.ptrans];
            end
        end
        params.p_prc.p = mean(params.p_prc.p);
        params.p_obs.ptrans = mean(params.p_obs.ptrans);
    end
    % Refit using average parameters
    for s = 1:length(m.data)
        % Extract inputs and response
        if isfield(m, 'u')
            u = eval(m.u);  % If special input;
        else
            u = m.data{s}.shock';
            y = u; % y does not matter here
        end

        if m.nofirstblock
            y(1:36) = [];
            u(1:36, :) = [];
        end

        % Ignore shock trials in response function
        if m.ignoreshocks
            y(u(:,1) == 1) = nan;
        end


        % Use sim model to fit model with averaged parameters

        % Get perceptual function
        c = str2func(m.prc_model);
        c = c();
        prc_fun = c.prc_fun;
        % Run it
        r.u = u;
        r.y = y;
        r.irr = [];
        % Fit perceptual model using averaged parameters
        [m.subfit{s}.avgtraj, infstates] = prc_fun(r, params.p_prc.p);
        m.avgparams.p_prc =  params.p_prc.p;

        % Get response function
        c = str2func(m.resp_model);
        c = c();
        resp_fun = c.obs_fun;
        m.avgparams.p_obs = params.p_obs;

        % Fit response function using averaged parameters and new infstates
        [~, m.subfit{s}.optim.avgyhat, ~] = resp_fun(r, infstates, ...
                                                     params.p_obs.ptrans);


        % Backup individual traj and replace with average traj
        m.subfit{s}.nonavg_traj = m.subfit{s}.traj;
        m.subfit{s}.optim.nonavg_yhat = m.subfit{s}.optim.yhat;

        m.subfit{s}.traj = m.subfit{s}.avgtraj;
        m.subfit{s}.optim.yhat = m.subfit{s}.optim.avgyhat;

    end

end


if m.simulate
    disp('Average parameters for bayes ideal observer')
    mean(minparam)
    m.sim_average_min_params = mean(minparam);
    m.sim_min_params = minparam;
    m.sim = sim;
    tapas_bayesian_parameter_average(sim{:})
    save(fullfile(m.path, m.name, [m.name '_data']), 'm')

else
    % Bayesian averaging across participants
    % m.bpa = tapas_bayesian_parameter_average(m.subfit{:});
    % Save model
    save(fullfile(m.path, m.name, [m.name '_data']), 'm')

    % Collect data in a table
    m = HGF_data2table(m);

    % Save model
    save(fullfile(m.path, m.name, [m.name '_data']), 'm')

    if m.makeplot == 1
        % Make some plots
        m = HGF_plot_model(m);

        % Save model
        save(fullfile(m.path, m.name, [m.name '_data']), 'm')
    end
end


end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put data in table
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function m = HGF_data2table(m)
allsubs = [];
allcond = {};

for s = 1:length(m.data)
    % Data
    sub = repmat(m.data{s}.sub, length(m.data{s}.response), 1);

    % Get predicted response

    pred = m.subfit{s}.optim.yhat;
    % get data
    if strcmp(m.name, 'HGF-whatworld')
        trial = (1:length(pred))';
        scr = m.data{s}.response';
        cue = m.data{s}.cuenum';
        cond = m.data{s}.condition';
    else
        trial = (1:length(pred))';
        scr = m.data{s}.response';
%         lpp = m.data{s}.lpp;
        cue = m.data{s}.cuenum';
        cond = m.data{s}.condition';
    end

    if m.logtrans
        scr = log(scr+2);
%         lpp = log(lpp+1);
    end
    if m.zscore
        scr = (scr-nanmean(scr))/nanstd(scr);
%         lpp = (lpp-nanmean(lpp))/nanstd(lpp);
    end

    if m.meancenter
        scr = (scr-nanmean(scr));
        lpp = (lpp-nanmean(lpp));
    end


    if m.uselpp
        % Simple to just replace for now.
        scr = lpp;
    end

    % Add block number
    block = [];
    for b = 1:7
        if b == 1
            block = [block; repmat(b, 36, 1)];
        else
            block = [block; repmat(b, 72, 1)];
        end
    end

    % Rename conditions and add double conditions for plots
    cond2 = cell(max(trial), 1);
    cond_plot = cell(max(trial), 1);
    cond_plot2 = cell(max(trial), 1);

    startcues = unique(cue(1:36));

    for c = 1:length(cond)

        switch cond{c}
            case 'CSminus '
                cond2{c} = 'CS-1';
                cond_plot{c} = ['CS-1_' num2str(block(c)) ...
                    '/CS-2_' num2str(block(c)+1) '_' num2str(find(cue(c) == startcues))];
                cond_plot2{c} = cond_plot{c};
            case 'CSminus2'
                cond2{c} = 'CS-2';
                if block(c) == 2
                    cond_plot{c} = ['CS-1_' num2str(block(c)-1) ...
                        '/CS-2_' num2str(block(c)) '_' num2str(find(cue(c) == startcues))];
                else
                    cond_plot{c} = ['CS-1_' num2str(block(c)-1) ...
                        '/CS-2_' num2str(block(c))];
                end
                cond_plot2{c} = cond_plot{c};

            case 'CSnaif1 '
                cond2{c} = 'CS-1';
                cond_plot{c} = ['CS-1_' num2str(block(c)) ...
                    '/CS-2_' num2str(block(c)+1)];
                cond_plot2{c} = cond_plot{c};
            case 'CSnaif2 '
                cond2{c} = 'CS-2';
                cond_plot{c} = ['CS-1_' num2str(block(c)-1) ...
                    '/CS-2_' num2str(block(c))];
                cond_plot2{c} = cond_plot{c};
            case 'CSplus  '
                cond2{c} = 'CS+';
                cond_plot{c} = ['CS+_' num2str(block(c)) ...
                    '/CS-E_' num2str(block(c)+1)];
                cond_plot2{c} = cond_plot{c};
            case 'CSplusSI'
                cond2{c} = 'CS++';
                cond_plot{c} = ['CS++_' num2str(block(c)) ...
                    '/CS-E_' num2str(block(c)+1)];
                cond_plot2{c} = ['CS+_' num2str(block(c)) ...
                    '/CS-E_' num2str(block(c)+1)];
%             case 'CSplusSI'
%                 cond2{c} = 'CS+';
%                 cond_plot{c} = ['CS+_' num2str(block(c)) ...
%                     '/CS-E_' num2str(block(c)+1)];
            case 'CSeteint'
                cond2{c} = 'CS-E';
                cond_plot{c} = ['CS+_' num2str(block(c)-1) ...
                    '/CS-E_' num2str(block(c))];
                cond_plot2{c} = cond_plot{c};

        end
    end

    ucond_plot = unique(cond_plot);
    count = ones(length(ucond_plot), 1);
    trial_within = cell(max(trial), 1);
    trial_within_wb = cell(max(trial), 1);

    for i = 1:length(cond_plot)

        % Reset count for each block
        if i > 1 && block(i-1) ~= block(i)
            count = ones(length(ucond_plot), 1);
        end

        % Where is it
        trial_within{i} = count(strmatch(cond_plot{i}, ucond_plot));

        % update
        count(strmatch(cond_plot{i}, ucond_plot)) = count(strmatch(cond_plot{i}, ucond_plot)) + 1;

        switch block(i)
            case 1
                trial_within_wb{i} = trial_within{i};
            otherwise
                trial_within_wb{i} = trial_within{i} + (18*(block(i)-1));
        end

    end

    ucond_plot = unique(cond_plot2);
    count = ones(length(ucond_plot), 1);
    trial_within_wcs = cell(max(trial), 1);
    trial_within_wb_wcs = cell(max(trial), 1);
    for i = 1:length(cond_plot)

        % Reset count for each block
        if i > 1 && block(i-1) ~= block(i)
            count = ones(length(ucond_plot), 1);
        end

        % Where is it
        trial_within_wcs{i} = count(strmatch(cond_plot2{i}, ucond_plot));

        % update
        count(strmatch(cond_plot2{i}, ucond_plot)) = count(strmatch(cond_plot2{i}, ucond_plot)) + 1;

        switch block(i)
            case 1
                trial_within_wb_wcs{i} = trial_within_wcs{i};
            otherwise
                trial_within_wb_wcs{i} = trial_within_wcs{i} + (18*(block(i)-1));
        end

    end


    % Get trialwise estimations for parameters that will be pltted
    m.traj_names = m.tolineplot;
    traj_data = nan(length(pred), length(m.traj_names)-1);

    if m.HGF
        % IF HGF, extract cue trajectory

        mu1hat = m.subfit{s}.traj.muhat(:, 1);
        sa2 = m.subfit{s}.traj.sa(:, 2);
        sa3 = m.subfit{s}.traj.sa(:, 3);
        mu2 = m.subfit{s}.traj.mu(:, 2);
        sa1hat = m.subfit{s}.traj.sahat(:, 2);
        mu3 = m.subfit{s}.traj.mu(:, 3);

        m.subfit{s}.traj.mu1hat = mu1hat;
        m.subfit{s}.traj.vhat = mu1hat; % Duplicate for plots
        m.subfit{s}.traj.sa1hat = mu1hat.*(1-mu1hat);
        m.subfit{s}.traj.sa2hat = tapas_sgm(mu2, 1).*(1 -tapas_sgm(mu2, 1)).*sa2;
        if m.hgflevels == 3
            m.subfit{s}.traj.pv = tapas_sgm(mu2, 1).*(1-tapas_sgm(mu2, 1)).*exp(mu3);
            m.subfit{s}.traj.vol = tapas_sgm(mu2, 1).*(1-tapas_sgm(mu2, 1)).*sa3;
        end
        for p = 1:length(m.traj_names)
            traj_data(:, p) = m.subfit{s}.traj.(m.traj_names{p});
        end

    else% Non HGF
        for p = 1:length(m.traj_names)
            traj_data(:, p) = m.subfit{s}.traj.(m.traj_names{p});
        end
    end


    % Get all parameters value
    % Perceptual parameters

    if ~m.HGF
        m.perc_param = fieldnames(m.subfit{s}.p_prc);
        m.perc_param = m.perc_param(1:end-2); % Remove summary
        perc_data = nan(length(pred), length(m.perc_param));
        for p = 1:length(m.perc_param)
            perc_data(:, p) = repmat(m.subfit{s}.p_prc.(m.perc_param{p}), length(pred), 1);
        end
    else % For HGF
        alllevels = {};
        alldata = [];
        m.perc_param = fieldnames(m.subfit{s}.p_prc);
        m.perc_param = m.perc_param(1:end-2); % Remove summary
        for p = 1:length(m.perc_param)
            for l = 1:length(m.subfit{s}.p_prc.(m.perc_param{p}))
                alllevels = [alllevels, {[m.perc_param{p}, '_' num2str(l)]}];
                alldata = [alldata, m.subfit{s}.p_prc.(m.perc_param{p})(l)];
            end
        end

        m.perc_param = alllevels';
        perc_data = nan(length(pred), length(m.perc_param));
        for p = 1:length(m.perc_param)
            perc_data(:, p) = repmat(alldata(p), length(pred), 1);
        end

    end

    % Response parameters
    m.resp_param = fieldnames(m.subfit{s}.p_obs);
    m.resp_param = m.resp_param(1:end-2); % Remove summary
    resp_data = nan(length(pred), length(m.resp_param));
    for p = 1:length(m.resp_param)
        resp_data(:, p) = repmat(m.subfit{s}.p_obs.(m.resp_param{p}), length(pred), 1);
    end


    % Get all model fits
    AIC = repmat(m.subfit{s}.optim.AIC, length(pred), 1);
    BIC = repmat(m.subfit{s}.optim.BIC, length(pred), 1);
    LME = repmat(m.subfit{s}.optim.LME, length(pred), 1);
    SSE = repmat(nansum(m.subfit{s}.optim.res.^2), length(pred), 1);
    nremoved = repmat(m.data{s}.nremoved, length(pred), 1);

    if m.nofirstblock
        scr(1:36) = [];
        cond(1:36) = [];
        sub(1:36) = [];
        block(1:36) = [];
        cond2(1:36) = [];
        cond_plot(1:36) = [];

    end

    % Put all in same array
    allsubs = [allsubs; sub, trial, cue, block, scr, pred, traj_data,...
        perc_data, resp_data, AIC, BIC, LME, SSE, nremoved];
    allcond = [allcond; [cond, cond2, cond_plot, cond_plot2,...
               trial_within, trial_within_wb, ...
               trial_within_wcs, trial_within_wb_wcs]];

end

tablehead = [{'sub', 'trial', 'cue', 'block', 'scr', 'pred'}, m.traj_names,...
    m.perc_param', m.resp_param', {'AIC', 'BIC', 'LME', 'SSE', 'nremoved',...
    'cond_original', 'cond', 'cond_plot', 'cond_plot2', 'trial_within',...
    'trial_within_wb', 'trial_within_wcs', 'trial_within_wb_wcs'}];

tabledata = [num2cell(allsubs), allcond];


m.tabdata = cell2table(tabledata, 'VariableNames', tablehead);
writetable(m.tabdata, fullfile(m.path, m.name, [m.name , '_data.csv']))

end


%%
function m = HGF_plot_model(m)
close all;

% Remove shock trials
if m.ignoreshocks == 1
    pdata = m.tabdata(~strcmp(m.tabdata.cond, 'CS++'), :);
else
    pdata = m.tabdata;
end

allplots = {};


% Plots parameters
p.path = fullfile(m.path, m.name, 'fig'); % Where to save plots
p.size = [10, 10, 900 600]; % Plot size
p.labelfsize = 24;  % Font size for axis labels
p.axisfsize = 22; % Font size for ticks labels
p.resolution = '-r150';  % Figure resolution
p.markesizelines = 10;  % Markersizes
p.markersizebox = 20;
p.linewidth = 3;

mkdir(fullfile(m.path, m.name, 'fig'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Line plots for trial wise values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tolineplot = [{'scr', 'pred'}];
tolineplot_labels = [{'SCR', 'Predicted SCR'}];

for v = 1:length(tolineplot)
    % Average across participants
    avg  = varfun(@nanmean, pdata, 'Inputvariables', tolineplot{v},  ...
        'GroupingVariables', {'cond_plot', 'block'});

    avg.Properties.VariableNames(end) = {'mean'};

    stdev = varfun(@nanstd, pdata, 'Inputvariables', tolineplot{v},  ...
        'GroupingVariables', {'cond_plot', 'block'});

    % Calculate SE
    se = table2array(stdev(:, end))./sqrt(length(unique(pdata.sub)));
    se = array2table(se, 'VariableNames', {'se'});

    %Merge tables
    plotdat = [avg, se];

    % Init figures
    figure('Renderer', 'painters', 'Position', p.size);
    hax = axes;

    hold on


    % Loop conditions
    cond = unique(plotdat.cond_plot);
    count1 = 0;
    count2 = 0;
    for c = 1:length(cond)
        pcond = cond(c);
        pl = plotdat(strmatch(pcond, plotdat.cond_plot), :);

        if strmatch(pcond{1}(3), '+')
            color = [0.298, 0.45, 0.69];
            marker = 'd';
            style = '-';
            off = 0.05;
            legend_name = 'CS+/CSE';
            count1 = count1 + 1;
            if count1 == 1 % Legend entry only once
                visibility = 'on';
            else
                visibility = 'off';
            end
        else
            color = [0.33, 0.66, 0.41];
            marker = '^';
            style = '--';
            off = -0.2;
            legend_name = 'CS1/CS2';
            count2 = count2 + 1;
            if count2 == 1  % Legend entry only once
                visibility = 'on';
            else
                visibility = 'off';
            end
        end
        pl.block(1) = pl.block(1) + 0.075 + off;
        if length(pl.block) > 1
            pl.block(2) = pl.block(2) - 0.075 + off;
        end

        % Plot figure
        h = errorbar(pl.block, pl.mean, pl.se, 'Color', color,...
            'Marker', marker, 'MarkerSize', p.markesizelines,...
            'LineStyle', style, 'MarkerFaceColor', color, 'DisplayName',...
            legend_name, 'HandleVisibility', visibility, ...
            'LineWidth', p.linewidth);

    end

    % Make it nice
    xlim([0.5, 7.5])
    xlabel('Block', 'FontSize', p.labelfsize)
    ylabel(tolineplot_labels{v}, 'FontSize', p.labelfsize)
    legend('location', 'Northwest')
    legend boxoff
    set(gca,'FontSize', p.axisfsize)

    % Add horizontal lines between blocks
    for b = 1.5:1:6.5
        line([b, b], get(hax, 'YLim'), 'Color', 'k', 'LineStyle', ':',...
            'HandleVisibility', visibility)
    end


    % White background
    set(gcf,'color','w');

    plt_name = fullfile(p.path, [m.name '_' tolineplot_labels{v} '.png']);

    % Save plot
    print(plt_name, '-dpng', p.resolution);

    % Get filename in a cell
    allplots = [allplots, {plt_name}];
    close;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trial-wise actual vs predicted
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start from full data
fulldata = m.tabdata;

% for i = 1:length(subs)
%     s = subs(i);
%     subdat{i} = fulldata(fulldata.sub == s, :);
%
%


% Get within block trial number

% Reorder trials of all participants to match first subject structure
%
% % Get first sub structure
% subdat = {};
% subs = unique(fulldata.sub);
% newdata = [];
% for i = 1:length(subs)
%     s = subs(i);
%     subdat{i} = fulldata(fulldata.sub == s, :);
%
%     % Create a condition + block + trial column
%     blockcount = repmat({ones(1, 5)}, length(unique(subdat{i}.block)), 1);
%
%     subdat{i}.condmatch = subdat{i}.cond_plot;
%
%     for row = 1:size(subdat{i}, 1)
%         rowdat = subdat{i}(row, :);
%         switch rowdat.cond{1}
%             case 'CS-1'
%                 trial = blockcount{rowdat.block(1)-m.nofirstblock}(1);
%                 blockcount{rowdat.block(1)-m.nofirstblock}(1) = trial + 1;
%             case 'CS-2'
%                 trial = blockcount{rowdat.block(1)-m.nofirstblock}(2);
%                 blockcount{rowdat.block(1)-m.nofirstblock}(2) = trial + 1;
%             case 'CS+'
%                 trial = blockcount{rowdat.block(1)-m.nofirstblock}(3);
%                 blockcount{rowdat.block(1)-m.nofirstblock}(3) = trial + 1;
%             case 'CS-E'
%                 trial = blockcount{rowdat.block(1)-m.nofirstblock}(4);
%                 blockcount{rowdat.block(1)-m.nofirstblock}(4) = trial + 1;
%             case 'CS++'
%                 trial = blockcount{rowdat.block(1)-m.nofirstblock}(5);
%                 blockcount{rowdat.block(1)-m.nofirstblock}(5) = trial + 1;
%         end
%
%         rowdat.condmatch = {[rowdat.cond{1} '_' num2str(rowdat.block(1)) '_' ...
%             num2str(trial)]};
%         subdat{i}.condmatch(row, :) = rowdat.condmatch;
%
%     end
%     % Get the order idx based on first part
%     if i == 1
%         template = subdat{i};
%         template.plot_trials(1:height(subdat{i})) = 1:height(subdat{i});
%         subdat{i}.plot_trials(1:height(subdat{i})) = 1:height(subdat{i});
%     else
%         [tf, idx] = ismember(template.condmatch, subdat{i}.condmatch);
%         % Reorder
%         subdat{i} = subdat{i}(idx, :);
%         subdat{i}.plot_trials(1:height(subdat{i})) = 1:height(subdat{i});
%     end
%
%     % Stack
%     newdata = [newdata ; subdat{i}];
% end
%

% If trim the graph. Cleaner but misleading
%func = @(x) trimmean(x, 7);

func = @(x) nanmean(x);

% Get a single participant to use as template
subs = unique(fulldata.sub);
subdat = fulldata(fulldata.sub == subs(1), :);
template = subdat;

if m.ignoreshocks == 1
    newdata = fulldata(~strcmp(fulldata.cond, 'CS++'), :);
else
    newdata = fulldata;
end

avg_scr  = varfun(@nanmean, newdata, 'Inputvariables', 'scr',  ...
    'GroupingVariables', {'trial_within_wb', 'cond_plot', 'block', 'cond'});

pred_scr  = varfun(@nanmean, newdata, 'Inputvariables', 'pred',  ...
    'GroupingVariables', {'trial_within_wb', 'cond_plot', 'block', 'cond'});

figure('Renderer', 'painters', 'Position', p.size);
hax = axes;
ucues = unique(newdata.cond_plot);
colors = {'g', 'b'};
count_1 = 0;
count_2 = 0;
for u = 1:length(ucues)
    selected = strcmp(ucues(u), pred_scr.cond_plot);
    trials = pred_scr.trial_within_wb(selected);
    c = ucues(u);

    if  c{1}(1:3) == 'CS-'
        str = '#4C72B0';
        color1 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
        str = '#0d264f';
        color2 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
        leg = 'CS-1';
        leg2 = 'CS-2';
        count_1 = count_1 + 1;
        if count_1 == 1
            hv = 'on';
        else
            hv = 'off';
        end
    else
        str = '#c44e52';
        color1 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
        str = '#55a868';
        color2 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
        leg = 'CS+';
        leg2 = 'CS-E';
        count_2 = count_2 + 1;
        if count_2 == 1
            hv = 'on';
        else
            hv = 'off';
        end
    end
    if ~isempty(trials)
        scatter(trials, avg_scr.nanmean_scr(selected), [],  color1,...
            'DisplayName', leg, 'HandleVisibility', 'off')
        hold on
        plot(trials, pred_scr.nanmean_pred(selected), 'Color',  color1, 'linewidth', 2,...
            'DisplayName', leg, 'HandleVisibility', hv)


        hold on
        if length(trials) == 27
            d = avg_scr.nanmean_scr(selected);
            pr = pred_scr.nanmean_pred(selected);
            plot(trials(10:27), pr(10:27), 'Color', color2, 'linewidth', 2,...
                'DisplayName', leg2, 'HandleVisibility', hv)
            scatter(trials(10:27),  d(10:27), [], color2,...
                'DisplayName', leg, 'HandleVisibility', 'off')
            hold on

        end
        if length(trials) == 36
            d = avg_scr.nanmean_scr(selected);
            pr = pred_scr.nanmean_pred(selected);

            plot(trials(19:36), pr(19:36), 'Color', color2, 'linewidth', 2,...
                'DisplayName', leg2, 'HandleVisibility', hv)
            scatter(trials(19:36), d(19:36), [], color2,...
                'DisplayName', leg2, 'HandleVisibility', 'off')
            hold on

        end
    end
end


%

% White background
set(gcf,'color','w');

% Add horizontal lines between blocks
% find trials with new bloc
bstart = [1];
for b = 2:length(unique(newdata.block))
    bstart = [bstart, bstart(b-1) + 18];
end
for b = 1:length(bstart)
    line([bstart(b)-0.5, bstart(b)-0.5], get(hax, 'YLim'), 'Color', 'k', 'LineStyle', ':',...
        'HandleVisibility', 'off')
    hold on
end

% Make it nice
xlim([1, max(newdata.trial_within_wb)])
xlabel('Trials within conditions', 'FontSize', p.labelfsize)
ylabel('SCR / Predicted SCR', 'FontSize', p.labelfsize)
legend('location', 'Southeast')
legend boxoff
set(gca,'FontSize', p.axisfsize)


plt_name = fullfile(p.path, [m.name '_scrvspred.png']);

allplots = [allplots, {plt_name}];
% Save plot
print(plt_name, '-dpng', p.resolution);
close;


pred_scr  = varfun(@nanmean, newdata, 'Inputvariables', 'pred',  ...
    'GroupingVariables', {'trial_within_wb_wcs', 'cond_plot2', 'block', 'cond'});

tolineplot = [ m.tolineplot];
tolineplot_labels = [m.tolineplot_labels];

for v = 1:length(tolineplot)
    % Average across participants

    avg  = varfun(@nanmean, pdata, 'Inputvariables', tolineplot{v},  ...
        'GroupingVariables', {'trial_within_wb_wcs', 'cond_plot2', 'block', 'cond'});


    avg.Properties.VariableNames(end) = {'mean'};

    stdev = varfun(@std, pdata, 'Inputvariables', tolineplot{v},  ...
        'GroupingVariables', {'trial_within_wb_wcs', 'cond_plot2', 'block', 'cond'});

    % Calculate SE
    se = table2array(stdev(:, end))./sqrt(length(unique(pdata.sub)));
    se = array2table(se, 'VariableNames', {'se'});


figure('Renderer', 'painters', 'Position', p.size);
hax = axes;
ucues = unique(newdata.cond_plot);
colors = {'g', 'b'};
count_1 = 0;
count_2 = 0;
for u = 1:length(ucues)
    selected = strcmp(ucues(u), pred_scr.cond_plot2);
    trials = pred_scr.trial_within_wb_wcs(selected);
    c = ucues(u);

    if  c{1}(1:3) == 'CS-'
        str = '#4C72B0';
        color1 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
        str = '#0d264f';
        color2 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
        leg = 'CS-1';
        leg2 = 'CS-2';
        count_1 = count_1 + 1;
        if count_1 == 1
            hv = 'on';
        else
            hv = 'off';
        end
    else
        str = '#c44e52';
        color1 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
        str = '#55a868';
        color2 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
        leg = 'CS+';
        leg2 = 'CS-E';
        count_2 = count_2 + 1;
        if count_2 == 1
            hv = 'on';
        else
            hv = 'off';
        end
    end
    if ~isempty(trials)

        plot(trials, avg.mean(selected), 'Color',  color1, 'linewidth', 2,...
            'DisplayName', leg, 'HandleVisibility', hv)


        hold on
        if length(trials) == 27
            d = avg.mean(selected);
            plot(trials(10:27), pr(10:27), 'Color', color2, 'linewidth', 2,...
                'DisplayName', leg2, 'HandleVisibility', hv)
            hold on

        end
        if length(trials) == 36
            pr = avg.mean(selected);

            plot(trials(19:36), pr(19:36), 'Color', color2, 'linewidth', 2,...
                'DisplayName', leg2, 'HandleVisibility', hv)
            hold on

        end
    end
end


%

% White background
set(gcf,'color','w');

% Add horizontal lines between blocks
% find trials with new bloc
bstart = [1];
for b = 2:length(unique(newdata.block))
    bstart = [bstart, bstart(b-1) + 18];
end
for b = 1:length(bstart)
    line([bstart(b)-0.5, bstart(b)-0.5], get(hax, 'YLim'), 'Color', 'k', 'LineStyle', ':',...
        'HandleVisibility', 'off')
    hold on
end

% Make it nice
xlim([1, max(newdata.trial_within_wb)])
xlabel('Trials within conditions', 'FontSize', p.labelfsize)
ylabel(tolineplot_labels{v}, 'FontSize', p.labelfsize)
legend('location', 'Southeast')
legend boxoff
set(gca,'FontSize', p.axisfsize)


plt_name = fullfile(p.path, [m.name '_' tolineplot_labels{v} '.png']);

allplots = [allplots, {plt_name}];
% Save plot
print(plt_name, '-dpng', p.resolution);
close;



end

%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Trial-wise trajectories by cues
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% shocks = zeros(height(newdata), 1);
% shocks(strmatch('CS+', newdata.cond)) = 0.5;
% shocks(strmatch('CS++', newdata.cond)) = 0.5;
% newdata.shockprob = shocks;
%
% % Save data with new columns
% m.tabdata = newdata;
% writetable(newdata, fullfile(m.path, m.name, [m.name , '_data_plots.csv']))
%
%
% actual  = varfun(@mean, newdata, 'Inputvariables', 'shockprob',  ...
%     'GroupingVariables', {'plot_trials'});
%
% expect = varfun(@mean, newdata, 'Inputvariables', 'vhat',  ...
%     'GroupingVariables', {'plot_trials'});
%
%
%
% figure('Renderer', 'painters', 'Position', p.size);
% hax = axes;
%
% scatter(1:length(actual.mean_shockprob), actual.mean_shockprob,...
%     'DisplayName', 'Shock probability')
% hold on
% scatter(1:length(expect.mean_vhat), expect.mean_vhat,...
%     'DisplayName', 'Shock expectation')
% hold on
%
% % White background
% set(gcf,'color','w');
%
% % Add horizontal lines between blocks
% % find trials with new bloc
% bstart = [1];
% for b = 2:length(template.block)
%     if template.block(b) ~= template.block(b-1)
%         bstart = [bstart, template.plot_trials(b)];
%     end
% end
% for b = 1:length(bstart)
%     line([bstart(b)-0.5, bstart(b)-0.5], get(hax, 'YLim'), 'Color', 'k', 'LineStyle', ':',...
%         'HandleVisibility', 'off')
%     hold on
% end
%
% % Make it nice
% xlim([template.trial(1), template.trial(end)])
% xlabel('Trials', 'FontSize', p.labelfsize)
% ylabel('Probability / Expectation', 'FontSize', p.labelfsize)
% legend('location', 'Southeast')
% legend boxoff
% set(gca,'FontSize', p.axisfsize)
%
%
% plt_name = fullfile(p.path, [m.name '_expectvspred.png']);
%
% allplots = [allplots, {plt_name}];
% % Save plot
% print(plt_name, '-dpng', p.resolution);
% close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trial-wise all trajectories for each subject
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if m.makeindividualplot

    tolineplot = [{'scr', 'pred'}, m.tolineplot];
    tolineplot_labels = [{'SCR', 'Predicted SCR',} m.tolineplot_labels];
    mkdir(fullfile(p.path, 'fig', 'sub_plots'))

    for s = 1:length(m.data)
        sub = m.data{s}.sub;
        subdat = m.tabdata(m.tabdata.sub == sub, :);
        mkdir(fullfile(p.path, 'sub_plots', num2str(sub)))
        for v = 1:length(tolineplot)

            % Average across participants
            avg  = varfun(@mean, subdat, 'Inputvariables', tolineplot{v},  ...
                'GroupingVariables', {'plot_trials'});

            avg.Properties.VariableNames(end) = {'mean'};


            % Init figures
            figure('Renderer', 'painters', 'Position', [10, 10, 450, 300]);
            hax = axes;

            hold on


            % Loop conditions

            % Plot figure
            h = plot(avg.plot_trials, avg.mean);

            % Make it nice
            xlabel('Trial', 'FontSize', p.labelfsize-10)
            ylabel(tolineplot_labels{v}, 'FontSize', p.labelfsize-10)

            set(gca,'FontSize', p.axisfsize)

            % Add horizontal lines between blocks
            % find trials with new bloc
            bstart = [1];
            for b = 2:length(template.block)
                if template.block(b) ~= template.block(b-1)
                    bstart = [bstart, template.plot_trials(b)];
                end
            end
            for b = 1:length(bstart)
                line([bstart(b)-0.5, bstart(b)-0.5], get(hax, 'YLim'), 'Color', 'k', 'LineStyle', ':',...
                    'HandleVisibility', 'off')
                hold on
            end

            % White background
            set(gcf,'color','w');


            plt_name = fullfile(p.path, 'sub_plots', num2str(sub),...
                [num2str(sub) '_' m.name '_' ...
                tolineplot_labels{v}]);

            % Save plot
            savefig(plt_name)
            print(plt_name, '-dpng', '-r150');

            close;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Box plots for parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Response parameters
% Get values for each part
avg  = varfun(@mean, pdata, 'Inputvariables', m.resp_param,  ...
    'GroupingVariables', {'sub'});

% Put in an array
rpdat = [];
for i = 1:length(m.resp_param)
    rpdat = [rpdat, avg.(['mean_'  m.resp_param{i}])];
end

% Plot box and mean
figure('Renderer', 'painters', 'Position', p.size);
hAxes = gca;
boxplot(rpdat,'Labels', m.resp_param)

hold on

for i = 1:length(m.resp_param)
    f = scatter(ones(size(rpdat, 1), 1).*(1+(rand(size(rpdat, 1), 1)-0.5)/5) + i -1,...
        rpdat(:,i), 100, 'k','filled', 'MarkerFaceAlpha', 0.4);
    hold on
end
% Print average fit indices
plot(mean(rpdat), 'd', 'MarkerSize', p.markersizebox, 'Color', [0.33, 0.66, 0.41],...
    'MarkerFaceColor', color)

xlabel('Response model parameters', 'FontSize', p.labelfsize)
ylabel('Value', 'FontSize', p.labelfsize)
set(gca,'FontSize', p.axisfsize)
% White background
set(gcf,'color','w');

% Save plot
allplots = [allplots, {fullfile(p.path, [m.name '_ResponseParam.png'])}];
print(fullfile(p.path, [m.name '_ResponseParam.png']), '-dpng',...
    p.resolution);
close;

% Perceptual parameters
% Get values for each part
avg  = varfun(@mean, pdata, 'Inputvariables', m.perc_param,  ...
    'GroupingVariables', {'sub'});

% Put in an array
rpdat = [];
for i = 1:length(m.perc_param)
    rpdat = [rpdat, avg.(['mean_'  m.perc_param{i}])];
end

% Plot box and mean
m.plot.resp_param = figure('Renderer', 'painters', 'Position', p.size);
hAxes = gca;
boxplot(rpdat,'Labels', m.perc_param)

hold on



for i = 1:length(m.perc_param)
    f = scatter(ones(size(rpdat,1), 1).*(1+(rand(size(rpdat,1), 1)-0.5)/5) + i -1,...
        rpdat(:,i), 100, 'k','filled', 'MarkerFaceAlpha', 0.4);
    hold on
end
% Print average fit indices
plot(mean(rpdat), 'd', 'MarkerSize', p.markersizebox, 'Color', [0.33, 0.66, 0.41],...
    'MarkerFaceColor', color)

xlabel('Perceptual model parameters', 'FontSize', p.labelfsize)
ylabel('Value', 'FontSize', p.labelfsize)
set(gca,'FontSize', p.axisfsize)
% White background
set(gcf,'color','w');

% Save plot

allplots = [allplots, {fullfile(p.path,...
    [m.name '_PerceptualParam.png'])}];


print(fullfile(p.path, [m.name '_PerceptualParam.png']),...
    '-dpng', p.resolution);
close;

% Model fits
fits = {'AIC', 'BIC', 'LME', 'SSE'};
% Get values for each part
avg  = varfun(@mean, pdata, 'Inputvariables', fits,  ...
    'GroupingVariables', {'sub'});

% Put in an array
rpdat = [];
for i = 1:length(fits)
    rpdat = [rpdat, avg.(['mean_'  fits{i}])];
end

% Plot box and mean
m.plot.resp_param = figure('Renderer', 'painters', 'Position', p.size);
hAxes = gca;
boxplot(rpdat,'Labels', fits)
hBoxPlot = hAxes.Children.Children;
hBoxPlot(2).Color = 'g';

hold on

for i = 1:length(fits)
    scatter(ones(size(rpdat, 1), 1).*(1+(rand(size(rpdat, 1), 1)-0.5)/5) + i -1,...
        rpdat(:,i), 100, 'k','filled', 'MarkerFaceAlpha', 0.4);
    hold on
    text(i +0.2 , mean(rpdat(:, i)), num2str(round(mean(rpdat(:, i)), 2)),...
        'FontSize', 20)
    hold on
end
% average fit indices
plot(mean(rpdat), 'd', 'MarkerSize', p.markersizebox, 'Color', [0.33, 0.66, 0.41],...
    'MarkerFaceColor', color)

xlabel('Perceptual model parameters', 'FontSize', p.labelfsize)
ylabel('Value', 'FontSize', p.labelfsize)
set(gca,'FontSize', p.axisfsize)
% White background
set(gcf,'color','w');

allplots = [allplots, {fullfile(p.path,...
    [m.name '_ModelFits.png'])}];

% Save plot
print(fullfile(p.path, [m.name '_ModelFits.png']), '-dpng', p.resolution)
close;

% Montage
montage(allplots, 'Size', [2, ceil(length(allplots)/2)], 'BackgroundColor', 'w')
set(gcf,'color','w');
title(['Summary plots for model ' m.name])
print(fullfile(m.path, m.name, [m.name '_SUMMARY.png']), '-dpng', p.resolution)
close;

end




function fearcond_compare_models(L, path, modnames, families)

% Plots parameters
p.resolution = '-r400';  % Figure resolution

options.modelNames = modnames;
if ~isempty(families)
    options.families = families;
end
% options.families = {[1, 2, 3], [4, 5, 6]}
[posterior,out] = VBA_groupBMC(L, options) ;


print(fullfile(path, 'Model_comparison.png'),...
    '-dpng', p.resolution)
save(fullfile(path, [p.comparison_name 'VBA_model_comp']), 'posterior', 'out')
close;
figure;
L_plot = L';
boxplot(L_plot,'Labels', modnames)
hold on

for i = 1:length(modnames)
    scatter(ones(size(L_plot, 1), 1).*(1+(rand(size(L_plot, 1), 1)-0.5)/5) + i -1,...
        L_plot(:,i), 100, 'k','filled', 'MarkerFaceAlpha', 0.4);
    hold on
    text(i +0.2 , mean(L_plot(:, i)), num2str(round(mean(L_plot(:, i)), 2)),...
        'FontSize', 20)
    hold on
end

print(fullfile(path, 'Models_LMEs_box.png'),...
    '-dpng', p.resolution)

close;
end
