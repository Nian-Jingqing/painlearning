clc; clear;


% eeg path
dpath = '/media/mp/lxhdd/2020_painlearning/derivatives/';

% Load non-eeg data table
bdata = readtable('/media/mp/lxhdd/2020_painlearning/derivatives/task-fearcond_alldata.csv');

% keep only shock trials
bdatans = bdata(~isnan(bdata.rating), :);

% Get parts
parts = unique(bdata.sub);

% Number of bootstraps (string)
nboots = '5000';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for p = 1:length(parts)
    s = parts(p);
    sdat = bdatans(bdatans.sub == s, :);

    rating{p, 1} = zscore(log(sdat.rating+1));
    nfrnorm{p, 1} = zscore(log(sdat.nfr+1));
    erpamp{p, 1} = zscore(sdat.erp_amp);
    vhat{p, 1} = zscore(sdat.vhat);
    sa1hat{p, 1} = zscore(sdat.sa1hat);
    erp{p, 1} = zscore(sdat.erp_amp);

    % Get single trial erp files
    file = fullfile(dpath, ['sub-' num2str(s)], 'eeg',...
                    ['s' num2str(s) '_task-fearcond_shock_singletrials-epo.fif']);


    % 468 trials x 70 channels x 600 time points
    erps = fiff_read_epochs(file);
    sdat = bdata(bdata.sub == s, :);

    % Drop EOG and status channel (67:70)
    dat = erps.data(:, 1:66, :, :);


%     % Load TF
%     file = fullfile(dpath, ['s' num2str(s)], 'tfr2',...
%                 ['s' num2str(s) '_induced_strials-tfr_data.npy']);
%     tfrs = readNPY(file);
%
%     % Remove unused trials
%     tfrs = tfrs(~isnan(sdat.rating), :, :, :);

    % Stack across participants
    if p ~= 1
       alldata = cat(4, alldata, dat);
%        alldata_tf = cat(5, alldata_tf, tfrs);
    else
       alldata = dat;
%        alldata_tf = tfrs;
    end
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run non-ERPs multilevel mediation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = 'vhat';
M = 'nfrnorm';
Y = 'rating';


[~, stats] = mediation(vhat, rating, nfrnorm,  'noverbose',...
                           'boot', 'bootsamples', str2double(nboots),...
                            'noverbose', 'doCIs');

out_name = ['MultiMediation_X_' X '_Y_' Y '_M_' M '_'...
             nboots 'bootsamples'];

betas = stats.beta;
betas_se = stats.ste;
pvals = stats.p;
save(fullfile('/media/mp/lxhdd/zoeydatapain/derivatives/', out_name),...
              'betas', 'betas_se', 'pvals')



X = 'sa1hat';
M = 'erp';
Y = 'rating';


[paths, stats] = mediation(sa1hat, rating, erp,  'noverbose',...
                           'boot', 'bootsamples', str2double(nboots),...
                           'noverbose');

out_name = ['MultiMediation_X_' X '_Y_' Y '_M_' M '_'...
             nboots 'bootsamples'];

betas = stats.beta;
betas_se = stats.ste;
pvals = stats.p;
save(fullfile('/media/mp/lxhdd/zoeydatapain/derivatives/', out_name),...
              'betas', 'betas_se', 'pvals')

close('all')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run mediation using ERPs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Variables to use
Xs = {'sa1hat'};
Ys = {'rating'};
Ms = {'eegdat'};

meds = {{'sa1hat', 'rating', 'eegdat'},...
        {'sa1hat', 'nfrnorm', 'eegdat'},...
        {'eegdat', 'rating', 'sa1hat'},...
        {'eegdat', 'nfrnorm', 'sa1hat'},...
        {'vhat', 'rating', 'eegdat'}...
        };

nboots = 5000;
 for m = 1:length(meds)
clear betas
clear betas_se
clear pvals
clear sbetas
%


    X = meds{m}{1};
    Y = meds{m}{2};
    M = meds{m}{3};


    nchans = size(alldata, 2);
    ntimes = size(alldata, 3);
    total = nchans*ntimes;
    count = 1;
    res = struct();
    for c = 1:nchans % Loop channels
        for t = 1:ntimes % Loop time points
            for p = 1:length(parts)
                % trials one chan, one time point, all parts
    %             eegdat{p, 1} = alldata(:, c, t, p) - mean(alldata(:, c, t, p));
                  eegdat{p, 1} = zscore(alldata(:, c, t, p));
%                   eegdat{p, 1} = alldata(:, c, t, p);

            end
            clc
            disp(['Running mediation - ' num2str(m) ' out of ' num2str(length(meds)) ' - ' num2str(count/total*100) ' % complete'])
            % Run multilevel mediation at specific chan and timepoint
            if nboots == 0
                eval(['[paths, stats] = mediation(' X ', ' Y ', '  M...
                      ', ''noverbose'');']);
            else
               eval(['[paths, stats] = mediation(' X ', ' Y ', '  M ...
                     ', '  '''noverbose''' ', ' '''boot''' ',' '''bootsamples''' ',' num2str(nboots) ');']);
            end
            % put betas and pvals in an array
            sbetas(:, :, c, t) = stats.paths;
            betas(c, t, :) = stats.beta;
            betas_se(c, t, :) = stats.ste;
            pvals(c, t, :) = stats.p;
            count = count + 1;
        end
    end
    out_name = ['MassMediation_X_' X '_Y_' Y '_M_' M '_nboots' num2str(nboots) '.mat'];


    save(fullfile('/media/mp/lxhdd/zoeydatapain/derivatives/', out_name), 'betas', 'betas_se', 'pvals', 'sbetas')

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Run mediation using TFR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Variables to use

meds = {{'sa1hat', 'rating', 'eegdat'},...
        {'sa1hat', 'nfrnorm', 'eegdat'},...
        {'eegdat', 'rating', 'sa1hat'},...
        {'eegdat', 'nfrnorm', 'sa1hat'},...
        {'vhat', 'rating', 'eegdat'}...
        };

clear betas
clear betas_se
clear pvals

nchans = size(alldata_tf, 2);
freqs = size(alldata_tf, 3);
ntimes = size(alldata_tf, 4);
total = nchans*ntimes*freqs;
count = 1;
res = struct();


for m = 1:length(meds)
    X = meds{m}{1};
    Y = meds{m}{2};
    M = meds{m}{3};


for c = 1:nchans % Loop channels
    for t = 1:ntimes % Loop time points
        for f = 1:freqs
            for p = 1:length(parts)
                % trials one chan, one time point, all parts
                eegdat{p, 1} = zscore(alldata_tf(:, c, f, t, p));
            end
            clc
            disp(['Running mediation - ' num2str(count/total*100) ' % complete'])
            % Run multilevel mediation at specific chan and timepoint
            eval(['[paths, stats] = mediation(' X ', ' Y ', '  M...
                  ',  ''noverbose'');']);
            % put betas and pvals in an array
            betas(c, t, :) = stats.mean;
            betas_se(c, t, :) = stats.ste;
            pvals(c, t, :) = stats.p;
            count = count + 1;
            end
    end
end
out_name = ['MassMediation_X_' X '_Y_' Y '_M_' M '_noboots_TFR'];


save(fullfile('/media/mp/lxhdd/zoeydatapain/derivatives/', out_name), 'betas', 'betas_se', 'pvals')

end
