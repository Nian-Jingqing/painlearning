function [logp, yhat, res] = hab(r, infStates, ptrans)
% Calculates the log-probability of log-reaction times y (in units of log-ms) according to the
% linear log-RT model developed with Louise Marshall and Sven Bestmann
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2014-2016 Christoph Mathys, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform parameters to their native space
be0  = ptrans(1);
be1  = ptrans(2);
ze   = exp(ptrans(3));

% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
% irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Weed irregular trials out from responses and inputs
y = r.y(:,1);
y(r.irr) = [];

u = r.u(:,1);

nshocks = zeros(length(u), 1);
for i = 1:length(u)
    nshocks(i) = sum(u(1:i));
end
% nshocks = (nshocks-min(nshocks))/(max(nshocks)-min(nshocks));

nshocks(r.irr) = [];
u(r.irr) = [];



% Calculate predicted scr
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
predscr = be1.*-log(nshocks+1);

% Calculate log-probabilities for non-irregular trials
% Note: 8*atan(1) == 2*pi (this is used to guard against
% errors resulting from having used pi as a variable).
reg = ~ismember(1:n,r.irr);
logp(reg) = -1/2 .* log(8 * atan(1) .* ze) - (y - predscr).^2 ./ (2 .* ze);
yhat(reg) = predscr;
res(reg) = y - predscr;

return;
