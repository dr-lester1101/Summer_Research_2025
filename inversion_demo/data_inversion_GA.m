%% INVERSION ETNA - GENETIC ALGORITHM VERSION %%
%
% Script file that performs GA inversion to invert harmonic infrasound
% observations for crater geometry using real data from Mount Etna
%
% Modified from MCMC version by Leighton Watson
% GA implementation by Daniel Spencer
% dsp65@uclive.ac.nz // daniel.spencer2007@gmail.com

clear all; clc;
cmap = get(gca,'ColorOrder');
set(0,'DefaultLineLineWidth',3);
set(0,'DefaultAxesFontSize',18);

path(pathdef)
addpath data/
addpath ../GRL2020/invOutput/
addpath ../source/resonance/
addpath ../source/SBPoperators/
addpath ../source/inv/

for i = 1:5
    figure(i); clf;
end

save_output = 0; % logical that determines if outputs are saved or not
plot_output = 1; % logical that determines if outputs are plotted

%% resonance 1D %%

% set the parameters for the resonance1d calculations 
T = 25; % total time (s)
N = 250; % number of grid points (formulas assume even N)
dt = T/N; % time step (s)
Nyquist = 1/(2*dt); % Nyquist frequency (Hz)
Nf = N/2+1; % number of frequency samples
freq = [0 Nyquist]; % frequency range (Hz)
discrParams = [T N Nf Nyquist dt]; % save parameters into array

craterTemp = 100; % crater temperature
atmoTemp = 0; % atmospheric temperature
temp = [craterTemp, atmoTemp]; 

order = 4; % order of numerical scheme (4, 6 or 8)
style = 'baffled piston'; % acoustic radiation model ('monopole' or ' baffled piston')
M = problemParametersInv(craterTemp,atmoTemp); % problem parameters required for resonance1d

%% data %%

dataStr = 'Etna2018Phase3';
datafile = strcat(dataStr,'.mat');
load(datafile); % load data
data_freq = [dataF, dataAmp]; % format data

filterband = [0.25 4.8]; % frequency band to filter
filterorder = 4; % order of butterworth filter
Fs = 10; % sampling frequency
filterProps = [filterband, filterorder, Fs]; % filter properties - same as for data

%% inversion parameters %%

%%% GA PARAMETERS %%%
popSize = 50; % population size (number of individuals)
nGen = 100; % number of generations
mutationRate = 0.1; % mutation probability (0-1)
crossoverRate = 0.8; % crossover probability (0-1)
eliteCount = 2; % number of elite individuals to preserve
tournamentSize = 3; % tournament selection size

freqLim = 3; % high cut frequency limit for misfit function (Hz)

%%% geometry parameters %%%
geomFlag = 1; % invert for geometry (boolean, 0 = no, 1 = yes)
geomR0 = 100; % radius of initial cylinder
geomDepth = 200; % depth 
geomParams = [geomDepth geomR0 geomR0 geomR0 geomR0 geomR0]; % first value is depth, other values are radius points that are equally spaced
geomLowerBnds = [50 80 1 1 1 1];
geomUpperBnds = [300 120 120 120 120 120];
nx = length(geomParams)-1; % number of geometry parameters

%%% source parameters %%%
srcFlag = 0; %invert for source (boolean, 0 = no, 1 = yes)
srcParams = 0.3;
srcStyle = 'Brune';
srcUpperBnds = [5];
srcLowerBnds = [0.01];

%%% format parameters %%%
if geomFlag && srcFlag
    upperBnds = [geomUpperBnds srcUpperBnds];
    lowerBnds = [geomLowerBnds srcLowerBnds];
    params = [geomParams srcParams];
    
elseif geomFlag
    upperBnds = geomUpperBnds;
    lowerBnds = geomLowerBnds;
    params = geomParams;
    
elseif srcFlag
    upperBnds = srcUpperBnds;
    lowerBnds = srcLowerBnds;
    params = srcParams;
   
end


%% GA inversion %%

fprintf('======================================\n');
fprintf('GENETIC ALGORITHM INVERSION\n');
fprintf('======================================\n');
fprintf('Population Size: %d\n', popSize);
fprintf('Generations: %d\n', nGen);
fprintf('Mutation Rate: %.2f\n', mutationRate);
fprintf('Crossover Rate: %.2f\n', crossoverRate);
fprintf('Elite Count: %d\n', eliteCount);
fprintf('======================================\n\n');

tic % start timing
[population, misfit, simSpec, f, bestIndiv] = ga_spec(popSize, nGen, ... % perform GA inversion
    mutationRate, crossoverRate, eliteCount, tournamentSize, ...
    geomParams, geomFlag, srcParams, srcFlag, srcStyle,...
    lowerBnds, upperBnds, discrParams, temp, ...
    filterProps, data_freq, freqLim);
toc % finish timing
disp(['Elapsed time is ',num2str(toc/60), ' minutes.']); % display timing in minutes

% Convert outputs for compatibility with plotting code
x = bestIndiv; % best individual per generation (for plotting)
count = nGen; % total generations (for plotting)

%% post-process outputs %%

fprintf('\n======================================\n');
fprintf('POST-PROCESSING RESULTS\n');
fprintf('======================================\n');

% format spectra
spec.int = simSpec; % initial spectra (from first generation)
spec.fin = simSpec; % final spectra (from best individual)

% For GA, average the best individuals from last 10 generations
lastGen = min(10, nGen); % last 10 generations (or fewer if nGen < 10)
x_final = bestIndiv(end-lastGen+1:end, :);

fprintf('Averaging last %d generations for final estimate...\n', lastGen);

% Calculate mean parameters
x_mean_norm = mean(x_final, 1);
params_mean = lowerBnds + x_mean_norm .* (upperBnds - lowerBnds);

fprintf('Mean parameters from last %d generations:\n', lastGen);
for i = 1:length(params_mean)
    fprintf('  Param %d: %.2f [%.2f, %.2f]\n', i, params_mean(i), lowerBnds(i), upperBnds(i));
end

% Extract mean geometry parameters
if geomFlag
    geomParams_mean = params_mean(1:length(geomParams));
else
    geomParams_mean = geomParams;
end

if srcFlag
    srcParams_mean = params_mean(end);
else
    srcParams_mean = srcParams;
end

% Generate mean spectrum
fprintf('Computing mean spectrum from averaged parameters...\n');
shape_mean = geomFunction(geomParams_mean);
depth_mean = shape_mean(1,1);

% Compute source
[S_mean, ~, ~, ~] = sourceFunction(1, srcParams_mean, srcStyle, discrParams);

% Compute resonance
style = 'baffled piston';
order = 4;
res_mean = resonance1d(shape_mean, depth_mean, freq, Nf, style, order, M);

% Convolve with source
sim_P_mean = res_mean.P(1:N/2+1) .* S_mean(1:N/2+1);

% Convert to time domain
sim_pFreq_mean = [sim_P_mean; conj(sim_P_mean(end-1:-1:2))];
sim_pFreq_mean(N/2+1) = real(sim_pFreq_mean(N/2+1));
sim_pTime_mean = ifft(sim_pFreq_mean, 'symmetric') / dt;

% Apply filtering
sim_pTimeFilt_mean = bandpass_butterworth(sim_pTime_mean, filterband, Fs, filterorder);

% Convert back to frequency domain
L = length(sim_pTimeFilt_mean);
NFFT = 2^nextpow2(L);
sim_pFreqFilt_mean = fft(sim_pTimeFilt_mean, NFFT) / L;
spec.mean = abs(sim_pFreqFilt_mean(1:NFFT/2+1));
spec.mean = spec.mean / max(spec.mean);

fprintf('Final best misfit: %.6f\n', misfit(end));
fprintf('======================================\n\n');

%% save outputs %%

pathname = strcat(pwd,'/invOutput'); % directory to save outputs
filename = strcat('DataInvOut_GA','_',dataStr,'_',... % file name
    'pop',num2str(popSize),'_gen',num2str(nGen),'_',...
    'R0',num2str(geomR0),'m_','D',num2str(geomDepth),'m_',...
    'T',num2str(craterTemp),'C_',...
    'freqLim',num2str(freqLim),'_nx',num2str(nx));
matfile = fullfile(pathname,filename); % path and file for saving outputs

if save_output == 1
    fprintf('Saving outputs to: %s.mat\n', filename);
    save(matfile,'x','misfit','spec','f','count','bestIndiv','population',...
        'params','geomParams','srcParams','geomFlag','srcFlag','srcStyle',...
        'lowerBnds','upperBnds','popSize','nGen','mutationRate','crossoverRate',...
        'eliteCount','tournamentSize','freqLim',...
        'discrParams','M','filterProps');
    fprintf('Outputs saved successfully.\n');
end

%% plot outputs %%

if plot_output == 1
    
    fprintf('Generating plots...\n');
    
    %%% crater geometry %%%
    
    figure(1);
    
    %%% Initial estimate of crater geometry
    shapeI = geomFunction(geomParams);
    depthI = shapeI(1,1);
    plot(shapeI(:,2), shapeI(:,1),'Color',cmap(1,:),'LineWidth',2,'DisplayName','Initial'); hold on;
    set(gca,'YDir','Reverse'); 
    xlabel('Radius (m)'); 
    ylabel('Depth (m)');
    plot(-shapeI(:,2), shapeI(:,1),'Color',cmap(1,:),'LineWidth',2,'HandleVisibility','off');
    plot([-shapeI(1,2) shapeI(1,2)],[depthI depthI],'Color',cmap(1,:),'LineWidth',2,'HandleVisibility','off');
    
    %%% Mean estimate of crater geometry
    shapeF = geomFunction(geomParams_mean);
    depthF = shapeF(1,1);
    plot(shapeF(:,2), shapeF(:,1),'Color',cmap(2,:),'LineWidth',2,'DisplayName','Mean (Last 10 Gen)');
    plot(-shapeF(:,2), shapeF(:,1),'Color',cmap(2,:),'LineWidth',2,'HandleVisibility','off');
    plot([-shapeF(1,2) shapeF(1,2)],[depthF depthF],'Color',cmap(2,:),'LineWidth',2,'HandleVisibility','off');
    
    %%% Best individual geometry
    params_best = lowerBnds + bestIndiv(end,:) .* (upperBnds - lowerBnds);
    geomParams_best = params_best(1:length(geomParams));
    shapeB = geomFunction(geomParams_best);
    depthB = shapeB(1,1);
    plot(shapeB(:,2), shapeB(:,1),'Color',cmap(3,:),'LineWidth',2,'LineStyle','--','DisplayName','Best Individual');
    plot(-shapeB(:,2), shapeB(:,1),'Color',cmap(3,:),'LineWidth',2,'LineStyle','--','HandleVisibility','off');
    plot([-shapeB(1,2) shapeB(1,2)],[depthB depthB],'Color',cmap(3,:),'LineWidth',2,'LineStyle','--','HandleVisibility','off');
    
    legend('Location','best');
    title('Crater Geometry Estimates');
    grid on;

    %%% spectra %%%

    figure(2);
    
    %%% Data
    plot(dataF, abs(dataAmp)./max(abs(dataAmp)),'k','LineWidth',3,'DisplayName','Observed Data');
    hold on; 
    xlim([0 freqLim]);
    xlabel('Frequency (Hz)');
    ylabel('Normalized Amplitude Spectra');
    
    %%% Final estimate of spectra (best individual)
    plot(f(1:N/2+1), abs(spec.fin(1:N/2+1)),'Color',cmap(2,:),'LineWidth',2,'DisplayName','Best Individual');
    
    %%% Mean estimate of spectra
    plot(f(1:N/2+1), abs(spec.mean(1:N/2+1)),'Color',cmap(3,:),'LineWidth',2,'DisplayName','Mean (Last 10 Gen)');
    
    legend('Location','best');
    title('Amplitude Spectra Comparison');
    grid on;

    %%% misfit evolution %%%
    
    figure(3);
    plot(1:nGen, misfit,'Color',cmap(1,:),'LineWidth',2); 
    hold on;
    ylabel('Best Misfit per Generation');
    xlabel('Generation Number');
    xlim([1 nGen]);
    title('GA Convergence History');
    grid on;
    
    % Mark where we start averaging (last 10 generations)
    xline(nGen - lastGen + 1, '--r', 'LineWidth', 1.5, 'DisplayName', 'Averaging Window');
    legend('Location','best');

    %%% parameter evolution (best individual) %%%
    
    figure(4);
    k = length(params);
    for i = 1:k
        % Scale best individual back to physical values
        paramPlot = lowerBnds(i) + bestIndiv(:,i) .* (upperBnds(i) - lowerBnds(i));
        
        subplot(2, ceil(k/2), i); 
        hold on; 
        box on;
        plot(1:nGen, paramPlot, 'Color', cmap(1,:), 'LineWidth', 2);
        xlabel('Generation');
        ylabel(sprintf('Parameter %d', i));
        yline(lowerBnds(i), '--r', 'LineWidth', 1);
        yline(upperBnds(i), '--r', 'LineWidth', 1);
        ylim([lowerBnds(i) upperBnds(i)]);
        title(sprintf('Parameter %d Evolution', i));
        grid on;
        
        % Mark averaging window
        xline(nGen - lastGen + 1, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
    end
    sgtitle('Parameter Evolution Over Generations');
    
    %%% parameter histograms (last N generations) %%%
    
    figure(5);
    for i = 1:k
        % Scale to physical values
        paramPlot = lowerBnds(i) + x_final(:,i) .* (upperBnds(i) - lowerBnds(i));
        
        subplot(2, ceil(k/2), i); 
        hold on; 
        box on;
        histogram(paramPlot, 10, 'FaceColor', cmap(1,:), 'EdgeColor', 'k');
        xlabel(sprintf('Parameter %d', i));
        ylabel('Frequency');
        xline(lowerBnds(i), '--r', 'LineWidth', 1.5, 'DisplayName', 'Bounds');
        xline(upperBnds(i), '--r', 'LineWidth', 1.5, 'HandleVisibility', 'off');
        xline(params_mean(i), '-g', 'LineWidth', 2, 'DisplayName', 'Mean');
        xlim([lowerBnds(i) upperBnds(i)]);
        title(sprintf('Parameter %d Distribution (Last %d Gen)', i, lastGen));
        grid on;
        if i == 1
            legend('Location','best');
        end
    end
    sgtitle(sprintf('Parameter Distributions (Last %d Generations)', lastGen));
    
    fprintf('Plotting complete.\n');
    
end

fprintf('\n======================================\n');
fprintf('GA INVERSION COMPLETE\n');
fprintf('======================================\n');
