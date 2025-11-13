%% Helper Function for data_inversion_GA.m %%
%
% GA_SPEC - Genetic Algorithm implementation
%
%
%
%
% Written by Daniel Spencer
% 13th November 2024
% dsp65@uclive.ac.nz // daniel.spencer2007@gmail.com

function [population, misfit, simSpec, f, bestIndiv] = ga_spec(popSize, nGen,
    mutationRate, crossoverRate, eliteCount, tournamentSize,
    geomParams, geomFlag, srcParams, srcFlag, srcStyle,
    lowerBnds, upperBnds, discrParams, temp,
    filterProps, data_freq, freqLim)
% INPUTS:
%   popSize         - population size (number of individuals)
%   nGen            - number of generations
%   mutationRate    - probability of mutation (0-1)
%   crossoverRate   - probability of crossover (0-1)
%   eliteCount      - number of elite individuals to preserve
%   tournamentSize  - size of tournament for selection
%   geomParams      - initial geometry parameters [depth, r1, r2, ...]
%   geomFlag        - boolean, invert for geometry (1) or not (0)
%   srcParams       - source parameters
%   srcFlag         - boolean, invert for source (1) or not (0)
%   srcStyle        - source model style (e.g., 'Brune')
%   lowerBnds       - lower bounds for parameters
%   upperBnds       - upper bounds for parameters
%   discrParams     - discretization parameters [T N Nf Nyquist dt]
%   temp            - temperature [crater, atmosphere]
%   filterProps     - filter properties [band, order, Fs]
%   data_freq       - observed data [frequency, amplitude]
%   freqLim         - high frequency limit for misfit calculation
%
% OUTPUTS:
%   population      - final population (normalized parameters in [0,1])
%   misfit          - best misfit per generation
%   simSpec         - simulated spectrum from best individual
%   f               - frequency vector
%   bestIndiv       - best individual per generation (normalized)
%
% Written by Daniel Spencer
% Modified from MCMC version by Leighton Watson

nParamsGeom = 0;
nParamsSrc = 0;

if geomFlag
  nParamsGeom = length(geomParams);
end

if srcFlag
  nParamsSrc = length(srcParams);
end

nParams = nParamsGeom + nParamsSrc;

%Initialise population randomly in [0,1]
population = rand(popSize, nParams);

% Storage arrays
misfit = zeros(nGen, 1);
bestIndiv = zeros(nGen, nParams);
fitness = zeros(popSize, 1);

% Extract discretization parameters
N = discrParams(2);
Nf = discrParams(3);
Nyquist = discrParams(4);
dt = discrParams(5);
freq = [0 Nyquist];

% Extract temperature parameters
craterTemp = temp(1);
atmoTemp = temp(2);
M = problemParametersInv(craterTemp, atmoTemp);

% Extract filter parameters
filterband = filterProps(1:2);
filterorder = filterProps(3);
Fs = filterProps(4);

% Extract data
dataF = data_freq(:,1);
dataAmp = data_freq(:,2);

% Normalize data amplitude
dataNorm = abs(dataAmp) / max(abs(dataAmp));

%% GA Loop

fprintf('Starting GA inversion with %d individuals for %d generations...\n', popSize, nGen);

for gen = 1:nGen
    
    %% 1. EVALUATE FITNESS (calculate misfit for each individual)
    
    for i = 1:popSize
        
        % Scale parameters from [0,1] to physical bounds
        params_scaled = lowerBnds + population(i,:) .* (upperBnds - lowerBnds);
        
        % Extract geometry and source parameters
        if geomFlag
            geomParams_current = params_scaled(1:nParamsGeom);
        else
            geomParams_current = geomParams;
        end
        
        if srcFlag
            srcParams_current = params_scaled(end);
        else
            srcParams_current = srcParams;
        end
        
        % Run forward model
        try
            % Generate crater shape
            shape = geomFunction(geomParams_current);
            depth = shape(1,1);
            
            % Check for valid geometry
            if depth <= 0 || any(shape(:,2) <= 0)
                fitness(i) = 1e10; % penalty for invalid geometry
                continue;
            end
            
            % Compute source function
            [S, ~, ~, ~] = sourceFunction(1, srcParams_current, srcStyle, discrParams);
            
            % Compute acoustic transfer function
            style = 'baffled piston';
            order = 4;
            res = resonance1d(shape, depth, freq, Nf, style, order, M);
            
            % Convolve with source and normalize
            sim_P = res.P(1:N/2+1) .* S(1:N/2+1);
            
            % Convert to time domain
            sim_pFreq = [sim_P; conj(sim_P(end-1:-1:2))];
            sim_pFreq(N/2+1) = real(sim_pFreq(N/2+1));
            sim_pTime = ifft(sim_pFreq, 'symmetric') / dt;
            
            % Apply filtering
            sim_pTimeFilt = bandpass_butterworth(sim_pTime, filterband, Fs, filterorder);
            
            % Convert back to frequency domain
            L = length(sim_pTimeFilt);
            NFFT = 2^nextpow2(L);
            sim_pFreqFilt = fft(sim_pTimeFilt, NFFT) / L;
            sim_fFilt = Fs/2 * linspace(0, 1, NFFT/2+1);
            
            % Normalize simulated spectrum
            sim_pFreqAbs = abs(sim_pFreqFilt(1:NFFT/2+1));
            simNorm = sim_pFreqAbs / max(sim_pFreqAbs);
            
            % Interpolate to data frequency grid
            simInterp = interp1(sim_fFilt, simNorm, dataF, 'linear', 0);
            
            % Calculate misfit (only up to freqLim)
            idx = dataF <= freqLim;
            fitness(i) = sum((dataNorm(idx) - simInterp(idx)).^2);
            
        catch
            % If forward model fails, assign high misfit
            fitness(i) = 1e10;
        end
    end
    
    %% 2. STORE BEST INDIVIDUAL
    
    [misfit(gen), bestIdx] = min(fitness);
    bestIndiv(gen,:) = population(bestIdx,:);
    
    %% 3. SELECTION
    
    newPopulation = zeros(popSize, nParams);
    
    % Elitism: keep best individuals unchanged
    [~, sortedIdx] = sort(fitness);
    newPopulation(1:eliteCount,:) = population(sortedIdx(1:eliteCount),:);
    
    % Tournament selection for remaining individuals
    for i = eliteCount+1:popSize
        % Randomly select tournamentSize individuals
        tournamentIdx = randperm(popSize, tournamentSize);
        tournamentFitness = fitness(tournamentIdx);
        
        % Select the best from tournament
        [~, winnerLocalIdx] = min(tournamentFitness);
        winnerIdx = tournamentIdx(winnerLocalIdx);
        
        newPopulation(i,:) = population(winnerIdx,:);
    end
    
    %% 4. CROSSOVER
    
    for i = eliteCount+1:2:popSize-1
        if rand < crossoverRate
            % Single-point crossover
            if nParams > 1
                crossPoint = randi([1, nParams-1]);
                
                % Swap genetic material after crossover point
                temp = newPopulation(i, crossPoint+1:end);
                newPopulation(i, crossPoint+1:end) = newPopulation(i+1, crossPoint+1:end);
                newPopulation(i+1, crossPoint+1:end) = temp;
            end
        end
    end
    
    %% 5. MUTATION
    
    for i = eliteCount+1:popSize
        for j = 1:nParams
            if rand < mutationRate
                % Uniform random mutation
                newPopulation(i,j) = rand;
                
                % Alternative: Gaussian mutation around current value
                % newPopulation(i,j) = newPopulation(i,j) + 0.1*randn;
                % newPopulation(i,j) = max(0, min(1, newPopulation(i,j))); % clamp to [0,1]
            end
        end
    end
    
    % Update population
    population = newPopulation;
    
    %% DISPLAY PROGRESS
    
    if mod(gen, 10) == 0 || gen == 1
        fprintf('Generation %d/%d | Best Misfit: %.6f | Mean Fitness: %.6f\n', ...
            gen, nGen, misfit(gen), mean(fitness));
    end
end

%% POST-PROCESSING: Generate outputs from best individual

fprintf('\nGA inversion complete. Generating final outputs...\n');

% Get best individual from last generation
params_final = lowerBnds + bestIndiv(end,:) .* (upperBnds - lowerBnds);

if geomFlag
    geomParams_final = params_final(1:nParamsGeom);
else
    geomParams_final = geomParams;
end

if srcFlag
    srcParams_final = params_final(end);
else
    srcParams_final = srcParams;
end

% Generate final spectrum
shape = geomFunction(geomParams_final);
depth = shape(1,1);

[S, ~, ~, ~] = sourceFunction(1, srcParams_final, srcStyle, discrParams);

style = 'baffled piston';
order = 4;
res = resonance1d(shape, depth, freq, Nf, style, order, M);

sim_P = res.P(1:N/2+1) .* S(1:N/2+1);

% Convert to time domain
sim_pFreq = [sim_P; conj(sim_P(end-1:-1:2))];
sim_pFreq(N/2+1) = real(sim_pFreq(N/2+1));
sim_pTime = ifft(sim_pFreq, 'symmetric') / dt;

% Apply filtering
sim_pTimeFilt = bandpass_butterworth(sim_pTime, filterband, Fs, filterorder);

% Convert back to frequency domain
L = length(sim_pTimeFilt);
NFFT = 2^nextpow2(L);
sim_pFreqFilt = fft(sim_pTimeFilt, NFFT) / L;
f = Fs/2 * linspace(0, 1, NFFT/2+1);

% Normalize
simSpec = abs(sim_pFreqFilt(1:NFFT/2+1));
simSpec = simSpec / max(simSpec);

fprintf('Final best misfit: %.6f\n', misfit(end));
fprintf('Best parameters: ');
fprintf('%.2f ', params_final);
fprintf('\n');

end





















