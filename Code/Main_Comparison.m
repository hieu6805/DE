% Main driver to compare DE variants on Rastrigin function
clear; close all; clc;
addpath(genpath(pwd)); % ensure functions visible

dim = 10; lb = -5.12*ones(1,dim); ub = 5.12*ones(1,dim);
fun = @(x) Rastrigin(x);
Gmax = 1000; runs = 10;
NP = 50;
opts.NP = NP; opts.Gmax = Gmax; opts.F = 0.5; opts.CR = 0.9;

algs = {'DE\_rand\_1\_bin\_modified','DE\_jDE','DE\_JADE'};
colors = {'-b','-r','-k','-g','-m'};
conv_all = zeros(Gmax,length(algs),runs);
for a = 1:length(algs)
    fprintf('Running %s ...\n',algs{a});
    for r = 1:runs
        switch a
            case 1
                [~,~,conv] = DE_rand_1_bin_modified(fun,dim,lb,ub,opts);
            case 2
                [~,~,conv] = DE_jDE(fun,dim,lb,ub,opts);
            case 3
                [~,~,conv] = DE_JADE(fun,dim,lb,ub,opts);
        end
        conv_all(:,a,r) = conv;
    end
end

% Compute median convergence and plot all on one figure
figure('Name','DE variants convergence on Rastrigin');
hold on; grid on;

for a = 1:length(algs)
    median_conv = median(squeeze(conv_all(:,a,:)), 2);
    plot(1:Gmax, median_conv, colors{mod(a-1,numel(colors))+1}, 'LineWidth', 2);
end

legend(algs, 'Interpreter','none');
xlabel('Generation');
ylabel('Best fitness so far');

% IMPORTANT: use linear scale and set nice limits (same as convergence plot)
set(gca, 'YScale', 'linear');       
ylim([0 105]);                      
yticks(0:10:100);                   


title('Convergence comparison (median of runs)');

% Boxplot of final best values (keep linear scale for consistency with the convergence plot)
final_vals = squeeze(conv_all(end,:,:));
figure('Name','Final best fitness distribution');
boxplot(final_vals', 'Labels', algs);
set(gca, 'XTickLabel', algs, 'TickLabelInterpreter', 'none');
ylabel('Final best fitness');
ylim([0 105]);                      
yticks(0:10:100);
grid on;