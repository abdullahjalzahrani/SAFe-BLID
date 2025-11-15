function [best_mask, log] = EMOGFS(X, Y, cfg, feat_names)
% EMOGFS - Enhanced Multi-Objective Genetic Feature Selection
% Binary chromosome length = number of features
[n,m] = size(X);
pop = rand(cfg.ga_pop, m) > 0.5;
fitness = -Inf(cfg.ga_pop,1);
best_mask = pop(1,:);
best_fit = -Inf;

log.generation = {};
for gen = 1:cfg.ga_gen
    % evaluate fitness of each chromosome
    for i=1:cfg.ga_pop
        mask = pop(i,:);
        if sum(mask)==0
            fitness(i) = -Inf;
            continue;
        end
        Xsub = X(:,mask);
        fitness(i) = fitness_function(Xsub, Y, cfg.alpha, cfg.beta, cfg);
    end
    % track best
    [fmax, idx] = max(fitness);
    if fmax > best_fit
        best_fit = fmax; best_mask = pop(idx,:);
    end
    % diversity measure: average bitwise std across population
    diversity = mean(std(double(pop),0,1));
    mu = cfg.mu_max * (1 - diversity / max(diversity, eps));
    % selection via softmax
    probs = softmax_stable(fitness);
    newpop = false(size(pop));
    % elitism
    [~, ord] = sort(fitness,'descend');
    for e=1:cfg.elitism
        newpop(e,:) = pop(ord(e),:);
    end
    % generate rest by tournament + two-point crossover + adaptive mutation
    ptr = cfg.elitism + 1;
    while ptr <= cfg.ga_pop
        p1 = tournament(fitness,3); p2 = tournament(fitness,3);
        parent1 = pop(p1,:); parent2 = pop(p2,:);
        child1 = parent1; child2 = parent2;
        if rand < cfg.crossover_rate
            pts = sort(randi([1,m],1,2));
            child1(pts(1):pts(2)) = parent2(pts(1):pts(2));
            child2(pts(1):pts(2)) = parent1(pts(1):pts(2));
        end
        % mutate
        child1 = mutate(child1, mu);
        child2 = mutate(child2, mu);
        newpop(ptr,:) = child1; ptr = ptr + 1;
        if ptr <= cfg.ga_pop
            newpop(ptr,:) = child2; ptr = ptr + 1;
        end
    end
    pop = newpop;
    log.generation{gen}.pop = pop;
    log.generation{gen}.best_fit = best_fit;
    log.generation{gen}.diversity = diversity;
    fprintf('GA gen %d bestfit %.4f diversity %.4f mu %.4f selected %d\n', gen, best_fit, diversity, mu, sum(best_mask));
end
end

% ---------- helpers ----------
function idx = tournament(fitness,k)
c = randi(numel(fitness),k,1);
[~,i] = max(fitness(c));
idx = c(i);
end

function out = mutate(chrom, mu)
m = numel(chrom);
out = chrom;
for b=1:m
    if rand < mu
        out(b) = ~out(b);
    end
end
end

function p = softmax_stable(x)
x(~isfinite(x)) = -1e9;
mx = max(x);
ex = exp(x - mx);
p = ex ./ sum(ex);
end
