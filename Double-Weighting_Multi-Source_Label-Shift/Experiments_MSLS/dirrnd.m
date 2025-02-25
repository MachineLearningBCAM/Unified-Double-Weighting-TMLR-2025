function dirichlet_sample = dirrnd(alpha, num_samples)
    % Generate samples from a Dirichlet distribution
    % alpha: concentration parameters (1 x num_classes)
    % num_samples: number of samples to generate
    % dirichlet_sample: (num_samples x num_classes)

    num_classes = length(alpha);
    dirichlet_sample = gamrnd(repmat(alpha, num_samples, 1), 1, num_samples, num_classes);
    dirichlet_sample = dirichlet_sample ./ sum(dirichlet_sample, 2);
end