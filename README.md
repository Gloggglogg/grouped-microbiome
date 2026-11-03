# grouped-microbiome

A small Python package for within-sample diversity, logistic associations, and
kernel-score tests with grouped observations. Functions accept numeric arrays
and caller-supplied group labels. The included demonstration generates every
observation randomly.

## Install and run

Requires Python 3.10 or newer.

```bash
python -m pip install .
python examples/synthetic.py
python -m unittest discover -s tests -v
```

## Arrays and alignment

Rows represent samples and must have the same order across all inputs. `y` is a
one-dimensional outcome coded zero or one, `groups` contains observed, hashable
scalar labels, and `covariates` contains numeric adjustment columns. Groups are
assumed independent; observations within a group can be dependent. At least two
groups are required for calculations using groups, and reliable inference needs
adequate independent group information.

The package retains every row and uses supplied units. Encode categories, choose
scaling, and resolve missing values before calling a function. Model columns must
have full rank, meaning that each column adds information beyond the others.

## Diversity

```python
from grouped_microbiome import alpha_diversity

diversity = alpha_diversity([[2, 2, 0], [1, 3, 4]])
```

`alpha_diversity(counts)` accepts a sample-by-taxon matrix of nonnegative counts
or weights with positive finite row totals. It returns three arrays:

- `richness`: the number of strictly positive entries in each row.
- `shannon`: minus the sum of `p * log(p)`, using natural logarithms and treating
  zero contributions as zero.
- `inverse_simpson`: `1 / sum(p**2)`.

Here `p` is each taxon's proportion of its sample total. Sequencing depth affects
observed richness. Sampling design and preprocessing remain the caller's choices.

## Logistic associations

```python
from grouped_microbiome import fit_logistic_association

result = fit_logistic_association(X, y, covariates=covariates, groups=groups)
```

`X` contains one or more predictors fitted jointly. A one-dimensional array
denotes one predictor. The function adds an intercept, followed by the `X`
columns and then the covariate columns. Supply covariates without an intercept.
The result includes term names, coefficients, standard errors, confidence
intervals, p-values, odds ratios, the covariance matrix, and diagnostics.

The model is ordinary logistic regression fitted through a binomial generalized
linear model in statsmodels. With `groups`, uncertainty uses its cluster sandwich
covariance with the small-sample correction enabled and a normal Wald reference.
With `groups=None`, uncertainty uses model-based covariance. The default
confidence level is 0.95. An odds ratio describes a one-unit increase in the
supplied predictor while holding the other model columns fixed. Standardize a
predictor before fitting to express its odds ratio per standard deviation.

The function raises an error if complete or quasi-complete separation makes
ordinary maximum likelihood estimates nonfinite, or if fitting fails. It also
rejects redundant columns and nonfinite inputs.

## Grouped kernel-score test

```python
from grouped_microbiome import kernel_test

result = kernel_test(X, y, groups=groups, covariates=covariates,
                     draws=9999, seed=42)
print(result.p_value)
```

Here `X` contains exact kernel factors: its dot product with its transpose defines
the sample-by-sample kernel. Choose the transformation that gives the intended
sample similarity before calling the function. The null mean uses Firth's
logistic likelihood with a Jeffreys penalty. The penalty supplies finite
working-mean estimates when separation prevents ordinary logistic fitting.
See [Firth's method](https://doi.org/10.1093/biomet/80.1.27).

The observed residual score and the kernel factors are projected to account for
the fitted covariates. Scores are summed within groups, and their empirical
covariance defines a Gaussian multiplier reference. This is an adaptation of the
generalized estimating equation kernel-score framework described by
[Wang and colleagues](https://doi.org/10.1002/gepi.21763), using a penalized working
mean. Each group receives one independent Gaussian multiplier. All positive
eigenvalues are retained; tiny negative values from floating-point roundoff are
recorded and clipped, and larger negative values stop the calculation.

The p-value is an asymptotic hypothesis-test approximation. Drawing its
conditional Gaussian reference by eigenvalues is algebraically exact for that
reference; finite-sample test size depends on the study design and available
independent groups. Assess calibration for the intended use. The returned
`reference` records the add-one Monte Carlo tail estimate, its resolution,
exceedance count, seed, and a 95% binomial interval for the reference tail
probability. That interval quantifies Monte Carlo uncertainty.

Lower-level functions support explicit designs and full kernels:

- `fit_logistic_null(design, y, method="firth")` requires a full-rank design
  containing an intercept in its column span. `method="ordinary"` requests
  ordinary maximum likelihood explicitly.
- `group_kernel_score(fit, groups, features=X)` accepts exact factors;
  `group_kernel_score(fit, groups, kernel=K)` accepts a symmetric positive
  semidefinite kernel.
- `gaussian_multiplier_pvalue(statistic, eigenvalues, draws=..., seed=...)`
  samples the conditional reference.
- `gower_from_profiles(profiles, metric)` constructs a Gower-centered kernel
  from square-root Jaccard distances for binary presence/absence profiles or
  square-root Bray-Curtis distances for nonnegative relative abundances summing
  to one. Use `metric="jaccard"` or `metric="braycurtis"`. Empty rows are rejected.

## Multiple tests

`adjust_pvalues(pvalues, method="bh")` applies the Benjamini-Hochberg procedure;
`method="holm"` applies Holm's procedure. The supplied vector defines the entire
set of tests being corrected. Values must be finite and between zero and one;
an empty vector returns an empty array. Benjamini-Hochberg targets the false
discovery rate under independence or suitable positive dependence. Holm controls
the probability of any false rejection within the supplied set.

## Example and checks

The synthetic example creates random count profiles, numeric covariates, binary
outcomes, and group labels. It demonstrates diversity, separate adjusted logistic
models, multiple-testing adjustment, and a kernel test after an illustrative
log-ratio transformation. Tests check numerical identities, independent reference
calculations, input failures, and reproducibility. The example's output describes
its random inputs.

## License

[MIT License](LICENSE).
