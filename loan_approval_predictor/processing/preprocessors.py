from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy import optimize


class GaussianTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, method='log', standardize=True):
        self.columns = columns
        self.method = method
        self.standardize = standardize


    def fit(self, X, y=None):
        method = self.method.lower()
        valid = {'log', 'boxcox', 'yeo-johnson'}
        if method not in valid:
            raise ValueError(f"method must be one of {valid}, got {self.method!r}")

        fit_dispatch = {
            'log': self._fit_log,
            'boxcox': self._fit_boxcox,
            'yeo-johnson': self._fit_yeo_johnson,
        }
        transform_dispatch = {
            'log': self._log_transform,
            'boxcox': self._boxcox_transform,
            'yeo-johnson': self._yeo_johnson_transform,
        }

        self.method_ = method
        self.lambdas_ = [None] * len(self.columns)
        self.means_ = {}
        self.stds_ = {}

        for i, col in enumerate(self.columns):
            x = X[col].to_numpy(dtype=float)
            self.lambdas_[i] = fit_dispatch[method](x)
            if self.standardize:
                t = transform_dispatch[method](x, self.lambdas_[i])
                self.means_[col] = t.mean()
                std = t.std()
                self.stds_[col] = std if std > 0 else 1.0
        return self
    

    def transform(self, X):
        transform_dispatch = {
            'log': self._log_transform,
            'boxcox': self._boxcox_transform,
            'yeo-johnson': self._yeo_johnson_transform,
        }
        new_X = X.copy()
        for i, col in enumerate(self.columns):
            x = X[col].to_numpy(dtype=float)
            t = transform_dispatch[self.method_](x, self.lambdas_[i])
            if self.standardize:
                t = (t - self.means_[col]) / self.stds_[col]
            new_X[col] = t
        return new_X


    def _fit_log(self, X):
        pass


    def _fit_boxcox(self, X):
        if np.any(X <= 0):
            raise ValueError("Box-Cox transformation requires strictly positive values.")
        
        def boxcox_lambda(opt_lambda, x):
            ln_x = np.log(x)
            transformed_x = self._boxcox_transform(x, opt_lambda)
            variance = np.var(transformed_x)
            # Calculating the log-likelihood for the Box-Cox transformation
            likelihood = -(x.shape[0]/2) * np.log(variance) + (opt_lambda-1) * np.sum(ln_x)
            return -likelihood

        return optimize.brent(boxcox_lambda, brack=(-2, 2), args=(X,))

    
    def _fit_yeo_johnson(self, X):
        def yeo_johnson_lambda(opt_lambda, x):
            transformed_x = self._yeo_johnson_transform(x, opt_lambda)
            variance = np.var(transformed_x)

            # Calculating the log-likelihood for the Yeo-Johnson transformation
            first_term_loss = -(x.shape[0] / 2) * np.log(variance)
            jacobian = (opt_lambda - 1) * np.sum(np.sign(x) * np.log1p(np.abs(x)))
            likelihood = first_term_loss + jacobian
            return -likelihood
        
        return optimize.brent(yeo_johnson_lambda, brack=(-2, 2), args=(X,))


    def _log_transform(self, X, lambda_=None):
        return np.log(X + 1)


    def _boxcox_transform(self, X, lambda_):
        # Check if X is strictly positive
        if np.any(X <= 0):
            raise ValueError("Box-Cox transformation requires strictly positive values.")
        
        if lambda_ is None or lambda_ == 0:
            return np.log(X)
        
        return (X**lambda_ - 1) / lambda_
        

    def _yeo_johnson_transform(self, X, lambda_):
        out = np.empty_like(X, dtype=float)
        pos = X >= 0
        neg = ~pos

        if abs(lambda_) < np.spacing(1.):
            out[pos] = np.log1p(X[pos])
        else:
            out[pos] = ((X[pos] + 1) ** lambda_ - 1) / lambda_

        if abs(lambda_ - 2) < np.spacing(1.):
            out[neg] = -np.log1p(-X[neg])
        else:
            out[neg] = -((-X[neg] + 1) ** (2 - lambda_) - 1) / (2 - lambda_)

        return out
    