---
sidebar_position: 1
---

# Mean change models

``` python
import fastcpd
from numpy import concatenate
from numpy.random import normal, multivariate_normal

covariance_mat = [[100, 0, 0], [0, 100, 0], [0, 0, 100]]
data = concatenate((multivariate_normal([0, 0, 0], covariance_mat, 300),
                    multivariate_normal([50, 50, 50], covariance_mat, 400),
                    multivariate_normal([2, 2, 2], covariance_mat, 300)))
fastcpd.mean(data)
```
