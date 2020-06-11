from ipywidgets import interact
import numpy as np
import math
import hashlib
import scipy.stats as stats
import json 
import nbinteract as nbi
from sklearn import svm
from collections import Counter
from sklearn.neural_network import MLPClassifier

opts1 = {
    'title': "Scatterplot of Errors",
    'ylabel': "Error",
    'xlabel': "Data Point",
    'ylim': (0, 500)}

def import_all():
    from ipywidgets import interact
    import numpy as np
    import math
    import hashlib
    import scipy.stats as stats
    import json 
    import nbinteract as nbi
    from sklearn import svm
    from collections import Counter
    from sklearn.neural_network import MLPClassifier

class CountMinSketch:
    def __init__(self, eps, delta):
        self.eps = eps
        self.delta = delta
        self.w = math.ceil(np.exp(1) / eps)
        self.d = math.ceil(np.log(1 / delta))
        self.tables = np.zeros((self.d, self.w))
        self.backup = {}

    def compute_hash(self, value, table_no):
        fn = hashlib.md5()
        inp = str(value) + str(0) + str(table_no)
        fn.update(inp.encode())
        out = int(fn.hexdigest(), 16)
        return out % self.w

    def count(self, value):
        if str(value) in self.backup: 
            self.backup[str(value)] = self.backup[str(value)] + 1
        else:
            self.backup[str(value)] = 1
        for i in range(self.d):
            j = self.compute_hash(value, i)
            self.tables[i][j] = self.tables[i][j] + 1

    def estimate(self, value):
        ests = []
        for i in range(self.d):
            j = self.compute_hash(value, i)
            ests.append(self.tables[i][j])
        return min(ests)

    def real_estimate(self, value):
        if str(value) in self.backup: return self.backup[str(value)]
        return -1

    def compute_size(self):
        size = 0
        for key in self.backup:
            size += abs(self.backup[key])
        return size

    def save_counts(self, count_filename='counts.txt', actual_filename='backups.txt'):
        np.savetxt(count_filename, self.tables)
        with open(actual_filename, 'w') as fp: json.dump(self.backup, fp)

    def load_counts(self, count_filename='counts.txt', actual_filename='backups.txt'):
        with open(actual_filename, 'r') as fp: 
            temp = json.load(fp)
            self.backup = temp
        self.tables = np.loadtxt(count_filename)
        
def load_data(cms, data):
  for el in data:
    cms.count(el)
def generate_sample(n=1000, dist='uniform', loc=0, scale=100, lambda_=5, s=100):
    if dist == 'uniform':
        float_sample = stats.uniform.rvs(loc, scale, n)
        return [int(el) for el in float_sample]
    if dist == 'zipf':
        float_sample = stats.zipf.rvs(a, size=n)
        return [(el) for el in float_sample]
    if dist == 'exp':
        float_sample = planck.rvs(lambda_, size=n)
        return [int(el) for el in float_sample]
    if dist == 'lognorm':
        float_sample = lognorm.rvs(s=s, size=n)
        return [int(el) for el in float_sample]
    if dist == 'geometric':
        float_sample =  geom.rvs(p, size=n)
        return [int(el) for el in float_sample]
    elif dist == 'normal':
        float_sample = stats.norm.rvs(loc, scale, n)
        return [int(el) for el in float_sample]
    else:
        return -1
    
# VISUALIZE ERROR PROBABILITY VS DELTA

# compute empirical probability of error exceeding the threshold
def compute_error_prob(cms, data, n):
  err = []
  for el in data:
    err.append(cms.estimate(el) - cms.real_estimate(el))
  avg_err = sum(err) / len(err)
  max_err = max(err)
  exceed = 0
  for el in err:
    if el > cms.eps * n:
      exceed += 1
  p = exceed / len(err)
  return p, avg_err, max_err, err

# run experiments on 10 values of delta interpolated between (min_delta, max_delta) and compute array of corresponding error probabilities
def error_prob_vs_delta(n=100000, eps=0.4, min_delta=0.01, max_delta=0.1):
  deltas = np.linspace(min_delta, max_delta, 10).tolist()
  # deltas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
  ps = []
  for delta in deltas:
    probs = 0
    # Average probabilities across 3 trials
    for i in range(3):
      cms = CountMinSketch(eps, delta)
      dt = generate_sample(n)
      load_data(cms, dt)
      p, avg_err, max_err, err = compute_error_prob(cms, dt, n)
      probs += p
    probs /= 3
    ps.append(probs)
  return deltas, ps

# graphing helper function
def graph_error_prob_vs_delta(deltas, ps, filename="p_vs_delta.png"):
  plt.scatter(deltas, ps)
  plt.show()
    
def generate_new_sample(eps, delta, distribution, n): 
    global sample
    sample = generate_sample(n=n, dist=distribution)
    return sample

# COUNT MIN ACROSS DISTS
def get_sample(eps, delta, distribution, n): 
    return generate_sample(n=n, dist=distribution)

def get_y_errors(xs, n, eps, delta, distribution):
    n = n
    threshold = eps * n

    mean = 0
    sd = 100
    sample = xs

    cms = CountMinSketch(eps, delta)
    load_data(cms, sample)

    p, avg_err, max_err, err = compute_error_prob(cms, sample, n)

    print("Average Error: " + str(avg_err))
    print("Maximum Error: " + str(max_err))
    print("Acceptable Threshold: " + str(threshold))
    print("Proportion of Errors Exceeding Threshold: " + str(p))

    return err

def get_data_for_hist_errors(n, eps, delta, distribution):
    
    n = n
    threshold = eps * n

    mean = 0
    sd = 100
    sample = generate_new_sample(eps, delta, distribution, n)
    cms = CountMinSketch(eps, delta)
    load_data(cms, sample)

    p, avg_err, max_err, err = compute_error_prob(cms, sample, n)

    print("Average Error: " + str(avg_err))
    print("Maximum Error: " + str(max_err))
    print("Acceptable Threshold: " + str(threshold))
    print("Proportion of Errors Exceeding Threshold: " + str(p))

    return err

def generate_hist_sample(n, distribution, loc=0, scale=10000, lambda_=5, s=1, a=6.5):
    import scipy.stats as stats
    dist = distribution
    if dist == 'uniform':
        float_sample = stats.uniform.rvs(loc, scale, n)
        return [int(el) for el in float_sample]
    if dist == 'zipf':
        float_sample = stats.zipf.rvs(a, size=n)
        return [int(el) for el in float_sample]
    if dist == 'exp':
        float_sample = stats.planck.rvs(lambda_, size=n)
        return [int(el) for el in float_sample]
    if dist == 'lognorm':
        float_sample = stats.lognorm.rvs(s=s, size=n)
        return [int(el) for el in float_sample]
    if dist == 'geometric':
        float_sample =  stats.geom.rvs(p, size=n)
        return [int(el) for el in float_sample]
    elif dist == 'normal':
        float_sample = stats.norm.rvs(loc, scale, n)
        return [int(el) for el in float_sample]
    else:
        return -1    
    


opts2 = {
    'title': "Distribution of Data",
    'ylabel': "Count",
    'xlabel': "Data Point",}

opts3 = {
    'title': "Distribution of errors",
    'ylabel': "Count",
    'xlabel': "Error Magnitude",}

# COUNT MIN ACROSS SDS
def get_sample_sd(eps, delta, n, sd): 
    return generate_sample(n=n, dist="normal", scale=sd)

def get_y_errors_sd(xs, n, eps, delta, sd):
    n = n
    threshold = eps * n

    mean = 0
    sd = 100
    sample = xs

    cms = CountMinSketch(eps, delta)
    load_data(cms, sample)

    p, avg_err, max_err, err = compute_error_prob(cms, sample, n)

    print("Average Error: " + str(avg_err))
    print("Maximum Error: " + str(max_err))
    print("Acceptable Threshold: " + str(threshold))
    print("Proportion of Errors Exceeding Threshold: " + str(p))

    return err
def get_data_for_hist_errors_sd(n, eps, delta, sd):
    n = n
    threshold = eps * n

    mean = 0
    sd = 100
    sample = get_sample_sd(eps, delta, n, sd)
    cms = CountMinSketch(eps, delta)
    load_data(cms, sample)

    p, avg_err, max_err, err = compute_error_prob(cms, sample, n)

    print("Average Error: " + str(avg_err))
    print("Maximum Error: " + str(max_err))
    print("Acceptable Threshold: " + str(threshold))
    print("Proportion of Errors Exceeding Threshold: " + str(p))

    return err

opts4 = {
    'title': "Scatterplot of Errors",
    'ylabel': "Error",
    'xlabel': "Data Point",
    'ylim': (0, 500),}

opts5 = {
    'title': "Distribution of errors",
    'ylabel': "Count",
    'xlabel': "Error Magnitude",}

class LearnedCountMinSketch:
    def __init__(self, eps, delta, train_data):
        self.eps = eps
        self.delta = delta
        self.cms = CountMinSketch(eps, delta)

        # set model
        X_train = train_data[0]
        Y_train = train_data[1]
        self.model = MLPClassifier(hidden_layer_sizes=(30, 40))
        self.model.fit(X_train.reshape(-1, 1), np.ravel(Y_train))
        self.perfect = {}

    def count(self, value):
        if (self.model.predict(np.array([value]).reshape(-1, 1)) == 1):
            if str(value) in self.perfect:
                self.perfect[str(value)] = self.perfect[str(value)] + 1
            else:
                self.perfect[str(value)] = 1
        else:
            self.cms.count(value)

    def estimate(self, value):
        if (self.model.predict(np.array([value]).reshape(-1, 1)) == 1):
            if str(value) in self.perfect: return self.perfect[str(value)]
            return 0
        else:
            return self.cms.estimate(value)

    def real_estimate(self, value):
        if str(value) in self.perfect: return self.perfect[str(value)]
        if str(value) in self.cms.backup: return self.cms.backup[str(value)]
        return -1

    def compute_size(self):
        size = 0
        for key in self.cms.backup:
            size += abs(self.cms.backup[key])
        for key in self.perfect:
            size += abs(self.perfect[key])
        return size
    
# label data --> threshold is what proportion of data points you want to call "heavy hitters"
def label_sample(sample, p = 0.05):
    n = len(sample)
    n_distinct = len(Counter(sample).keys())
    num = int(n_distinct * p)
    X_train = np.array(sample)
    Y_train = np.zeros_like(X_train)
    hh = Counter(sample).most_common(num)
    hh = set([el[0] for el in hh])
    for i in range(n):
        if X_train[i] in hh:
            Y_train[i] = 1
    return X_train, Y_train, hh

# VISUALIZE ML LEARNED COUNT MIN SKETCH

import numpy as np
import hashlib
import json 

class CountMinSketch:
    def __init__(self, eps, delta):
        self.eps = eps
        self.delta = delta
        self.w = math.ceil(np.exp(1) / eps)
        self.d = math.ceil(np.log(1 / delta))
        self.tables = np.zeros((self.d, self.w))
        self.backup = {}

    def compute_hash(self, value, table_no):
        fn = hashlib.md5()
        inp = str(value) + str(0) + str(table_no)
        fn.update(inp.encode())
        out = int(fn.hexdigest(), 16)
        return out % self.w

    def count(self, value):
        if str(value) in self.backup: 
            self.backup[str(value)] = self.backup[str(value)] + 1
        else:
            self.backup[str(value)] = 1
        for i in range(self.d):
            j = self.compute_hash(value, i)
            self.tables[i][j] = self.tables[i][j] + 1

    def estimate(self, value):
        ests = []
        for i in range(self.d):
            j = self.compute_hash(value, i)
            ests.append(self.tables[i][j])
        return min(ests)

    def real_estimate(self, value):
        if str(value) in self.backup: return self.backup[str(value)]
        return -1

    def compute_size(self):
        size = 0
        for key in self.backup:
            size += abs(self.backup[key])
        return size

    def save_counts(self, count_filename='counts.txt', actual_filename='backups.txt'):
        np.savetxt(count_filename, self.tables)
        with open(actual_filename, 'w') as fp: json.dump(self.backup, fp)

    def load_counts(self, count_filename='counts.txt', actual_filename='backups.txt'):
        with open(actual_filename, 'r') as fp: 
            temp = json.load(fp)
            self.backup = temp
        self.tables = np.loadtxt(count_filename)

import scipy.stats as stats

def generate_sample(n=1000, dist='uniform', loc=0, scale=1000, lambda_=5, s=1):
  if dist == 'uniform':
    float_sample = stats.uniform.rvs(loc, scale, n)
    return [int(el) for el in float_sample]
  if dist == 'zipf':
    float_sample = stats.zipf.rvs(loc + 1, size=n)
    return [int(el) for el in float_sample]
  if dist == 'exp':
    float_sample = stats.planck.rvs(lambda_, size=n)
    return [int(el) for el in float_sample]
  if dist == 'lognorm':
    float_sample = stats.lognorm.rvs(s=scale, size=n)
    return [int(el) for el in float_sample]
  if dist == 'geometric':
    float_sample =  stats.geom.rvs(p, size=n)
    return [int(el) for el in float_sample]
  elif dist == 'normal':
    float_sample = stats.norm.rvs(loc, scale, n)
    return [int(el) for el in float_sample]
  else:
    return -1


def compute_error_prob(cms, data, n):
  err = []
  for el in data:
    err.append(cms.estimate(el) - cms.real_estimate(el))
  avg_err = sum(err) / len(err)
  max_err = max(err)
  exceed = 0
  for el in err:
    if el > cms.eps * n:
      exceed += 1
  return exceed / len(err), err

def load_data(cms, data):
  for el in data:
    cms.count(el)

def run_experiment():
  n = 1000
  eps = 0.01
  min_delta = 0.01
  max_delta = 0.1
  deltas = np.linspace(min_delta, max_delta, 10).tolist()
  # deltas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
  ps = []
  for delta in deltas:
    probs = 0
    for i in range(3):
      cms = CountMinSketch(eps, delta)
      print(cms.w)
      dt = generate_sample(n)
      load_data(cms, dt)
      p, err = compute_error_prob(cms, dt, n)
      print(err)
      probs += p
    probs /= 3
    ps.append(probs)
  return deltas, ps, err

# VISUALIZE ERROR PROBABILITY VS DELTA

# compute empirical probability of error exceeding the threshold
def compute_error_prob_ml(cms, data, n):
  err = []
  for el in data:
    err.append(cms.estimate(el) - cms.real_estimate(el))
  avg_err = sum(err) / len(err)
  max_err = max(err)
  exceed = 0
  for el in err:
    if el > cms.eps * n:
      exceed += 1
  p = exceed / len(err)
  return p, avg_err, max_err, err

# run experiments on 10 values of delta interpolated between (min_delta, max_delta) and compute array of corresponding error probabilities
def error_prob_vs_delta(n=100000, eps=0.4, min_delta=0.01, max_delta=0.1):
  deltas = np.linspace(min_delta, max_delta, 10).tolist()
  # deltas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
  ps = []
  for delta in deltas:
    probs = 0
    # Average probabilities across 3 trials
    for i in range(3):
      cms = CountMinSketch(eps, delta)
      dt = generate_sample(n)
      load_data(cms, dt)
      p, avg_err, max_err, err = compute_error_prob(cms, dt, n)
      probs += p
    probs /= 3
    ps.append(probs)
  return deltas, ps

# graphing helper function
def graph_error_prob_vs_delta(deltas, ps, filename="p_vs_delta.png"):
  plt.scatter(deltas, ps)
  plt.show()

    
# compute empirical probability of error exceeding the threshold
def compute_error_prob(cms, data, n):
  err = []
  for el in data:
    err.append(cms.estimate(el) - cms.real_estimate(el))
  avg_err = sum(err) / len(err)
  max_err = max(err)
  exceed = 0
  for el in err:
    if el > cms.eps * n:
      exceed += 1
  p = exceed / len(err)
  return p, avg_err, max_err, err
  #return err

# run experiments on 10 values of delta interpolated between (min_delta, max_delta) and compute array of corresponding error probabilities
def error_prob_vs_delta(n=100000, eps=0.4, min_delta=0.01, max_delta=0.1):
  deltas = np.linspace(min_delta, max_delta, 10).tolist()
  # deltas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
  ps = []
  for delta in deltas:
    probs = 0
    # Average probabilities across 3 trials
    for i in range(3):
      cms = CountMinSketch(eps, delta)
      dt = generate_sample(n)
      load_data(cms, dt)
      p, avg_err, max_err, err = compute_error_prob(cms, dt, n)
      probs += p
    probs /= 3
    ps.append(probs)
  return deltas, ps

# graphing helper function
def graph_error_prob_vs_delta(deltas, ps, filename="p_vs_delta.png"):
  plt.scatter(deltas, ps)
  plt.show()
    
def opt_1_normal(sd):
    eps = 0.01
    delta = 0.05
    n = 1000
    threshold = eps * n
    mean = 0
    p = 0.2
    sample = generate_sample(n=n, dist='normal', loc=mean, scale=sd)
    X_tr, Y_tr, hh = label_sample(sample, p)

    cms = CountMinSketch(eps, delta)

    load_data(cms, sample)
    p, avg_err, max_err, err = compute_error_prob(cms, sample, n)
    return err

def opt_1_learned(sd):
    eps = 0.01
    delta = 0.05
    n = 1000
    threshold = eps * n
    mean = 0
    
    p = 0.2
    sample = generate_sample(n=n, dist='normal', loc=mean, scale=sd)
    X_tr, Y_tr, hh = label_sample(sample, p)

    lcms = LearnedCountMinSketch(eps, delta, [X_tr, Y_tr])

    load_data(lcms, sample)
    p, avg_err, max_err, err = compute_error_prob(lcms, sample, n)
    return err

class RuleCountMinSketch:
    def __init__(self, eps, delta, hh):
        self.eps = eps
        self.delta = delta
        self.cms = CountMinSketch(eps, delta)
        self.hh = hh
        self.perfect = {}

    def count(self, value):
        if value in self.hh:
            if str(value) in self.perfect:
                self.perfect[str(value)] = self.perfect[str(value)] + 1
            else:
                self.perfect[str(value)] = 1
        else:
            self.cms.count(value)

    def estimate(self, value):
        if (value in self.hh):
            if str(value) in self.perfect: return self.perfect[str(value)]
            return 0
        else:
            return self.cms.estimate(value)

    def real_estimate(self, value):
        if str(value) in self.perfect: return self.perfect[str(value)]
        if str(value) in self.cms.backup: return self.cms.backup[str(value)]
        return -1

    def compute_size(self):
        size = 0
        for key in self.cms.backup:
            size += abs(self.cms.backup[key])
        for key in self.perfect:
            size += abs(self.perfect[key])
        return size
    
def opt_2(sd):
    eps = 0.01
    delta = 0.05
    n = 1000
    threshold = eps * n
    mean = 0
    p = 0.2
    sample = generate_sample(n=n, dist='normal', loc=mean, scale=sd)
    X_tr, Y_tr, hh = label_sample(sample, p)

    rcms = RuleCountMinSketch(eps, delta, hh)

    load_data(rcms, sample)
    p, avg_err, max_err, err = compute_error_prob(rcms, sample, n)
    return err
