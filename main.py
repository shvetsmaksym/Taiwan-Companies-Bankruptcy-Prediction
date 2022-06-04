from imblearn.under_sampling import NearMiss 
nr = NearMiss()
X_near, Y_near = nr.fit_sample(x.to_numpy(), y.to_numpy().ravel())
c = Counter(Y_near)
for out, _ in c.items():
  points = where(y == out)[0]
  pyplot.scatter(X_near[points, 0], X_near[points, 1], out=str(out))
pyplot.legend()
pyplot.show()