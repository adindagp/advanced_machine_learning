import numpy as np
from scipy.spatial import distance as dist


class NearestNeighbors:
    def __init__(self,
                 n_neighbors = 5 ):
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def predict(self, data_tes):
        distances = []
        for i, x in enumerate(self.x):
            dist_val = dist.euclidean(data_tes, x)
            distances.append((dist_val, self.y[i]))
        distances.sort()
        nearest_classes = [item[1] for item in distances[:self.n_neighbors]]
        predicted_class = max(set(nearest_classes), key=nearest_classes.count)
        return predicted_class

x_data = [(10,50),(8,60),(7,40),(0,10),(1,0)]
y_data = [0,0,0,1,1]

knn = NearestNeighbors(n_neighbors=3)
knn.fit(x_data, y_data)

new_point = (5, 30)
predicted_class = knn.predict(new_point)
print("Predicted class:", predicted_class)

